# ==========================================================================
# PARTIE III - Section B : systeme Seq2Seq encodeur-decodeur
# Traduction automatique simple anglais -> francais
#
# Corpus : Helsinki-NLP/opus-100 (paire en-fr) REEL, charge via Hugging Face
# `datasets`, filtre sur les phrases courtes. Premier lancement :
# telechargement + mise en cache. Prerequis : pip install datasets.
# Le pipeline (tokenisation, vocabulaire, padding, masquage, encodeur-
# decodeur, teacher forcing, decodage glouton/beam, BLEU) est inchange.
# ==========================================================================

import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilise : {device}")


# --------------------------------------------------------------------------
# B.1 Construction du corpus parallele original anglais -> francais
# --------------------------------------------------------------------------
import re
import unicodedata

from datasets import load_dataset

# --------------------------------------------------------------------------
# B.1 Chargement d'un CORPUS PARALLELE REEL via Hugging Face `datasets`
#     Dataset : Helsinki-NLP/opus-100, paire de langues 'en-fr'. Chaque
#     exemple est un dict {'translation': {'en': ..., 'fr': ...}}.
#     Premier lancement : telechargement + mise en cache. Necessite :
#         pip install datasets
# --------------------------------------------------------------------------
MT_DATASET = "Helsinki-NLP/opus-100"
MT_CONFIG = "en-fr"
MAX_PAIRS = 30000     # nb de paires courtes collectees (avant filtrage <unk>)
MIN_TOKENS = 2        # on evite les paires d'un seul mot (ex. "barking")
MAX_TOKENS = 4        # phrases tres courtes => alignement plus facile a apprendre


def _normalize(s):
    """Minuscule, accents conserves, ponctuation retiree, apostrophe -> espace
    (coherent avec la tokenisation par espaces : \"j'ai\" -> \"j ai\")."""
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = s.replace("\u2019", "'").replace("'", " ")
    s = re.sub(r"[^a-z\u00e0-\u00ff ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_translation_pairs():
    ds = load_dataset(MT_DATASET, MT_CONFIG, split="train")
    pairs, seen = [], set()
    for ex in ds:
        en = _normalize(ex["translation"]["en"])
        fr = _normalize(ex["translation"]["fr"])
        if not en or not fr:
            continue
        le, lf = len(en.split()), len(fr.split())
        if not (MIN_TOKENS <= le <= MAX_TOKENS) or not (MIN_TOKENS <= lf <= MAX_TOKENS):
            continue
        key = (en, fr)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((en, fr))
        if len(pairs) >= MAX_PAIRS:
            break
    random.shuffle(pairs)
    return pairs, f"{MT_DATASET} ({MT_CONFIG})"


corpus_pairs, CORPUS_SOURCE = load_translation_pairs()
print(f"[INFO] Corpus parallele REEL : {len(corpus_pairs)} paires "
      f"(anglais -> francais) -- source : {CORPUS_SOURCE}")
print("[INFO] Exemples :")
for p in corpus_pairs[:5]:
    print(f"  EN: {p[0]:35s} FR: {p[1]}")

# --------------------------------------------------------------------------
# B.2 Tokenisation et construction des vocabulaires (avec tokens speciaux)
# --------------------------------------------------------------------------
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


def tokenize(s):
    return s.lower().strip().split()


class Vocab:
    def __init__(self, sentences, min_freq=1):
        counter = Counter()
        for s in sentences:
            counter.update(tokenize(s))
        self.itos = [PAD, SOS, EOS, UNK] + [w for w, c in counter.items() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, sentence, add_eos=True):
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(sentence)]
        if add_eos:
            ids = ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[i]
            if w == EOS:
                break
            if w in (PAD, SOS):
                continue
            words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.itos)


# --------------------------------------------------------------------------
# Vocabulaire PLAFONNE (min_freq) : les mots rares (vus < MIN_FREQ fois) sont
# remplaces par <unk>. opus-100 est tres diversifie : sans plafond, le
# vocabulaire (~8000 mots) depasse le nombre de phrases d'entrainement et le
# modele n'apprend rien (perplexite enorme, BLEU ~ 0). On retire aussi les
# paires trop riches en <unk> pour donner un signal d'entrainement propre.
# --------------------------------------------------------------------------
MIN_FREQ = 3
MAX_UNK = 1   # nb max de tokens <unk> tolere par phrase (source ou cible)

_src_vocab = Vocab([p[0] for p in corpus_pairs], min_freq=MIN_FREQ)
_tgt_vocab = Vocab([p[1] for p in corpus_pairs], min_freq=MIN_FREQ)


def _n_unk(vocab, sentence):
    return sum(1 for w in tokenize(sentence) if w not in vocab.stoi)


corpus_pairs = [
    (en, fr) for en, fr in corpus_pairs
    if _n_unk(_src_vocab, en) <= MAX_UNK and _n_unk(_tgt_vocab, fr) <= MAX_UNK
]

# vocabulaire final reconstruit sur les paires effectivement conservees
src_vocab = Vocab([p[0] for p in corpus_pairs], min_freq=2)
tgt_vocab = Vocab([p[1] for p in corpus_pairs], min_freq=2)
print(f"[INFO] Paires conservees apres filtrage <unk> : {len(corpus_pairs)}")
print(f"[INFO] Taille vocabulaire source (EN) : {len(src_vocab)}")
print(f"[INFO] Taille vocabulaire cible  (FR) : {len(tgt_vocab)}")

# --------------------------------------------------------------------------
# B.3 Encodage, padding et masquage, split train/val/test
# --------------------------------------------------------------------------
MAX_LEN = max(max(len(tokenize(en)) for en, fr in corpus_pairs),
              max(len(tokenize(fr)) for en, fr in corpus_pairs)) + 1  # +1 pour <eos>


def encode_pad(vocab, sentence):
    ids = vocab.encode(sentence)
    ids = ids[:MAX_LEN]
    pad_len = MAX_LEN - len(ids)
    return ids + [vocab.stoi[PAD]] * pad_len


X = torch.tensor([encode_pad(src_vocab, en) for en, fr in corpus_pairs], dtype=torch.long)
Y = torch.tensor([encode_pad(tgt_vocab, fr) for en, fr in corpus_pairs], dtype=torch.long)

n = X.shape[0]
idx = torch.randperm(n)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
X_test, Y_test = X[test_idx], Y[test_idx]
test_pairs_ref = [corpus_pairs[i] for i in test_idx.tolist()]

print(f"[INFO] Longueur max (avec <eos>) : {MAX_LEN}")
print(f"[INFO] Split : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


# --------------------------------------------------------------------------
# B.4 Architecture encodeur-decodeur (GRU)
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        # src : (batch, seq_len)
        emb = self.embedding(src)
        outputs, hidden = self.gru(emb)
        return outputs, hidden  # hidden : (1, batch, hidden_dim) -> contexte condense


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, input_token, hidden):
        # input_token : (batch, 1)
        emb = self.embedding(input_token)
        output, hidden = self.gru(emb, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab_size, sos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.sos_idx = sos_idx
        self.device = device

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_len=MAX_LEN):
        batch_size = src.shape[0]
        _, hidden = self.encoder(src)

        input_token = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
        outputs = torch.zeros(batch_size, max_len, self.tgt_vocab_size, device=self.device)

        for t in range(max_len):
            logits, hidden = self.decoder.forward_step(input_token, hidden)
            outputs[:, t] = logits
            if tgt is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing : on injecte le VRAI token cible precedent
                # comme entree du pas suivant, plutot que la prediction du
                # modele -- accelere et stabilise l'apprentissage en debut
                # d'entrainement, au prix d'un decalage train/inference
                # (le modele ne voit ses propres erreurs qu'a l'inference).
                input_token = tgt[:, t:t + 1]
            else:
                input_token = logits.argmax(1, keepdim=True)
        return outputs


# --------------------------------------------------------------------------
# B.5 Boucle d'entrainement
# --------------------------------------------------------------------------
def train_seq2seq(n_epochs=200, lr=1e-3, batch_size=32, teacher_forcing_ratio=0.6):
    torch.manual_seed(SEED)
    encoder = Encoder(len(src_vocab))
    decoder = Decoder(len(tgt_vocab))
    model = Seq2Seq(encoder, decoder, len(tgt_vocab), tgt_vocab.stoi[SOS], device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD])  # masquage du padding

    Xtr, Ytr = X_train.to(device), Y_train.to(device)
    Xva, Yva = X_val.to(device), Y_val.to(device)
    n_tr = Xtr.shape[0]

    history = {"train_loss": [], "val_loss": [], "val_perplexity": []}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_tr)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_tr, batch_size):
            b_idx = perm[i:i + batch_size]
            src_b, tgt_b = Xtr[b_idx], Ytr[b_idx]
            optimizer.zero_grad()
            outputs = model(src_b, tgt_b, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(outputs.reshape(-1, len(tgt_vocab)), tgt_b.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_outputs = model(Xva, Yva, teacher_forcing_ratio=0.0)  # pas de teacher forcing en validation
            val_loss = criterion(val_outputs.reshape(-1, len(tgt_vocab)), Yva.reshape(-1)).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(float(np.exp(min(val_loss, 20))))

    train_time = time.time() - t0
    return model, history, train_time


# --------------------------------------------------------------------------
# B.6 Decodage glouton (greedy)
# --------------------------------------------------------------------------
def greedy_decode(model, src_sentence_ids, max_len=MAX_LEN):
    model.eval()
    src = src_sentence_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        _, hidden = model.encoder(src)
        input_token = torch.tensor([[tgt_vocab.stoi[SOS]]], device=device)
        result_ids = []
        for _ in range(max_len):
            logits, hidden = model.decoder.forward_step(input_token, hidden)
            next_id = logits.argmax(1).item()
            if next_id == tgt_vocab.stoi[EOS]:
                break
            result_ids.append(next_id)
            input_token = torch.tensor([[next_id]], device=device)
    return tgt_vocab.decode(result_ids)


# --------------------------------------------------------------------------
# B.7 Decodage par recherche en faisceau (beam search)
# --------------------------------------------------------------------------
def beam_search_decode(model, src_sentence_ids, beam_width=3, max_len=MAX_LEN):
    """Maintient les `beam_width` sequences partielles les plus probables
    (en log-vraisemblance cumulee) a chaque pas, plutot que de ne garder
    que le meilleur choix local comme le decodage glouton -- permet
    d'eviter certaines erreurs locales qui penalisent la suite de la
    sequence generee."""
    model.eval()
    src = src_sentence_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        _, hidden0 = model.encoder(src)
        # chaque candidat : (liste_ids, log_prob_cumulee, hidden_state, termine)
        beams = [([], 0.0, hidden0, False)]
        for _ in range(max_len):
            all_candidates = []
            for ids, score, hidden, done in beams:
                if done:
                    all_candidates.append((ids, score, hidden, done))
                    continue
                last_token = ids[-1] if ids else tgt_vocab.stoi[SOS]
                input_token = torch.tensor([[last_token]], device=device)
                logits, new_hidden = model.decoder.forward_step(input_token, hidden)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                topk_logp, topk_idx = log_probs.topk(beam_width)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    new_done = (idx == tgt_vocab.stoi[EOS])
                    new_ids = ids + ([] if new_done else [idx])
                    all_candidates.append((new_ids, score + lp, new_hidden, new_done))
            # normalisation par longueur pour ne pas favoriser systematiquement
            # les sequences courtes (biais classique du beam search)
            all_candidates.sort(key=lambda c: c[1] / max(len(c[0]), 1), reverse=True)
            beams = all_candidates[:beam_width]
            if all(done for _, _, _, done in beams):
                break
    best_ids, best_score, _, _ = beams[0]
    return tgt_vocab.decode(best_ids)


# --------------------------------------------------------------------------
# B.8 BLEU (implementation simple, BLEU-4 avec penalite de brievete)
# --------------------------------------------------------------------------
def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference, hypothesis, max_n=4):
    """BLEU d'une phrase avec lissage additif (Chen & Cherry, methode 1) :
    une precision nulle a l'ordre n est remplacee par un petit epsilon au
    lieu d'annuler tout le score -- indispensable pour des phrases courtes
    ou les 3-grammes/4-grammes manquent presque toujours."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if len(hyp_tokens) == 0:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        ref_counts = ngram_counts(ref_tokens, n)
        hyp_counts = ngram_counts(hyp_tokens, n)
        overlap = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        total = max(len(hyp_tokens) - n + 1, 0)
        if total == 0:
            p = 1.0                      # phrase plus courte que n : ordre ignore
        elif overlap == 0:
            p = 1.0 / (2.0 * total)      # lissage additif
        else:
            p = overlap / total
        log_p += (1.0 / max_n) * np.log(p)
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else float(
        np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return float(bp * np.exp(log_p))


def corpus_bleu(references, hypotheses, max_n=4):
    """BLEU au NIVEAU CORPUS (definition standard) : compteurs de n-grammes
    ecretes agreges sur tout le corpus + penalite de brievete globale,
    plutot qu'une moyenne de BLEU par phrase. C'est cette agregation qui
    permet au score de refleter de petites differences entre decodeurs
    (glouton vs beam), invisibles avec une moyenne par phrase saturee a 0."""
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len_total = 0
    hyp_len_total = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = tokenize(ref), tokenize(hyp)
        ref_len_total += len(r)
        hyp_len_total += len(h)
        for n in range(1, max_n + 1):
            rc = ngram_counts(r, n)
            hc = ngram_counts(h, n)
            clipped[n - 1] += sum(min(c, rc.get(g, 0)) for g, c in hc.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    precisions = []
    for n in range(max_n):
        if totals[n] == 0:
            precisions.append(1e-9)
        elif clipped[n] == 0:
            precisions.append(1.0 / (2.0 * totals[n]))   # lissage additif
        else:
            precisions.append(clipped[n] / totals[n])
    log_p = float(np.mean([np.log(p) for p in precisions]))
    bp = 1.0 if hyp_len_total >= ref_len_total else float(
        np.exp(1 - ref_len_total / max(hyp_len_total, 1)))
    return float(bp * np.exp(log_p))


# --------------------------------------------------------------------------
# B.9 Entrainement et evaluation
# --------------------------------------------------------------------------
model, history, train_time = train_seq2seq(n_epochs=45, lr=1e-3, teacher_forcing_ratio=0.6)
print(f"\n[SEQ2SEQ] Entrainement termine en {train_time:.1f}s -- "
      f"val_loss final={history['val_loss'][-1]:.3f}, perplexite={history['val_perplexity'][-1]:.2f}")

plt.figure(figsize=(9, 5))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.xlabel("Epoque")
plt.ylabel("Perte (cross entropy, padding masque)")
plt.title("Entrainement du systeme Seq2Seq (encodeur-decodeur GRU, teacher forcing)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p3_seq2seq_training_curve.png"), dpi=150)
plt.close()

# --- Comparaison decodage glouton vs beam search sur le jeu de test ------
greedy_hyps, beam_hyps, refs = [], [], []
example_rows = []
for i in range(len(X_test)):
    src_ids = X_test[i]
    en_sentence, fr_reference = test_pairs_ref[i]
    greedy_out = greedy_decode(model, src_ids)
    beam_out = beam_search_decode(model, src_ids, beam_width=3)
    greedy_hyps.append(greedy_out)
    beam_hyps.append(beam_out)
    refs.append(fr_reference)
    if i < 8:
        example_rows.append({"en": en_sentence, "reference_fr": fr_reference,
                              "greedy": greedy_out, "beam": beam_out})

bleu_greedy = corpus_bleu(refs, greedy_hyps)
bleu_beam = corpus_bleu(refs, beam_hyps)

print(f"\n[EVALUATION TEST] BLEU (decodage glouton) : {bleu_greedy:.4f}")
print(f"[EVALUATION TEST] BLEU (beam search, k=3)  : {bleu_beam:.4f}")
print("\n[EXEMPLES DE TRADUCTION]")
for row in example_rows:
    print(f"  EN     : {row['en']}")
    print(f"  REF FR : {row['reference_fr']}")
    print(f"  GLOUTON: {row['greedy']}")
    print(f"  BEAM   : {row['beam']}")
    print()

# Figure comparative BLEU
plt.figure(figsize=(5, 4))
plt.bar(["Glouton", "Beam search (k=3)"], [bleu_greedy, bleu_beam], color=["#4C72B0", "#55A868"])
plt.ylabel("Score BLEU (corpus de test)")
plt.title("Decodage glouton vs beam search")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p3_bleu_greedy_vs_beam.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# B.10 Export des resultats
# --------------------------------------------------------------------------
results_section_b = {
    "corpus_source": CORPUS_SOURCE,
    "corpus_size": len(corpus_pairs),
    "src_vocab_size": len(src_vocab),
    "tgt_vocab_size": len(tgt_vocab),
    "max_len": MAX_LEN,
    "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "train_time_s": round(train_time, 2),
    "final_val_loss": history["val_loss"][-1],
    "final_val_perplexity": history["val_perplexity"][-1],
    "bleu_greedy": bleu_greedy,
    "bleu_beam_k3": bleu_beam,
    "example_translations": example_rows,
}

with open(os.path.join(os.path.dirname(__file__), "..", "results_partie3_seq2seq.json"), "w", encoding="utf-8") as f:
    json.dump(results_section_b, f, indent=2, ensure_ascii=False)

print("\n[INFO] Resultats exportes dans results_partie3_seq2seq.json")
print("[INFO] Figures sauvegardees dans le dossier figures/")

