# ==========================================================================
# PARTIE III - RNN, LSTM, GRU et Seq2Seq
# Section A : modele de langage (prediction du prochain caractere)
#             comparaison RNN simple / LSTM / GRU, BPTT, gradient clipping
# Section B : systeme Seq2Seq encodeur-decodeur (traduction anglais->francais)
#             teacher forcing, decodage glouton, beam search, BLEU
#
# NOTE SUR LES DONNEES : le modele de langage est entraine sur un CORPUS REEL
# charge via Hugging Face `datasets` (IMDb, dataset cite dans l'enonce).
# Premier lancement : telechargement automatique puis mise en cache locale.
# Prerequis : pip install datasets.
# ==========================================================================

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilise : {device}")

# ==========================================================================
# SECTION A : MODELE DE LANGAGE (RNN vs LSTM vs GRU)
# ==========================================================================

from datasets import load_dataset

# --------------------------------------------------------------------------
# A.1 Chargement d'un CORPUS REEL via Hugging Face `datasets`
#     Dataset : IMDb (avis de films, corpus reel cite dans l'enonce).
#     Premier lancement : telechargement automatique puis mise en cache
#     locale (les lancements suivants sont hors-ligne). Necessite :
#         pip install datasets
# --------------------------------------------------------------------------
MAX_CHARS = 400_000          # plafond pour un temps d'entrainement raisonnable
LM_DATASET = "stanfordnlp/imdb"


def load_lm_corpus():
    ds = load_dataset(LM_DATASET, split="train")
    parts, total = [], 0
    for ex in ds:
        txt = ex["text"].replace("<br />", " ").replace("<br/>", " ").strip()
        if not txt:
            continue
        parts.append(txt)
        total += len(txt) + 1
        if total >= MAX_CHARS:
            break
    raw = "\n".join(parts)[:MAX_CHARS]
    return raw, LM_DATASET


CORPUS_LM, LM_SOURCE = load_lm_corpus()
print(f"[INFO] Corpus LM : {len(CORPUS_LM)} caracteres, "
      f"{len(set(CORPUS_LM))} caracteres uniques (source : {LM_SOURCE})")

# --------------------------------------------------------------------------
# A.2 Preparation des donnees au niveau caractere
# --------------------------------------------------------------------------
chars = sorted(list(set(CORPUS_LM)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)

encoded = torch.tensor([char_to_idx[c] for c in CORPUS_LM], dtype=torch.long)

SEQ_LEN = 60


def make_sequences(encoded_text, seq_len, stride=3):
    """Cree des sequences (input, target) par fenetre glissante (avec un
    pas 'stride' pour limiter la redondance entre sequences successives) :
    target est l'input decale d'un caractere (prediction du caractere
    suivant)."""
    inputs, targets = [], []
    for i in range(0, len(encoded_text) - seq_len, stride):
        inputs.append(encoded_text[i:i + seq_len])
        targets.append(encoded_text[i + 1:i + seq_len + 1])
    return torch.stack(inputs), torch.stack(targets)


# Corpus reel beaucoup plus volumineux que precedemment : on augmente le pas
# de la fenetre glissante et on plafonne le nombre de sequences pour garder
# un temps d'entrainement raisonnable (surtout sur CPU).
STRIDE = 5
MAX_SEQUENCES = 18000

X_all, Y_all = make_sequences(encoded, SEQ_LEN, stride=STRIDE)
if X_all.shape[0] > MAX_SEQUENCES:
    keep = torch.randperm(X_all.shape[0])[:MAX_SEQUENCES]
    keep, _ = torch.sort(keep)  # on garde l'ordre chronologique (split temporel)
    X_all, Y_all = X_all[keep], Y_all[keep]

n_total = X_all.shape[0]
n_train = int(0.85 * n_total)
X_train_lm, Y_train_lm = X_all[:n_train], Y_all[:n_train]
X_val_lm, Y_val_lm = X_all[n_train:], Y_all[n_train:]
print(f"[INFO] Sequences LM : train={n_train}, val={n_total - n_train} "
      f"(longueur={SEQ_LEN}, pas={STRIDE})")


# --------------------------------------------------------------------------
# A.3 Modele de langage recurrent generique (RNN / LSTM / GRU)
# --------------------------------------------------------------------------
class CharRNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=128, cell_type="lstm", n_layers=1):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if cell_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        else:
            raise ValueError("cell_type doit etre rnn, lstm ou gru")
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def train_lm(cell_type, n_epochs=60, lr=2e-3, clip_value=None, batch_size=64,
             optimizer_type="adam", data_override=None, hidden_dim=128, init_std=None):
    """Entraine un modele de langage caractere. Si clip_value est fourni,
    le gradient global est ecrete (clip_grad_norm_) avant chaque pas
    d'optimisation -- mecanisme standard pour stabiliser la BPTT.
    optimizer_type='sgd' (sans normalisation adaptative) est utilise pour
    bien mettre en evidence le phenomene d'explosion du gradient, masque
    par Adam dans la plupart des configurations courantes. init_std permet
    de forcer une initialisation a forte variance des poids recurrents
    (weight_hh), afin de provoquer deliberement une instabilite numerique
    typique de la BPTT sur des sequences longues."""
    torch.manual_seed(SEED)
    model = CharRNNLM(vocab_size, cell_type=cell_type, hidden_dim=hidden_dim).to(device)
    if init_std is not None:
        with torch.no_grad():
            for name, p in model.rnn.named_parameters():
                if "weight_hh" in name:
                    nn.init.normal_(p, mean=0.0, std=init_std)
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if data_override is None:
        Xtr, Ytr = X_train_lm.to(device), Y_train_lm.to(device)
        Xva, Yva = X_val_lm.to(device), Y_val_lm.to(device)
    else:
        Xtr, Ytr, Xva, Yva = [t.to(device) for t in data_override]
    n = Xtr.shape[0]

    history = {"train_loss": [], "val_loss": [], "val_perplexity": [],
               "grad_norm": [], "max_grad_norm": []}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, epoch_grad_norm, epoch_max_norm, n_batches = 0.0, 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()  # BPTT : le gradient est retropropage a travers
                              # les SEQ_LEN pas de temps deroules

            total_norm = torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None
            )).item()
            if not np.isfinite(total_norm):
                total_norm = 1e4  # sentinel pour representer une explosion numerique (NaN/inf)
            epoch_grad_norm += total_norm
            epoch_max_norm = max(epoch_max_norm, total_norm)

            if clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if not np.isfinite(loss.item()):
                break

        epoch_loss /= n_batches
        epoch_grad_norm /= n_batches

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(Xva)
            val_loss = criterion(val_logits.reshape(-1, vocab_size), Yva.reshape(-1)).item()
            val_loss_capped = val_loss if np.isfinite(val_loss) else 20.0
            val_perplexity = float(np.exp(min(val_loss_capped, 20)))  # cap pour eviter overflow numerique a l'affichage

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss_capped)
        history["val_perplexity"].append(val_perplexity)
        history["grad_norm"].append(epoch_grad_norm)
        history["max_grad_norm"].append(epoch_max_norm)

    train_time = time.time() - t0
    return model, history, train_time


def generate_text(model, start_str, length=200, temperature=0.8):
    model.eval()
    chars_idx = [char_to_idx.get(c, 0) for c in start_str]
    input_seq = torch.tensor([chars_idx], dtype=torch.long).to(device)
    hidden = None
    generated = start_str
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            probs = torch.softmax(logits[0, -1] / temperature, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            generated += idx_to_char[next_idx]
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)
    return generated


# --------------------------------------------------------------------------
# A.4 Comparaison RNN simple / LSTM / GRU (avec gradient clipping standard)
# --------------------------------------------------------------------------
lm_results = {}
lm_histories = {}
for cell_type in ["rnn", "lstm", "gru"]:
    model, history, t_train = train_lm(cell_type, n_epochs=20, clip_value=2.0)
    lm_histories[cell_type] = history
    n_params = sum(p.numel() for p in model.parameters())
    lm_results[cell_type] = {
        "final_val_loss": history["val_loss"][-1],
        "final_val_perplexity": history["val_perplexity"][-1],
        "n_params": n_params,
        "train_time_s": round(t_train, 2),
    }
    print(f"[LM cell={cell_type:4s}] val_loss={history['val_loss'][-1]:.3f}  "
          f"perplexity={history['val_perplexity'][-1]:.2f}  n_params={n_params}  t={t_train:.1f}s")
    if cell_type == "lstm":
        best_lm_model = model

seed_str = CORPUS_LM[:24]  # amorce tiree du debut du corpus reel
sample_generated = generate_text(best_lm_model, seed_str, length=180)
print(f"\n[GENERATION LSTM] {sample_generated}")

plt.figure(figsize=(9, 5))
for cell_type, hist in lm_histories.items():
    plt.plot(hist["val_perplexity"], label=f"{cell_type.upper()}")
plt.xlabel("Epoque")
plt.ylabel("Perplexite (validation)")
plt.title("Comparaison RNN / LSTM / GRU : perplexite sur le modele de langage")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p3_lm_perplexity_comparison.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# A.5 BPTT : croissance du gradient en fonction de la longueur de sequence
# --------------------------------------------------------------------------
# Demonstration "manuel de cours" du phenomene d'explosion/disparition du
# gradient : on calcule, pour un RNN simple, la norme du gradient total
# apres une seule passe avant/arriere, en fonction de la longueur T de la
# sequence deroulee, pour deux regimes d'initialisation des poids
# recurrents (weight_hh) :
#   - init "stable" (std proche de l'initialisation par defaut de PyTorch)
#   - init "instable" (std artificiellement elevee)
# Ce protocole isole proprement l'effet de la profondeur de la BPTT, sans
# les effets confondants d'un entrainement multi-epoques (ou la saturation
# de tanh peut masquer/attenuer le phenomene au fil des mises a jour).
def build_probe_rnn(std, hidden_dim=128, embed_dim=32, seed=1):
    torch.manual_seed(seed)
    emb = nn.Embedding(vocab_size, embed_dim)
    rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
    fc = nn.Linear(hidden_dim, vocab_size)
    with torch.no_grad():
        nn.init.normal_(rnn.weight_hh_l0, mean=0.0, std=std)
    return emb, rnn, fc


def gradient_norm_for_length(std, T, batch_size=4):
    emb, rnn, fc = build_probe_rnn(std)
    params = list(emb.parameters()) + list(rnn.parameters()) + list(fc.parameters())
    x = torch.randint(0, vocab_size, (batch_size, T))
    y = torch.randint(0, vocab_size, (batch_size, T))
    logits = fc(rnn(emb(x))[0])
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    for p in params:
        p.grad = None
    loss.backward()
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)).item()
    return norm if np.isfinite(norm) else float("inf")


T_VALUES = [10, 20, 40, 80, 160, 320]
STD_STABLE = 1.0 / np.sqrt(128)  # ~ initialisation par defaut (1/sqrt(hidden_dim))
STD_UNSTABLE = 1.3

grad_vs_T = {"stable": [], "unstable": []}
for T in T_VALUES:
    grad_vs_T["stable"].append(gradient_norm_for_length(STD_STABLE, T))
    grad_vs_T["unstable"].append(gradient_norm_for_length(STD_UNSTABLE, T))

print(f"\n[BPTT] Norme du gradient en fonction de la longueur de sequence T :")
print(f"  init stable   (std={STD_STABLE:.3f}) : {[round(v, 3) for v in grad_vs_T['stable']]}")
print(f"  init instable (std={STD_UNSTABLE:.3f}) : {grad_vs_T['unstable']}")

# Remplacement des +inf par une valeur plafond pour permettre le trace en
# echelle logarithmique (l'explosion reelle est notee dans le rapport)
PLOT_CAP = 1e18
grad_vs_T_plot = {
    "stable": grad_vs_T["stable"],
    "unstable": [min(v, PLOT_CAP) for v in grad_vs_T["unstable"]],
}

plt.figure(figsize=(8, 5))
plt.plot(T_VALUES, grad_vs_T_plot["stable"], "o-", label=f"init stable (std={STD_STABLE:.3f})", color="seagreen")
plt.plot(T_VALUES, grad_vs_T_plot["unstable"], "o-", label=f"init instable (std={STD_UNSTABLE})", color="crimson")
plt.yscale("log")
plt.xlabel("Longueur de la sequence deroulee T (pas de BPTT)")
plt.ylabel("Norme du gradient (echelle log)")
plt.title("Effet de la profondeur de la BPTT sur la norme du gradient (RNN simple)")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p3_bptt_gradient_growth.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# A.6 Gradient clipping : effet de l'ecretage sur un gradient explose
# --------------------------------------------------------------------------
# On reprend le cas instable a T=40 (gradient deja tres largement explose
# mais encore numeriquement fini) et on montre l'effet de
# nn.utils.clip_grad_norm_ : la DIRECTION du gradient est conservee, seule
# sa NORME est ramenee a la valeur maximale autorisee.
emb_demo, rnn_demo, fc_demo = build_probe_rnn(STD_UNSTABLE)
params_demo = list(emb_demo.parameters()) + list(rnn_demo.parameters()) + list(fc_demo.parameters())
x_demo = torch.randint(0, vocab_size, (4, 40))
y_demo = torch.randint(0, vocab_size, (4, 40))
logits_demo = fc_demo(rnn_demo(emb_demo(x_demo))[0])
loss_demo = nn.functional.cross_entropy(logits_demo.reshape(-1, vocab_size), y_demo.reshape(-1))
loss_demo.backward()
norm_before = torch.sqrt(sum((p.grad ** 2).sum() for p in params_demo if p.grad is not None)).item()
nn.utils.clip_grad_norm_(params_demo, max_norm=1.0)
norm_after = torch.sqrt(sum((p.grad ** 2).sum() for p in params_demo if p.grad is not None)).item()

print(f"\n[CLIPPING] Exemple a T=40, init instable (std={STD_UNSTABLE}) :")
print(f"  norme du gradient AVANT clipping : {norm_before:.4g}")
print(f"  norme du gradient APRES clip_grad_norm_(max_norm=1.0) : {norm_after:.4g}")
print(f"  -> sans clipping, la mise a jour des poids (W -= lr * grad) serait")
print(f"     numeriquement destructrice ; le clipping la ramene a une echelle")
print(f"     raisonnable tout en conservant la direction de descente.")

# --------------------------------------------------------------------------
# A.7 Export des resultats de la section A
# --------------------------------------------------------------------------
results_section_a = {
    "corpus_source": LM_SOURCE,
    "corpus_length_chars": len(CORPUS_LM),
    "vocab_size": vocab_size,
    "seq_len": SEQ_LEN,
    "rnn_lstm_gru_comparison": lm_results,
    "sample_generation_lstm": sample_generated,
    "bptt_gradient_growth": {
        "T_values": T_VALUES,
        "std_stable": float(STD_STABLE),
        "std_unstable": float(STD_UNSTABLE),
        "grad_norm_stable": [float(v) for v in grad_vs_T["stable"]],
        "grad_norm_unstable": [str(v) for v in grad_vs_T["unstable"]],
    },
    "gradient_clipping_example": {
        "T": 40,
        "std_unstable": float(STD_UNSTABLE),
        "norm_before_clip": float(norm_before) if np.isfinite(norm_before) else "inf",
        "norm_after_clip_max1": float(norm_after),
    },
}

with open(os.path.join(os.path.dirname(__file__), "..", "results_partie3_lm.json"), "w", encoding="utf-8") as f:
    json.dump(results_section_a, f, indent=2, ensure_ascii=False)

print("\n[INFO] Resultats de la section A exportes dans results_partie3_lm.json")
print("[INFO] Figures sauvegardees dans le dossier figures/")

