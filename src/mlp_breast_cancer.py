# ==========================================================================
# PARTIE I - MLP et ingenierie PyTorch
# Classification supervisee sur donnees tabulaires reelles
# Dataset : Breast Cancer Wisconsin (sklearn.datasets, donnees reelles
#           de diagnostic de cancer du sein, 569 patients, 30 variables)
# ==========================================================================
#
# Ce script couvre l'ensemble du travail demande pour la Partie I :
#   1. Preparation des donnees (nettoyage, encodage, normalisation, split)
#   2. Deux implementations d'un MLP : nn.Sequential et classe personnalisee
#   3. Inspection des parametres via named_parameters() et state_dict()
#   4. Comparaison de 3 strategies d'initialisation (gaussienne, constante,
#      Xavier)
#   5. Sauvegarde / rechargement du meilleur modele
#   6. Gestion explicite du device (CPU/GPU)
#   7. Evaluation : accuracy, precision, recall, F1, matrice de confusion
#
# Toutes les figures sont enregistrees dans ../figures/ et les resultats
# numeriques dans results_partie1.json, afin d'etre repris dans le rapport.

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 0. Device : on verifie quel materiel est disponible et on s'assure que
#    modele ET donnees seront places sur le meme device (source classique
#    d'erreurs runtime en PyTorch).
# --------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilise : {device}")

# --------------------------------------------------------------------------
# 1. Preparation des donnees
# --------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target  # y in {0,1} : 0 = malin, 1 = benin (deja encode)
feature_names = data.feature_names
class_names = data.target_names

print(f"[INFO] Dataset Breast Cancer Wisconsin : X={X.shape}, y={y.shape}")
print(f"[INFO] Classes : {class_names} (deja encodees en 0/1, pas de NaN)")

# Nettoyage : on verifie l'absence de valeurs manquantes (dataset propre par
# construction, mais la verification fait partie de la rigueur methodologique)
assert not np.isnan(X).any(), "Valeurs manquantes detectees"

# Split train / val / test (60 / 20 / 20), stratifie pour conserver
# l'equilibre des classes dans chaque sous-ensemble
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

# Normalisation : fit UNIQUEMENT sur le train pour eviter toute fuite
# d'information (data leakage) vers val/test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"[INFO] Tailles : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


def to_tensor(X_, y_):
    return (torch.tensor(X_, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32).unsqueeze(1))


X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

n_features = X_train.shape[1]

# --------------------------------------------------------------------------
# 2. Deux implementations du MLP
# --------------------------------------------------------------------------

# --- Version A : nn.Sequential -------------------------------------------
def build_mlp_sequential(n_in, n_hidden=(64, 32), n_out=1):
    """MLP construit avec nn.Sequential : rapide a ecrire, mais moins
    flexible (pas de logique conditionnelle dans le forward, pas de
    branches multiples)."""
    layers = []
    dims = [n_in] + list(n_hidden)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], n_out))
    return nn.Sequential(*layers)


# --- Version B : classe personnalisee -------------------------------------
class MLPCustom(nn.Module):
    """MLP defini via une classe personnalisee heritant de nn.Module.
    Avantage par rapport a nn.Sequential : on peut definir un forward()
    arbitraire (sauts de connexion, branches, traitements intermediaires,
    impression de formes intermediaires pour le debogage, etc.)."""

    def __init__(self, n_in, n_hidden=(64, 32), n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], n_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------------------------------
# 3. Inspection des parametres : named_parameters() et state_dict()
# --------------------------------------------------------------------------
model_demo = MLPCustom(n_features).to(device)

print("\n[INSPECTION] named_parameters() :")
total_params = 0
param_report_lines = []
for name, p in model_demo.named_parameters():
    n = p.numel()
    total_params += n
    line = f"  {name:20s} shape={tuple(p.shape)} requires_grad={p.requires_grad} n_params={n}"
    print(line)
    param_report_lines.append(line)
print(f"  TOTAL parametres entrainables : {total_params}")

print("\n[INSPECTION] state_dict() (cles uniquement) :")
state_keys = list(model_demo.state_dict().keys())
print(" ", state_keys)

# --------------------------------------------------------------------------
# 4. Strategies d'initialisation : gaussienne, constante, Xavier
# --------------------------------------------------------------------------
def init_gaussian(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.01)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


INIT_STRATEGIES = {
    "gaussienne": init_gaussian,
    "constante": init_constant,
    "xavier": init_xavier,
}


# --------------------------------------------------------------------------
# 5. Boucle d'entrainement generique
# --------------------------------------------------------------------------
def train_model(model, n_epochs=150, lr=1e-2, weight_decay=1e-4):
    model = model.to(device)
    Xtr, ytr = X_train_t.to(device), y_train_t.to(device)
    Xva, yva = X_val_t.to(device), y_val_t.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr)
        loss = criterion(logits, ytr)
        loss.backward()          # retropropagation : calcule dL/dW pour
                                   # chaque parametre via la chaine de derivees
        optimizer.step()          # mise a jour des poids : W <- W - lr * dL/dW

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = criterion(val_logits, yva).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == yva).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return model, history, best_state, best_val_loss


# --------------------------------------------------------------------------
# 6. Comparaison des 3 strategies d'initialisation (sur la classe MLPCustom)
# --------------------------------------------------------------------------
results_init = {}
histories_init = {}

for name, init_fn in INIT_STRATEGIES.items():
    torch.manual_seed(SEED)
    model = MLPCustom(n_features)
    model.apply(init_fn)
    trained_model, history, best_state, best_val_loss = train_model(model, n_epochs=150)
    histories_init[name] = history
    results_init[name] = {
        "best_val_loss": best_val_loss,
        "final_val_acc": history["val_acc"][-1],
    }
    print(f"[INIT={name}] best_val_loss={best_val_loss:.4f} final_val_acc={history['val_acc'][-1]:.4f}")

# Figure comparant les 3 strategies d'initialisation
plt.figure(figsize=(9, 5))
for name, hist in histories_init.items():
    plt.plot(hist["val_loss"], label=f"val_loss ({name})")
plt.xlabel("Epoque")
plt.ylabel("Perte de validation (BCE)")
plt.title("Influence de la strategie d'initialisation sur la convergence (MLP, Breast Cancer)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_init_strategies.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 7. Entrainement final : on retient Xavier (meilleure strategie attendue)
#    et on compare nn.Sequential vs classe personnalisee
# --------------------------------------------------------------------------
torch.manual_seed(SEED)
model_seq = build_mlp_sequential(n_features)
model_seq.apply(init_xavier)
model_seq, hist_seq, best_state_seq, best_loss_seq = train_model(model_seq, n_epochs=200)

torch.manual_seed(SEED)
model_custom = MLPCustom(n_features)
model_custom.apply(init_xavier)
model_custom, hist_custom, best_state_custom, best_loss_custom = train_model(model_custom, n_epochs=200)

plt.figure(figsize=(9, 5))
plt.plot(hist_seq["train_loss"], label="train_loss (nn.Sequential)")
plt.plot(hist_seq["val_loss"], label="val_loss (nn.Sequential)")
plt.plot(hist_custom["train_loss"], "--", label="train_loss (classe personnalisee)")
plt.plot(hist_custom["val_loss"], "--", label="val_loss (classe personnalisee)")
plt.xlabel("Epoque")
plt.ylabel("Perte (BCE)")
plt.title("nn.Sequential vs classe personnalisee (init Xavier)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_sequential_vs_custom.png"), dpi=150)
plt.close()

# On retient le meilleur des deux modeles (sur la perte de validation) comme
# "meilleur modele" a sauvegarder
if best_loss_custom <= best_loss_seq:
    best_overall_state = best_state_custom
    best_overall_name = "MLPCustom"
    best_overall_arch = lambda: MLPCustom(n_features)
else:
    best_overall_state = best_state_seq
    best_overall_name = "nn.Sequential"
    best_overall_arch = lambda: build_mlp_sequential(n_features)

print(f"[INFO] Meilleur modele retenu : {best_overall_name}")

# --------------------------------------------------------------------------
# 8. Sauvegarde puis rechargement du meilleur modele
# --------------------------------------------------------------------------
CKPT_PATH = os.path.join(os.path.dirname(__file__), "best_mlp.pt")
torch.save({
    "state_dict": best_overall_state,
    "arch_name": best_overall_name,
    "n_features": n_features,
}, CKPT_PATH)
print(f"[INFO] Modele sauvegarde dans {CKPT_PATH}")

# Rechargement dans une INSTANCE FRAICHE pour bien demontrer le cycle complet
checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=True)
reloaded_model = best_overall_arch().to(device)
reloaded_model.load_state_dict(checkpoint["state_dict"])
reloaded_model.eval()
print("[INFO] Modele recharge avec succes depuis le checkpoint.")

# --------------------------------------------------------------------------
# 9. Evaluation finale sur le jeu de TEST (jamais vu pendant l'entrainement)
# --------------------------------------------------------------------------
with torch.no_grad():
    test_logits = reloaded_model(X_test_t.to(device))
    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
    test_preds = (test_probs > 0.5).astype(int)

y_test_np = y_test.astype(int)

acc = accuracy_score(y_test_np, test_preds)
prec = precision_score(y_test_np, test_preds)
rec = recall_score(y_test_np, test_preds)
f1 = f1_score(y_test_np, test_preds)
cm = confusion_matrix(y_test_np, test_preds)

print("\n[EVALUATION TEST]")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-score  : {f1:.4f}")
print(f"  Matrice de confusion :\n{cm}")

# Figure : matrice de confusion
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Matrice de confusion - MLP (Breast Cancer, jeu de test)")
plt.colorbar()
plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predit")
plt.ylabel("Reel")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_confusion_matrix.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 10. Export des resultats pour le rapport
# --------------------------------------------------------------------------
results = {
    "dataset": "Breast Cancer Wisconsin (sklearn)",
    "n_samples": int(X.shape[0]),
    "n_features": int(n_features),
    "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "total_trainable_params": int(total_params),
    "state_dict_keys": state_keys,
    "init_strategies_comparison": results_init,
    "best_overall_model": best_overall_name,
    "best_val_loss_sequential": float(best_loss_seq),
    "best_val_loss_custom": float(best_loss_custom),
    "test_metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    },
}

with open(os.path.join(os.path.dirname(__file__), "..", "results_partie1.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Resultats exportes dans results_partie1.json")
print("[INFO] Figures sauvegardees dans le dossier figures/")
