# ==========================================================================
# PARTIE II - CNN et vision par ordinateur
# Classification d'images reelles avec reseaux de neurones convolutionnels
# Dataset : Digits (sklearn.datasets.load_digits), 1797 images reelles de
#           chiffres manuscrits 8x8, redimensionnees en 32x32 (format
#           classique LeNet). Substitut leger et auto-suffisant de
#           MNIST/Fashion-MNIST (memes proprietes pedagogiques : images
#           en niveaux de gris, 10 classes, ecriture manuscrite reelle).
# ==========================================================================

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.transform import resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilise : {device}")

# --------------------------------------------------------------------------
# 1. Calculs manuels : taille de sortie en convolution et en pooling
# --------------------------------------------------------------------------
# Formule generale (par dimension) :
#   out = floor((in + 2*padding - kernel) / stride) + 1
def conv_output_size(in_size, kernel, padding=0, stride=1):
    return (in_size + 2 * padding - kernel) // stride + 1


manual_calcs = []
configs = [
    {"in_size": 32, "kernel": 5, "padding": 0, "stride": 1, "desc": "Conv 5x5, sans padding, stride 1 (image 32x32, ex. LeNet C1)"},
    {"in_size": 32, "kernel": 5, "padding": 2, "stride": 1, "desc": "Conv 5x5, padding=2 (sortie preservee : 'same')"},
    {"in_size": 28, "kernel": 3, "padding": 1, "stride": 2, "desc": "Conv 3x3, padding=1, stride=2 (sous-echantillonnage)"},
    {"in_size": 28, "kernel": 2, "padding": 0, "stride": 2, "desc": "Pooling 2x2, stride=2 (division par 2 de la resolution)"},
]
print("\n[CALCULS MANUELS] Taille de sortie = floor((in + 2p - k)/s) + 1")
for c in configs:
    out = conv_output_size(c["in_size"], c["kernel"], c["padding"], c["stride"])
    line = f"  {c['desc']}: in={c['in_size']} -> out={out}"
    print(line)
    manual_calcs.append({**c, "out_size": out})

# --------------------------------------------------------------------------
# 2. Implementations manuelles : correlation croisee 2D, max-pooling, avg-pooling
# --------------------------------------------------------------------------
def corr2d_manual(X, K):
    """Correlation croisee 2D 'from scratch' (pas de retournement du noyau,
    contrairement a la convolution mathematique stricte -- c'est bien la
    correlation croisee qui est utilisee dans les CNN modernes).
    X: tensor (H, W) ; K: tensor (kh, kw) -> sortie (H-kh+1, W-kw+1)."""
    h, w = X.shape
    kh, kw = K.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    Y = torch.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = (X[i:i + kh, j:j + kw] * K).sum()
    return Y


def maxpool2d_manual(X, pool_size, stride):
    h, w = X.shape
    ph, pw = pool_size
    out_h = (h - ph) // stride + 1
    out_w = (w - pw) // stride + 1
    Y = torch.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = X[i * stride:i * stride + ph, j * stride:j * stride + pw].max()
    return Y


def avgpool2d_manual(X, pool_size, stride):
    h, w = X.shape
    ph, pw = pool_size
    out_h = (h - ph) // stride + 1
    out_w = (w - pw) // stride + 1
    Y = torch.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = X[i * stride:i * stride + ph, j * stride:j * stride + pw].mean()
    return Y


# --- Verification face aux couches PyTorch -------------------------------
torch.manual_seed(0)
X_test = torch.rand(10, 10)
K_test = torch.rand(3, 3)

# correlation croisee manuelle vs F.conv2d (avec le MEME noyau, sans biais)
manual_corr = corr2d_manual(X_test, K_test)
torch_corr = F.conv2d(X_test.unsqueeze(0).unsqueeze(0),
                       K_test.unsqueeze(0).unsqueeze(0)).squeeze()
corr_match = torch.allclose(manual_corr, torch_corr, atol=1e-5)

manual_max = maxpool2d_manual(X_test, (2, 2), 2)
torch_max = F.max_pool2d(X_test.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
max_match = torch.allclose(manual_max, torch_max, atol=1e-5)

manual_avg = avgpool2d_manual(X_test, (2, 2), 2)
torch_avg = F.avg_pool2d(X_test.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze()
avg_match = torch.allclose(manual_avg, torch_avg, atol=1e-5)

print("\n[VERIFICATION] Implementations manuelles vs PyTorch :")
print(f"  Correlation croisee 2D : identique a F.conv2d -> {corr_match}")
print(f"  Max-pooling            : identique a F.max_pool2d -> {max_match}")
print(f"  Average-pooling        : identique a F.avg_pool2d -> {avg_match}")

# --------------------------------------------------------------------------
# 3. Preparation des donnees (Digits, redimensionnees en 32x32)
# --------------------------------------------------------------------------
digits = load_digits()
images_8x8 = digits.images  # (1797, 8, 8), valeurs 0-16
labels = digits.target

images_32 = np.stack([resize(img, (32, 32), anti_aliasing=True) for img in images_8x8])
images_32 = images_32.astype(np.float32)
# Normalisation : centrage/reduction sur l'ensemble (statistiques globales)
mean_, std_ = images_32.mean(), images_32.std()
images_32 = (images_32 - mean_) / std_

X_train, X_temp, y_train, y_temp = train_test_split(
    images_32, labels, test_size=0.4, random_state=SEED, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

print(f"\n[INFO] Dataset Digits (32x32) : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


def to_img_tensor(X_, y_):
    Xt = torch.tensor(X_, dtype=torch.float32).unsqueeze(1)  # (N,1,32,32)
    yt = torch.tensor(y_, dtype=torch.long)
    return Xt, yt


X_train_t, y_train_t = to_img_tensor(X_train, y_train)
X_val_t, y_val_t = to_img_tensor(X_val, y_val)
X_test_t, y_test_t = to_img_tensor(X_test, y_test)

# Visualisation de quelques exemples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"label={y_train[i]}")
    ax.axis("off")
plt.suptitle("Exemples du dataset Digits (redimensionne 32x32)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p2_dataset_samples.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 4. CNN parametrable de type LeNet (pour les etudes d'ablation)
# --------------------------------------------------------------------------
class ConfigurableCNN(nn.Module):
    """CNN inspire de LeNet, dont les hyperparametres architecturaux sont
    exposes pour permettre l'etude comparative demandee (padding, stride,
    type de pooling, nombre de filtres, presence d'une convolution 1x1)."""

    def __init__(self, n_filters1=6, n_filters2=16, padding=0, stride=1,
                 pooling="max", use_1x1=False, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters1, kernel_size=5, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(n_filters1, n_filters2, kernel_size=5, padding=padding, stride=stride)
        self.use_1x1 = use_1x1
        if use_1x1:
            self.conv1x1 = nn.Conv2d(n_filters2, n_filters2, kernel_size=1)

        if pooling == "max":
            self.pool = nn.MaxPool2d(2, 2)
        elif pooling == "avg":
            self.pool = nn.AvgPool2d(2, 2)
        else:
            raise ValueError("pooling doit etre 'max' ou 'avg'")

        self.relu = nn.ReLU()

        # Calcul dynamique de la taille en sortie des convolutions pour
        # dimensionner correctement la premiere couche dense (evite les
        # erreurs de dimension lorsque padding/stride changent)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 32)
            out = self._features(dummy)
            flat_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def _features(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        if self.use_1x1:
            x = self.relu(self.conv1x1(x))
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_activations(self, x):
        """Renvoie aussi les cartes de caracteristiques intermediaires,
        pour la visualisation et l'interpretation des representations."""
        a1 = self.relu(self.conv1(x))
        p1 = self.pool(a1)
        a2 = self.relu(self.conv2(p1))
        p2 = self.pool(a2)
        flat = p2.view(p2.size(0), -1)
        out = self.fc3(self.relu(self.fc2(self.relu(self.fc1(flat)))))
        return out, {"conv1": a1, "pool1": p1, "conv2": a2, "pool2": p2}


class SimpleMLPBaseline(nn.Module):
    """MLP simple sur image aplatie, pour comparaison directe avec le CNN."""

    def __init__(self, in_dim=32 * 32, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# --------------------------------------------------------------------------
# 5. Boucle d'entrainement generique (classification multi-classe)
# --------------------------------------------------------------------------
def train_classifier(model, n_epochs=25, lr=1e-3, batch_size=64):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr, ytr = X_train_t.to(device), y_train_t.to(device)
    Xva, yva = X_val_t.to(device), y_val_t.to(device)
    n = Xtr.shape[0]

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_out = model(Xva)
            val_loss = criterion(val_out, yva).item()
            val_acc = (val_out.argmax(1) == yva).float().mean().item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    train_time = time.time() - t0
    return model, history, train_time


def evaluate_on_test(model):
    model.eval()
    with torch.no_grad():
        out = model(X_test_t.to(device))
        preds = out.argmax(1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    n_params = sum(p.numel() for p in model.parameters())
    return acc, cm, n_params


# --------------------------------------------------------------------------
# 6. Modele de reference (LeNet-like) + comparaison avec MLP
# --------------------------------------------------------------------------
torch.manual_seed(SEED)
cnn_ref = ConfigurableCNN(n_filters1=6, n_filters2=16, padding=0, stride=1, pooling="max")
cnn_ref, hist_cnn_ref, t_cnn = train_classifier(cnn_ref, n_epochs=25)
acc_cnn, cm_cnn, params_cnn = evaluate_on_test(cnn_ref)
print(f"\n[CNN reference] test_acc={acc_cnn:.4f}  n_params={params_cnn}  temps={t_cnn:.1f}s")

torch.manual_seed(SEED)
mlp_baseline = SimpleMLPBaseline()
mlp_baseline, hist_mlp, t_mlp = train_classifier(mlp_baseline, n_epochs=25)
acc_mlp, cm_mlp, params_mlp = evaluate_on_test(mlp_baseline)
print(f"[MLP baseline]  test_acc={acc_mlp:.4f}  n_params={params_mlp}  temps={t_mlp:.1f}s")

plt.figure(figsize=(9, 5))
plt.plot(hist_cnn_ref["val_acc"], label="CNN (LeNet-like)")
plt.plot(hist_mlp["val_acc"], label="MLP simple")
plt.xlabel("Epoque")
plt.ylabel("Accuracy de validation")
plt.title("CNN vs MLP sur le meme dataset d'images (Digits 32x32)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p2_cnn_vs_mlp.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 7. Etude d'ablation : padding, stride, pooling, nb de filtres, conv 1x1
# --------------------------------------------------------------------------
ablation_configs = [
    {"name": "reference (k=5,p=0,s=1,max,f=6/16)", "padding": 0, "stride": 1, "pooling": "max", "n_filters1": 6, "n_filters2": 16, "use_1x1": False},
    {"name": "avec padding=2 (same)", "padding": 2, "stride": 1, "pooling": "max", "n_filters1": 6, "n_filters2": 16, "use_1x1": False},
    {"name": "stride=2 (sous-echantillonnage agressif)", "padding": 0, "stride": 2, "pooling": "max", "n_filters1": 6, "n_filters2": 16, "use_1x1": False},
    {"name": "average-pooling au lieu de max-pooling", "padding": 0, "stride": 1, "pooling": "avg", "n_filters1": 6, "n_filters2": 16, "use_1x1": False},
    {"name": "plus de filtres (f=16/32)", "padding": 0, "stride": 1, "pooling": "max", "n_filters1": 16, "n_filters2": 32, "use_1x1": False},
    {"name": "moins de filtres (f=4/8)", "padding": 0, "stride": 1, "pooling": "max", "n_filters1": 4, "n_filters2": 8, "use_1x1": False},
    {"name": "avec convolution 1x1 additionnelle", "padding": 0, "stride": 1, "pooling": "max", "n_filters1": 6, "n_filters2": 16, "use_1x1": True},
]

ablation_results = []
for cfg in ablation_configs:
    torch.manual_seed(SEED)
    m = ConfigurableCNN(n_filters1=cfg["n_filters1"], n_filters2=cfg["n_filters2"],
                         padding=cfg["padding"], stride=cfg["stride"],
                         pooling=cfg["pooling"], use_1x1=cfg["use_1x1"])
    m, hist, t_train = train_classifier(m, n_epochs=20)
    acc, cm, n_params = evaluate_on_test(m)
    ablation_results.append({
        "name": cfg["name"], "test_acc": acc, "n_params": n_params,
        "train_time_s": round(t_train, 2), "final_val_acc": hist["val_acc"][-1],
    })
    print(f"[ABLATION] {cfg['name']:45s} test_acc={acc:.4f}  n_params={n_params:6d}  t={t_train:.1f}s")

# Figure recapitulative de l'ablation
plt.figure(figsize=(10, 5))
names = [r["name"] for r in ablation_results]
accs = [r["test_acc"] for r in ablation_results]
plt.barh(names, accs, color="#4C72B0")
plt.xlabel("Accuracy (test)")
plt.title("Etude d'ablation architecturale du CNN (Digits)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p2_ablation_study.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 8. Visualisation des cartes de caracteristiques (feature maps)
# --------------------------------------------------------------------------
sample_idx = 0
sample_img = X_test_t[sample_idx:sample_idx + 1].to(device)
sample_label = y_test[sample_idx]
_, activations = cnn_ref.forward_with_activations(sample_img)

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
conv1_maps = activations["conv1"].detach().cpu()[0]
for i in range(6):
    axes[0, i].imshow(conv1_maps[i], cmap="viridis")
    axes[0, i].set_title(f"conv1 - filtre {i}")
    axes[0, i].axis("off")
conv2_maps = activations["conv2"].detach().cpu()[0]
for i in range(6):
    axes[1, i].imshow(conv2_maps[i], cmap="viridis")
    axes[1, i].set_title(f"conv2 - filtre {i}")
    axes[1, i].axis("off")
plt.suptitle(f"Cartes de caracteristiques pour un exemple (label reel = {sample_label})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p2_feature_maps.png"), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 9. Export des resultats
# --------------------------------------------------------------------------
results = {
    "dataset": "Digits (sklearn), redimensionne 8x8 -> 32x32",
    "n_samples": int(images_32.shape[0]),
    "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "manual_size_calculations": manual_calcs,
    "manual_vs_pytorch_match": {
        "cross_correlation": bool(corr_match),
        "max_pooling": bool(max_match),
        "avg_pooling": bool(avg_match),
    },
    "cnn_reference": {"test_acc": float(acc_cnn), "n_params": int(params_cnn), "train_time_s": round(t_cnn, 2)},
    "mlp_baseline": {"test_acc": float(acc_mlp), "n_params": int(params_mlp), "train_time_s": round(t_mlp, 2)},
    "ablation_study": ablation_results,
}

with open(os.path.join(os.path.dirname(__file__), "..", "results_partie2.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Resultats exportes dans results_partie2.json")
print("[INFO] Figures sauvegardees dans le dossier figures/")
