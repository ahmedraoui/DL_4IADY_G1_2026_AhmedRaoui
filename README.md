# Projet Deep Learning -- EMSI Casablanca (2025-2026)

Implementation, experimentation et analyse critique de trois familles d'architectures de deep learning -- **MLP**, **CNN**, et architectures recurrentes (**RNN / LSTM / GRU / Seq2Seq**) -- appliquees a trois types de donnees reelles : tabulaires, images, et sequences (langage / traduction).

## Idee generale du projet

Ce repository contient l'integralite du travail individuel realise pour l'evaluation finale du module Deep Learning :

- **Partie I** -- MLP sur donnees tabulaires reelles (Breast Cancer Wisconsin) : construction via `nn.Sequential` et via une classe personnalisee, comparaison de strategies d'initialisation, sauvegarde/rechargement, evaluation complete.
- **Partie II** -- CNN sur images reelles (dataset Digits, redimensionne 32x32) : implementation manuelle de la correlation croisee 2D et du pooling (verifiees face a PyTorch), etude d'ablation architecturale (padding, stride, pooling, filtres, conv 1x1), visualisation des cartes de caracteristiques, comparaison CNN vs MLP.
- **Partie III** -- Architectures recurrentes et Seq2Seq : comparaison RNN / LSTM / GRU sur un modele de langage, demonstration experimentale de l'explosion du gradient (BPTT) et de l'effet du gradient clipping, systeme encodeur-decodeur de traduction anglais-francais avec decodage glouton, beam search et evaluation BLEU.

Chaque partie est documentee en detail (theorie, methodologie, resultats chiffres, analyse critique, question de synthese) dans le rapport scientifique complet : `docs/Rapport_Projet_Deep_Learning_EMSI.pdf`.

**Note sur les donnees** : les Parties I et II utilisent des jeux reels via `scikit-learn` (Breast Cancer Wisconsin, Digits). La **Partie III utilise des corpus reels charges via Hugging Face `datasets`** : **IMDb** (`stanfordnlp/imdb`) pour le modele de langage et **Helsinki-NLP/opus-100** (paire `en-fr`) pour la traduction. Au premier lancement, ces datasets sont telecharges puis mis en cache localement (lancements suivants hors-ligne), exactement comme les fonctions `fetch_*` de scikit-learn. Installation requise : `pip install datasets`.

## Structure du repository

```
.
├── README.md                          <- ce fichier
├── notebooks/                         <- notebooks Jupyter executables (.ipynb)
│   ├── 01_MLP_breast_cancer.ipynb
│   ├── 02_CNN_digits.ipynb
│   ├── 03_RNN_LSTM_GRU_language_model.ipynb
│   └── 04_Seq2Seq_traduction.ipynb
├── src/                                <- code source complet, commente, en scripts .py autonomes
│   ├── mlp_breast_cancer.py
│   ├── cnn_digits.py
│   ├── lm_rnn_lstm_gru.py
│   └── seq2seq_translation.py
├── figures/                            <- les 11 figures generees (courbes, matrices de confusion, visualisations)
├── results_partie1.json                <- metriques chiffrees, Partie I
├── results_partie2.json                <- metriques chiffrees, Partie II
├── results_partie3_lm.json             <- metriques chiffrees, Partie III (modele de langage)
├── results_partie3_seq2seq.json        <- metriques chiffrees, Partie III (Seq2Seq)
└── docs/
    ├── Rapport_Projet_Deep_Learning_EMSI.pdf   <- rapport scientifique complet (lecture directe sur GitHub)
  
```

## Comment executer le projet

Chaque script de `src/` est autonome et executable directement :

```bash
pip install torch scikit-learn scikit-image numpy matplotlib datasets
python3 src/mlp_breast_cancer.py
python3 src/cnn_digits.py
python3 src/lm_rnn_lstm_gru.py
python3 src/seq2seq_translation.py
```

Les notebooks de `notebooks/` correspondent exactement au meme code, decoupe en cellules pour une lecture et une execution interactives. Les notebooks de la Partie III (03 et 04) doivent etre relances pour regenerer leurs sorties (corpus reels). Chaque script/notebook regenere les figures dans `figures/` et les metriques dans les fichiers `results_partieX.json` a la racine.

## Ressources volumineuses

Aucun fichier de ce projet ne depasse quelques centaines de kilo-octets (le plus volumineux etant le rapport au format docx, ~650 Ko) : **Git LFS n'est pas necessaire** pour ce repository.

## Reproductibilite

Toutes les graines aleatoires (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) sont fixees a 42 dans chaque script, garantissant des resultats identiques a chaque execution.
