# Manuel utilisateur — Moteur de recherche multimédia

**Projet I-ILIA-014 · Machine and Deep Learning for Multimedia Retrieval**
*Groupe 05 — Marie Canfyn, Flavio Drogo — FPMs, Année académique 2025-2026*

---

## 1. Présentation

L'application met à disposition deux moteurs de recherche multimédia
accessibles via une interface web unique :

| Moteur | Base | Type de requête | Modèle |
|---|---|---|---|
| Unimodal **à deux niveaux** | *Cars* (10 000 images, 10 marques / 96 modèles) | Image | HOG, SIFT-BoVW, ORB-BoVW, ConvNeXt-Base, DINOv2 ViT-B/14 |
| Multimodal | *Flickr8K* (8 000 images, 5 légendes / image) | Texte ou image | CLIP ViT-B/32 + FAISS |

Le service est exposé en HTTP par un serveur Flask sur le **port 5000**
et accepte également des requêtes programmatiques (API REST).

---

## 2. Pré-requis

### 2.1 Lancement par Docker (recommandé)

- Docker ≥ 20.10
- 6 Go de RAM libre
- 5 Go d'espace disque pour l'image

### 2.2 Lancement en mode développement

- Python 3.10 – 3.12
- [`uv`](https://docs.astral.sh/uv/) (gestionnaire de paquets)
- (Optionnel) GPU CUDA pour ré-extraire les descripteurs profonds

---

## 3. Démarrage rapide (Docker)

```bash
git clone https://github.com/16Flavio/Machine-and-DeepLearning-for-Multimedia-Retrieval.git
cd Machine-and-DeepLearning-for-Multimedia-Retrieval

# (1) Récupération des poids et descripteurs pré-calculés depuis Hugging Face
#     voir la section README pour les instructions détaillées

# (2) Construction et démarrage du conteneur
docker build -t mir-app .
docker run -d -p 5000:5000 --name mir mir-app
```

Ouvrez ensuite [http://localhost:5000](http://localhost:5000) dans votre
navigateur.

Pour arrêter et nettoyer :

```bash
docker stop mir && docker rm mir
```

---

## 4. Démarrage en mode développement

```bash
git clone https://github.com/16Flavio/Machine-and-DeepLearning-for-Multimedia-Retrieval.git
cd Machine-and-DeepLearning-for-Multimedia-Retrieval

uv sync                       # installation des dépendances
python app.py                 # démarrage du serveur Flask
```

Trois ressources doivent être présentes localement avant le premier
démarrage :

```
data/raw/Cars/                                  images du corpus Cars
data/raw/Flickr8k_dataset/Images/               images du corpus Flickr8k
data/raw/Flickr8k_dataset/captions.txt          légendes Flickr8k
models/*.pth                                    poids ConvNeXt + DINOv2
results/features/*.npy                          descripteurs pré-calculés
results/features/CLIP_flickr_*                  embeddings CLIP + filenames
```

Si l'un de ces fichiers manque, la console affichera un message clair
(`FileNotFoundError`) indiquant lequel.

---

## 5. Utilisation de l'interface web

L'interface comporte **deux pages** accessibles via le menu de navigation
en haut à droite.

### 5.1 Onglet *Moteur de recherche d'image uni modal à deux niveaux*

![Page unimodale](../../Site_Internet/Images/) <!-- placeholder visuel -->

#### Étape 1 — Sélection de l'image de requête

Deux modes sont disponibles :

- **Choisir une image existante** : commencez à taper le nom du fichier
  (ex. `0_1_BMW_X3_207`) dans la barre de recherche. L'autocomplétion
  proposera les images disponibles dans la galerie *Cars*.
- **Téléverser une image** : cliquez sur *Parcourir* pour soumettre une
  image inédite (formats acceptés : JPG, PNG).

L'image de requête s'affiche immédiatement à gauche.

#### Étape 2 — Choix du ou des descripteurs

Six descripteurs peuvent être combinés librement :

| Bouton | Description |
|---|---|
| **ORB** | Bag-of-Visual-Words sur points-clés ORB (binaires, K=256) |
| **SIFT** | Bag-of-Visual-Words sur points-clés SIFT (K=256) |
| **HOG** | Histogramme de gradients orientés (descripteur global) |
| **ConvNext Classifier** | ConvNeXt-Base entraîné par classification, dim=2048 |
| **ConvNext Metric** | ConvNeXt-Base entraîné par metric learning (ArcFace), dim=2048 |
| **ViT Metric** | DINOv2 ViT-B/14 entraîné par metric learning, dim=1024 |

> Si plusieurs descripteurs sont sélectionnés, leurs classements sont
> fusionnés via *Reciprocal Rank Fusion* (RRF, c=60).

#### Étape 3 — Choix de la mesure de similarité

| Bouton | Adapté pour |
|---|---|
| **Euclidienne** | Embeddings profonds, descripteurs L2-normalisés |
| **Cosinus** | Embeddings profonds (équivaut à la corrélation) |
| **Chi carré** | Histogrammes BoVW, HOG |
| **Bhattacharyya** | Histogrammes BoVW, HOG |

#### Étape 4 — Profondeur de la recherche (Top-K)

Saisissez un entier positif (typiquement **20** ou **50**, conformément
à l'énoncé). La page affichera autant de vignettes que demandé.

#### Étape 5 — Lancement

Cliquez sur **Lancer la recherche**. Après quelques centaines de
millisecondes (jusqu'à quelques secondes pour une image téléversée
nécessitant l'inférence d'un modèle profond), trois zones se
remplissent :

1. **Galerie de résultats** : vignettes triées par similarité
   décroissante avec le score de similarité affiché.
2. **Courbe Précision/Rappel** : tracée avec Plotly, accompagnée de
   la valeur d'*Average Precision* (AP) pour cette requête.
3. **Indicateurs** : nombre d'images pertinentes dans la galerie,
   classe attendue.

#### Visualisation 3D (UMAP)

Le bouton **Visualisation 3D (UMAP)** ouvre un nuage de points
interactif (Plotly) projetant la galerie complète dans un espace 3D
appris par UMAP. Les couleurs correspondent aux classes
*marque + modèle*. Cette vue permet de juger visuellement la séparation
des classes dans l'espace latent.

### 5.2 Onglet *Moteur de recherche multi modal*

#### Recherche texte → image

1. Saisissez une description en anglais dans la barre de recherche
   (ex. *« A dog running on the beach »*).
2. Choisissez la profondeur Top-K.
3. Cliquez sur **Lancer la recherche**.

L'application encode le texte avec **CLIP ViT-B/32**, puis effectue une
recherche par produit scalaire (similarité cosinus sur vecteurs
L2-normalisés) sur la galerie Flickr8K indexée par **FAISS**.

#### Recherche image → image

Téléversez une image quelconque ; CLIP encodera l'image et retournera
les images de Flickr8K dont l'embedding est le plus proche dans
l'espace latent partagé.

#### Évaluation automatique

Lorsque la requête textuelle est suffisamment précise (les *content
words* recoupent intégralement les légendes d'au moins une image),
l'application calcule automatiquement une courbe Précision/Rappel et
une *Average Precision*, en considérant comme pertinentes toutes les
images dont les légendes contiennent les mots de la requête.

---

## 6. Utilisation programmatique de l'API

Toutes les fonctionnalités sont également exposées via une API REST.

### 6.1 Recherche unimodale par nom de fichier

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
        "filename": "0_1_BMW_X3_207.jpg",
        "descriptors": ["ConvNextClassifier"],
        "metric": "Cosinus",
        "top_k": 50
      }'
```

### 6.2 Recherche unimodale par image téléversée

```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@/chemin/vers/ma_voiture.jpg" \
  -F "descriptors=ConvNextClassifier" \
  -F "descriptors=VitMetric" \
  -F "metric=Cosinus" \
  -F "top_k=20"
```

### 6.3 Recherche multimodale texte → image

```bash
curl -X POST http://localhost:5000/api/search_multimodal \
  -H "Content-Type: application/json" \
  -d '{"text": "A dog running on the beach", "top_k": 10}'
```

### 6.4 Recherche multimodale image → image

```bash
curl -X POST http://localhost:5000/api/search_multimodal \
  -F "image=@/chemin/vers/photo.jpg" \
  -F "top_k=10"
```

### 6.5 Format de réponse

Toutes les routes de recherche renvoient :

```json
{
  "results": [
    {"filename": "0_1_BMW_X3_201.jpg", "score": 0.987},
    {"filename": "0_1_BMW_X3_212.jpg", "score": 0.961},
    "..."
  ],
  "pr_curve": {
    "recall":     [0.0, 0.01, 0.02, "..."],
    "precision":  [1.0, 1.0, 1.0,  "..."],
    "average_precision": 0.94,
    "total_relevant": 162
  }
}
```

### 6.6 Récupérer une image servie

```bash
curl http://localhost:5000/cars_dataset/0_1_BMW_X3_207.jpg --output car.jpg
curl http://localhost:5000/flickr_dataset/1000268201_693b08cb0e.jpg --output flickr.jpg
```

---

## 7. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| Page blanche / 404 | Serveur Flask non démarré | Vérifier les logs ; relancer `python app.py` ou `docker logs mir` |
| `Vocabulaire ORB introuvable` | KMeans BoVW non généré | Lancer `python src/part1_unimodal/extract_classical.py` |
| `Checkpoint manquant` | Poids ConvNeXt/DINOv2 absents | Télécharger les poids depuis Hugging Face (cf. README) |
| `CLIP gallery not found` | Embeddings Flickr8K non calculés | Lancer `python src/part2_multimodal/extract_clip.py` |
| Recherche très lente sur image téléversée | Premier appel = chargement modèle | Les requêtes suivantes seront rapides (cache mémoire) |
| `top_k invalide` | Champ Top-K vide ou non numérique | Saisir un entier positif |

---

## 8. Crédits

- **Modèles pré-entraînés** : ConvNeXt-Base (Meta AI), DINOv2 (Meta AI),
  CLIP ViT-B/32 (OpenAI).
- **Bibliothèques** : PyTorch, timm, transformers, FAISS, OpenCV,
  scikit-image, scikit-learn, UMAP, Flask.
- **Auteurs** : Canfyn Marie & Drogo Flavio, FPMs 2026.
