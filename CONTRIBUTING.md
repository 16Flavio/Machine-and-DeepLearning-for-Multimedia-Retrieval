Bienvenue dans le guide de contribution du projet "Machine and Deep Learning for Multimedia Retrieval" (I-ILIA-014) pour le Groupe 05 (Marie Canfyn et Flavio Drogo).

Ce document explique comment nous allons organiser notre code, utiliser Git sans nous marcher sur les pieds, et les étapes clés pour mener à bien ce projet d'ici notre présentation du 15 juin 2026 à 11h10.

---

## 1. Organisation du dépôt

Notre projet est divisé en plusieurs dossiers pour séparer l'exploration, la logique métier et le déploiement Cloud (SaaS):

* **`notebooks/`** : Pour les tests, l'exploration des données et le prototypage.
* **`src/`** : Le code source propre (fonctions, classes, scripts finaux).
* **`data/`** : Les données (ignorées par Git). L'indexation locale se fera ici et ne sera pas hébergée sur la ressource de déploiement.


* 
**`deploy/`** : Les fichiers nécessaires au déploiement (Dockerfile).


* 
**`docs/`** : Le rapport de 20 pages et le manuel d'utilisation (à rendre pour le 1er juin 2026).



---

## 2. Comment travailler avec Git (Workflow)

Pour éviter les conflits, nous ne coderons **jamais** directement sur la branche `main`. Nous utiliserons des branches pour chaque nouvelle fonctionnalité (feature branch).

### Étape 1 : Récupérer les dernières mises à jour

Avant de commencer à travailler, assurez-vous d'avoir la dernière version du code :

```bash
git checkout main
git pull origin main

```

### Étape 2 : Créer une branche de travail

Créez une branche explicite selon ce sur quoi vous travaillez :

```bash
git checkout -b feature/nom-de-la-fonctionnalite
# Exemple : git checkout -b feature/indexation-sift

```

### Étape 3 : Travailler et sauvegarder (Commit)

Faites vos modifications. Ensuite, ajoutez les fichiers modifiés et créez un commit avec un message clair :

```bash
git add nom_du_fichier.py   # Ou "git add ." pour tout ajouter
git commit -m "Ajout de la fonction d'extraction SIFT"

```

### Étape 4 : Envoyer sur GitHub

Envoyez votre branche sur le dépôt distant :

```bash
git push origin feature/nom-de-la-fonctionnalite

```

### Étape 5 : Pull Request (PR)

Sur GitHub, ouvrez une **Pull Request** pour fusionner votre branche vers `main`. L'autre membre du groupe relit le code et valide (Merge). Une fois fusionnée, vous pouvez supprimer la branche.

---

## 3. Gestion des dépendances avec `uv`

Pour garantir que nos environnements locaux (ceux de Marie et Flavio) et notre conteneur Docker sur le Cloud utilisent exactement les mêmes versions de bibliothèques, nous utilisons **`uv`**. C'est un gestionnaire de paquets et d'environnements ultra-rapide (écrit en Rust) qui crée un fichier `uv.lock` robuste.

### Installation de `uv` (à faire une seule fois)

Si vous ne l'avez pas encore installé sur votre machine locale :

* **Mac/Linux :** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows :** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Initialiser et activer l'environnement

À la racine du projet, créez l'environnement virtuel :

```bash
uv venv

```

Puis activez-le (à faire à chaque fois que vous ouvrez un nouveau terminal pour travailler sur le projet) :

* **Mac/Linux :** `source .venv/bin/activate`
* **Windows :** `.venv\Scripts\activate`

### Ajouter de nouvelles dépendances

Pour installer un nouveau package (cela mettra automatiquement à jour nos fichiers `pyproject.toml` et `uv.lock`) :

```bash
uv add faiss-cpu flask numpy

```

*⚠️ Attention pour PyTorch :* L'installation de PyTorch (obligatoire pour nos descripteurs CNN et ViT ) nécessite souvent de spécifier la source pour exploiter correctement votre GPU ou CPU. Utilisez cette commande pour bien cibler la version :

```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118

```

### Synchroniser le projet après un `git pull`

Si votre binôme a ajouté de nouvelles dépendances sur GitHub, mettez à jour votre environnement local en une seule commande :

```bash
uv sync

```

---

## 4. Organisation du travail : Roadmap et Validation

Voici l'ordre chronologique recommandé pour développer le projet pas à pas.

### Phase 1 : Partie I - Moteur Unimodal (Images)

* **Par où commencer :** Le notebook `notebooks/01_descriptor_tests.ipynb`.
* **Tâches :**
1. Explorer la base "Cars" (10 classes, 14 167 images).


2. Tester différents descripteurs. **Attention :** Il est obligatoire de choisir au moins deux descripteurs deep learning (un CNN et un ViT) parmi nos 3 meilleurs descripteurs. On peut réduire la résolution pour SIFT.


3. Implémenter le script `src/part1_unimodal/indexer.py` pour indexer la base.




* **Validation :**
* L'utilisateur peut choisir la mesure de similarité (Euclidienne, FLANN, etc.).


* Tester les requêtes spécifiques au groupe 05 (R1 à R15 pour les classes 0, 2, 4, 6, 8).


* Afficher le Top-20, Top-50 et calculer Recall, Precision, AP, MAP et R-Precision. Remplir les tableaux de métriques.





### Phase 2 : Partie II - Moteur Multimodal

* 
**Par où commencer :** `notebooks/02_clip_exploration.ipynb` et la base de données Flickr8K (8 000 images, 5 textes par image).


* **Tâches :**
1. Télécharger Flickr8k.


2. Implémenter `src/part2_multimodal/clip_encoder.py` pour convertir images et textes en vecteurs via CLIP.


3. Créer la base de recherche rapide avec FAISS dans `src/part2_multimodal/faiss_index.py`.




* **Validation :**
* Réaliser des requêtes texte-vers-image et image-vers-texte.


* Évaluer les résultats avec Precision, Recall et mAP sur 3 images et 3 textes différents.


* 
*(Optionnel)* Comparer avec un autre modèle comme BLIP ou AlignVLM.





Phase 3 : Partie III - Déploiement Cloud (SaaS) (Optionnel) 

* **Par où commencer :** `src/web_app/app.py`.
* **Tâches :**
1. Créer l'interface web (Flask/Django/PHP) permettant de lancer les recherches et d'afficher les résultats (images, scores, courbes R/P).


2. Réduire la taille des descripteurs pour optimiser la mémoire.


3. Créer le `deploy/Dockerfile` qui gère les entrées/sorties.




* **Validation :**
* Le service est accessible via une adresse IP et un port.


* L'interface web est personnalisée (ex: page de login) et sécurisée.





### Phase 4 : Finalisation et Livrables

* **Rapport et Code :** Rédaction du rapport de 20 pages et du manuel d'utilisation. Soumission via Moodle avant le 1er juin 2026.


* 
**Présentation :** Préparation des slides pour la présentation en présentiel du 15 juin 2026 (15 minutes de présentation, 5 à 10 minutes de questions).
