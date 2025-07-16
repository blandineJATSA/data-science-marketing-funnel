# data-science-marketing-funnel

# 🛒 Analyse Funnel Conversion E-commerce

## 🎯 Objectif

Ce projet a pour objectif d'**analyser le funnel de conversion Freemium ➔ Payant en contexte e-commerce / SaaS** afin de :
- Détecter les points de friction dans le parcours utilisateur.
- Identifier des leviers d'amélioration du taux de conversion.
- Explorer l'impact des catégories, marques, prix, jour/heure sur le funnel.
- Analyser le comportement des paniers abandonnés et les cohortes d'utilisateurs.

---

## 🗂️ Données

Les données proviennent de [Kaggle: Ecommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).

**Variables disponibles :**
- `event_time`: datetime de l'événement
- `event_type`: type d'événement (`view`, `cart`, `purchase`, `remove_from_cart`)
- `product_id`: identifiant produit
- `category_id`: identifiant catégorie
- `category_code`: nom lisible de catégorie
- `brand`: marque
- `price`: prix du produit
- `user_id`: identifiant utilisateur
- `user_session`: identifiant de session utilisateur

Les données sont transformées et stockées au format Parquet pour optimiser la manipulation dans DuckDB/Polars.

---

## 📝 Plan d'analyse

Le plan détaillé d'analyse est disponible ici :

➡️ [📄 Plan d'analyse funnel](docs/analysis_plan.md)

Résumé :
- Analyse du funnel (`view ➔ cart ➔ purchase`) : taux de conversion, délais entre étapes.
- Segmentation par catégorie, marque, prix, jour/heure.
- Analyse des paniers abandonnés.
- Analyse de récurrence et cohortes utilisateur.

---

## ⚙️ Setup

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/<ton-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
```
### 2️⃣ Créer un environnement virtuel

```bash
Copier
Modifier
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```
### 3️⃣ Installer les dépendances
```bash

pip install -r requirements.txt
```
Paquets principaux :

duckdb

polars

pandas

matplotlib, seaborn, plotly



### 4️⃣ Organisation des dossiers

/data/             # Contiendra les fichiers parquet téléchargés depuis Kaggle
/notebooks/        # Contiendra les notebooks d'analyse
/src/              # Scripts d'ETL et fonctions utilitaires
/docs/             # Documentation du projet
README.md
requirements.txt

### 🚀 Usage
1️⃣ Télécharger les données Parquet filtrées depuis Kaggle dans le dossier /data/.
2️⃣ Lancer le notebook d'exploration initiale dans VS Code ou Jupyter :

```bash
jupyter notebook notebooks/01_funnel_exploration.ipynb
```
3️⃣ Suivre les étapes d'analyse funnel, de segmentation et d'extraction des données filtrées pour modélisation.


🛠️ Prochaines étapes
✅ Déployer un dashboard Plotly ou Streamlit pour visualiser le funnel.
✅ Construire un modèle de classification propension achat.
✅ Étendre le projet à d'autres datasets SaaS ou e-commerce.

✨ Contact
👩‍💻 Blandine JATSA
📧 [Ton email ou profil LinkedIn]
