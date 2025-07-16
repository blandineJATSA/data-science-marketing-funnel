# 🎯 Plan d'analyse funnel e-commerce

Ce document décrit **les analyses à réaliser sur le dataset e-commerce** pour détecter les points de friction dans le tunnel de conversion et identifier des leviers d'amélioration du taux de conversion.

---

## 1️⃣ Funnel de conversion

**Objectif :** Identifier à quelles étapes les utilisateurs quittent le tunnel de conversion.

**Analyses à réaliser :**
- Compter le nombre de **sessions uniques** avec :
  - `view` (consultation)
  - `cart` (ajout panier)
  - `purchase` (achat)
- Calculer :
  - Taux de conversion `view ➔ cart`
  - Taux de conversion `cart ➔ purchase`
  - Taux de conversion global `view ➔ purchase`
- Calculer le **temps moyen** :
  - Entre `view` et `cart`
  - Entre `cart` et `purchase`

---

## 2️⃣ Segmentation du funnel

**Objectif :** Identifier les catégories, marques, plages de prix et moments qui influencent la conversion.

**Découpes prévues :**
- Par `category_code` : identifier les catégories à fort/faible taux de conversion.
- Par `brand` : certaines marques freinent-elles ou accélèrent-elles la conversion ?
- Par `price` : analyser l’impact du prix sur le taux de conversion.
- Par `jour` et `heure` (`event_time`) : détecter les périodes favorables aux conversions.

---

## 3️⃣ Analyse des paniers abandonnés

**Objectif :** Comprendre les comportements des sessions qui ajoutent au panier sans finaliser l'achat.

**Points à analyser :**
- Identifier les sessions avec `cart` sans `purchase`.
- Lister les produits les plus concernés par l'abandon.
- Calculer le prix moyen des paniers abandonnés.
- Estimer le temps passé avant abandon.

---

## 4️⃣ Analyse de récurrence et cohortes utilisateur

**Objectif :** Comprendre le comportement d'achat dans le temps.

**Analyses à prévoir :**
- Compter le nombre d'achats par `user_id`.
- Calculer le délai moyen entre le premier `view` et le premier `purchase` par utilisateur.
- Construire des **cohortes par date de première interaction** pour suivre l'évolution des conversions dans le temps.

---

## Utilisation de ce plan

Ce plan guidera :
✅ Les étapes d'**ETL et d'exploration initiale dans Polars/DuckDB**  
✅ La création de **notebooks d'exploration funnel et visualisations**  
✅ L'extraction de sous-ensembles filtrés pour modélisation (classification propension achat, LTV, etc.)

---

**🚀 Prêt à être intégré dans le workflow VS Code et versionné sur GitHub.**

