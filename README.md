# Prévision de la pluie en Australie

Ce projet cherche à prévoir s'il pleuvra le lendemain (J+1) et le surlendemain (J+2) dans 49 stations météorologiques australiennes, à partir des seules observations du jour (températures, humidité, pression, vent, précipitations). Les données réunissent 145 460 relevés quotidiens de 2007 à 2017. Le découpage entre entraînement et test est chronologique : les modèles sont évalués sur une période postérieure à leur entraînement, comme en conditions réelles. Les deux erreurs (fausse alerte et pluie manquée) ayant un coût, la performance est mesurée par le F1-score de la classe « pluie ».

## Équipe

- Axelle Komano
- Nicolas Jourdaine
- Yan Huang

## Résultats sur l'ensemble de test

| Horizon | XGBoost (modèle retenu) | Meilleur modèle de l'horizon |
|---|---|---|
| J+1 | F1 = 0,660 | XGBoost (0,660) |
| J+2 | F1 = 0,436 | Réseau de neurones (0,471) |

XGBoost est retenu pour l'application : meilleur F1 à J+1, entraînement et prédiction rapides, fichier léger et interprétable avec SHAP. À J+2, le réseau de neurones fait légèrement mieux, au prix d'un entraînement bien plus long.

## Contenu du dépôt

```
app.py                         application Streamlit
graphiques.py                  graphiques Plotly et préparation d'une observation brute (communs au notebook 04 et à l'application)
requirements.txt               bibliothèques de l'application
.streamlit/config.toml         thème de l'application
static/fonts/                  polices de l'application (Geist, Plus Jakarta Sans) et leurs licences
notebooks/
    01_exploration.ipynb                   exploration des données brutes
    02_preprocessing.ipynb                 nettoyage, cible J+2, découpage chronologique, imputation, encodage, sélection des variables, mise à l'échelle
    03_modelisation.ipynb                  modèle naïf et cinq modèles (J+1 et J+2) : déséquilibre des classes, hyperparamètres, seuil de décision, évaluation sur le test, interprétabilité (SHAP)
    04_visualisations_interactives.ipynb   fichiers de données de l'application et graphiques interactifs
data/
    raw/weatherAUS.csv             données brutes
    output/app/                    données affichées par l'application (exportées par le notebook 04)
    output/X_test_J1_arbres.csv    variables préparées du test (graphique SHAP de l'application)
    output/X_test_J2_arbres.csv
models/
    modele_xgboost.json            modèle XGBoost final, J+1
    modele_xgboost_J2.json         modèle XGBoost final, J+2
    preprocessing_artifacts.joblib transformations du pré-traitement ajustées sur l'entraînement
    modeles_parametres.joblib      paramètres et seuils de décision retenus, J+1
    modeles_parametres_J2.joblib   paramètres et seuils de décision retenus, J+2
```

## Lancer l'application en local

Depuis la racine du dépôt :

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Sous macOS ou Linux, l'environnement s'active avec `source venv/bin/activate`. L'application a été développée avec Python 3.14.

## Application en ligne

Lien de l'application déployée : https://prevision-pluie-australie.streamlit.app

## Notebooks

Les notebooks sont fournis pour la lecture, avec leurs sorties. Les données intermédiaires et les autres modèles (régression logistique, Random Forest, KNN, réseau de neurones) ne sont pas dans le dépôt : ils sont régénérés en exécutant les notebooks 01 à 03 dans l'ordre, à partir du fichier brut `data/raw/weatherAUS.csv`, puis le notebook 04. Les notebooks enregistrent leurs figures dans un dossier `figures/` (et `figures/app/` pour le notebook 04) à créer à la racine du dépôt avant l'exécution.

Bibliothèques des notebooks, en plus de celles de `requirements.txt` :

- Python 3.14.2
- scikit-learn 1.9.1
- XGBoost 3.4.1
- imbalanced-learn 0.14.2
- TensorFlow 2.22.0rc0
- SHAP 0.52.0
- Matplotlib 3.10.8, Seaborn 0.13.2 et SciPy 1.18.1
- Kaleido 1.4.0 (export des figures du notebook 04)

Le rapport et le support de présentation sont remis séparément.

## Source des données

Jeu de données « Rain in Australia » publié sur Kaggle (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package), construit à partir des observations quotidiennes du Bureau of Meteorology australien (http://www.bom.gov.au/climate/data).
