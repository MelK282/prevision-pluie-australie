# app.py : application Streamlit « Prévision de la pluie en Australie »
# Graphiques : fonctions de graphiques.py ; données : fichiers de data/output/app/, données brutes, modèles XGBoost
# et artefacts du pré-traitement sauvegardés (aucun réentraînement)

import datetime as dt
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
from xgboost import XGBClassifier
from graphiques import (MODELES, UNITES_VARIABLES, libelle, libelle_axe, nombre_fr, entier_fr, preparer_brut,
                        scores_seuil, calculer_sensibilite, graphique_repartition_cible, graphique_valeurs_manquantes,
                        graphique_distribution_pluie, graphique_correlation_cible, graphique_pluie_par_station,
                        graphique_taux_mensuel, graphique_biais_evite, graphique_imputation,
                        graphique_encodage_cyclique, graphique_paires_redondantes, graphique_jeu_reduit,
                        graphique_f1_par_pli, graphique_evolution, graphique_comparaison_modeles, graphique_seuil,
                        graphique_shap_synthese, graphique_shap_beeswarm, graphique_odds_ratios, graphique_stations, graphique_jauge,
                        graphique_contribution, graphique_sensibilite)

st.set_page_config(layout="wide", page_title="Prévision de la pluie en Australie")

# Feuille de style de l'application (injectée une seule fois) : polices servies depuis static/fonts/, couleurs,
# en-tête et barre d'onglets fixes au défilement, cartes, chiffres clés, onglets, espacements (grille de 8 px)
STYLE = """
<style>
@font-face {font-family: 'Geist'; src: url('app/static/fonts/geist-latin-wght-normal.woff2') format('woff2');
            font-weight: 100 900; font-style: normal; font-display: swap;}
@font-face {font-family: 'Plus Jakarta Sans';
            src: url('app/static/fonts/plus-jakarta-sans-latin-wght-normal.woff2') format('woff2');
            font-weight: 200 800; font-style: normal; font-display: swap;}
:root {--fond: #F6F9FC; --fond-menu: #E9F0F7; --surface: #FFFFFF; --surface-discrete: #EEF4FA; --survol: #E1ECF6;
       --texte: #10233A; --texte-secondaire: #566B80; --trait: #D6E1EC; --accent: #0B4F8A; --accent-teinte: #DCEEFA;
       --accent-vif: #02C4FA; --encre: #000B0B; --ambre-teinte: #FBEBD2; --succes: #24785B; --danger: #BA323C;
       --police-texte: 'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif;
       --police-titres: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
       --hauteur-entete: 92px;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
[data-testid="stMarkdownContainer"], [data-testid="stMetricLabel"], [data-baseweb="tab"], [data-baseweb="select"],
[data-testid="stDataFrame"] {font-family: var(--police-texte) !important; color: var(--texte);}
.stApp h1, .stApp h2, .stApp h3, .stApp p.titre-app, .stApp p.titre-onglet, [data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    font-family: var(--police-titres) !important; color: var(--accent); letter-spacing: -0.02em;}
.stApp * {overflow-wrap: normal !important; word-break: normal !important; hyphens: none !important;}
.stApp {background: var(--fond);}
[data-testid="stSidebar"] {background: var(--fond-menu); border-right: 1px solid var(--trait); width: 15rem !important;
                           min-width: 15rem !important; max-width: 15rem !important;}
[data-testid="stSidebar"] label p {white-space: nowrap;}
[data-testid="stHeader"] {background: transparent; pointer-events: none;}
[data-testid="stHeader"] button, [data-testid="stHeader"] a, [data-testid="stToolbar"] {pointer-events: auto;}
[data-testid="stAppDeployButton"] {display: none;}
[data-testid="stMainBlockContainer"] {padding-top: 1.5rem; padding-bottom: 48px;}
[data-testid="stVerticalBlock"] {gap: 24px;}
.stApp h2 {font-size: 22px; font-weight: 600; padding: 24px 0 0 0; margin: 0;}
.stApp h3 {font-size: 20px; font-weight: 600; padding: 8px 0 0 0; margin: 0;}
[data-testid="stLayoutWrapper"]:has(> .st-key-entete) {position: sticky; top: 0; z-index: 990;}
.st-key-entete {background: var(--fond); border-bottom: 1px solid var(--trait); padding: 8px 0 12px 0;
                min-height: var(--hauteur-entete);}
.stApp p.titre-app {font-size: 28px !important; font-weight: 600; line-height: 1.2; margin: 0;}
.stApp p.auteurs {font-size: 13px !important; color: var(--texte-secondaire) !important; margin: 4px 0 0 0;}
.stApp p.titre-onglet {font-size: 20px !important; font-weight: 600; line-height: 1.2; margin: 8px 0 0 0;}
[data-testid="stTabs"] [role="tablist"] {position: sticky; top: var(--hauteur-entete); z-index: 980;
                                         background: var(--fond); gap: 4px; border-bottom: 1px solid var(--trait);
                                         flex-wrap: wrap; overflow: visible !important;}
[data-testid="stTab"] {padding: 0 6px;}
[data-testid="stTab"] p {color: var(--texte-secondaire) !important; font-size: 14px !important;}
[data-testid="stTab"][aria-selected="true"] {box-shadow: inset 0 -2px 0 var(--accent);}
[data-testid="stTab"][aria-selected="true"] p {color: var(--accent) !important; font-weight: 600;}
.stVerticalBlock[class*="st-key-carte_"] {background: var(--surface); border: 1px solid var(--trait) !important;
                                          border-radius: 12px !important; padding: 16px 20px !important;
                                          box-shadow: 0 1px 3px #000b0b15; gap: 12px;}
[data-testid="stMetricLabel"] p {font-size: 13px !important; color: var(--texte-secondaire) !important;
                                 white-space: normal !important; overflow: visible !important;
                                 text-overflow: clip !important; line-height: 1.4;}
[data-testid="stMetricValue"] {font-size: 28px !important; font-weight: 600 !important; line-height: 1.2;}
.stApp [data-testid="stMetricValue"] p {font-family: var(--police-titres) !important; font-size: 28px !important;
                                         font-weight: 600 !important; line-height: 1.2; color: var(--accent);}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stMetricValue"] div,
[data-testid="stMetricDelta"] div, [data-testid="stMetricValue"] p,
[data-testid="stMetricDelta"] p {white-space: normal !important;
                                                                      overflow: visible !important;
                                                                      text-overflow: clip !important;}
.stButton button {background: var(--accent-vif); color: var(--encre); border: none; border-radius: 8px;
                  font-weight: 600; min-height: 36px;}
.stButton button:hover {background: #00AEDF; color: var(--encre);}
.stButton button p {color: var(--encre);}
[data-testid="stSelectbox"] [role="group"] {border: 1px solid var(--texte-secondaire);}
[data-testid="stSelectbox"] [role="group"]:hover,
[data-testid="stSelectbox"] [role="group"]:focus-within {border-color: var(--accent);}
.stApp div.verdict {width: 100%; border-radius: 12px; padding: 16px 20px; box-sizing: border-box;}
.stApp div.verdict-pluie {background: var(--accent-teinte);}
.stApp div.verdict-sec {background: var(--ambre-teinte);}
.stApp div.verdict p {margin: 0;}
.stApp p.verdict-titre {font-size: 16px !important; font-weight: 600;}
.stApp p.verdict-proba {font-family: var(--police-titres) !important; font-size: 28px !important; font-weight: 600;
                        line-height: 1.3; color: var(--accent);}
.stApp p.verdict-contexte {font-size: 13px !important; color: var(--texte-secondaire) !important;}
.stApp p.prevision-correcte {color: var(--succes) !important; font-weight: 600;}
.stApp p.prevision-incorrecte {color: var(--danger) !important; font-weight: 600;}
.stApp p.prevision-indisponible {color: var(--texte-secondaire) !important; font-weight: 600;}
[class*="st-key-compact_"] .stVerticalBlock[class*="st-key-carte_"] {padding: 10px 16px !important; gap: 4px;}
[class*="st-key-compact_"] [data-testid="stMetricValue"],
.stApp [class*="st-key-compact_"] [data-testid="stMetricValue"] p {font-size: 22px !important;
                                                                    white-space: nowrap !important;}
[class*="st-key-compact_"] [data-testid="stHorizontalBlock"] {gap: 12px;}
[class*="st-key-prevision_"] .modebar-container {display: none !important;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------------------
# Chargement (une seule fois, gardé en mémoire)
# ---------------------------------------------------------------------------------------------------------------

@st.cache_data
def charger_csv(nom):
    # Fichier exporté par le notebook 04 (data/output/app/)
    return pd.read_csv('data/output/app/' + nom)


@st.cache_data
def charger_brut():
    # Données brutes, avec le mois de chaque relevé (pour la simulation)
    brut = pd.read_csv('data/raw/weatherAUS.csv')
    brut['Mois'] = pd.to_datetime(brut['Date']).dt.month
    return brut


@st.cache_data
def charger_variables_test(suffixe):
    # Variables préparées du test, non mises à l'échelle (02_preprocessing), pour le graphique SHAP en essaim
    return pd.read_csv('data/output/X_test_' + suffixe + '_arbres.csv')


@st.cache_resource
def charger_artefacts():
    # Transformations ajustées sur l'ensemble d'entraînement (02_preprocessing)
    return joblib.load('models/preprocessing_artifacts.joblib')


@st.cache_resource
def charger_modele(horizon):
    # Modèle XGBoost final de l'horizon ('J+1' ou 'J+2'), sauvegardé par 03_modelisation
    modele = XGBClassifier()
    modele.load_model(FICHIERS_XGBOOST[horizon])
    return modele


@st.cache_resource
def charger_seuils():
    # Seuil de décision retenu pour chaque modèle et chaque horizon (03_modelisation)
    return {'J+1': joblib.load('models/modeles_parametres.joblib')['seuil_retenu'],
            'J+2': joblib.load('models/modeles_parametres_J2.joblib')['seuil_retenu']}


@st.cache_resource
def charger_explainer():
    # Explication SHAP du modèle XGBoost J+1 (valeurs exactes pour les arbres, en log-odds)
    return shap.TreeExplainer(charger_modele('J+1'))


@st.cache_data
def valeurs_par_defaut(station, mois):
    # Médiane (mesures) ou valeur la plus fréquente (directions, pluie du jour) de la station pour ce mois, sur la
    # période d'entraînement ; valeur de toutes les stations si la station ne mesure pas la variable
    brut = charger_brut()
    entrainement = brut[brut['Date'] < '2015-11-10']
    sous_ensemble = entrainement[(entrainement['Location'] == station) & (entrainement['Mois'] == mois)]
    defauts = sous_ensemble[MESURES].median().fillna(entrainement[MESURES].median()).to_dict()
    for col in DIRECTIONS + ['RainToday']:
        comptes = sous_ensemble[col].value_counts()
        if len(comptes) == 0:
            comptes = entrainement[col].value_counts()
        defauts[col] = comptes.index[0]
    return defauts


# ---------------------------------------------------------------------------------------------------------------
# Constantes et textes
# ---------------------------------------------------------------------------------------------------------------

FICHIERS_XGBOOST = {'J+1': 'models/modele_xgboost.json', 'J+2': 'models/modele_xgboost_J2.json'}
SUFFIXES = {'J+1': 'J1', 'J+2': 'J2'}
PAGES = ["Présentation", "Exploration des données", "Pré-traitement", "Modélisation", "Prédiction"]
MESURES = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
           'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
DIRECTIONS = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
# Mesures entières (humidité, vitesses du vent) ; les autres mesures ont une décimale
MESURES_ENTIERES = ['Humidity9am', 'Humidity3pm', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
# Variables de l'analyse de sensibilité (la température maximale est retirée des données préparées : sans effet)
VARIABLES_SENSIBILITE = ['MinTemp', 'Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                         'Pressure3pm', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Rainfall']
# Observations du jour : grille de 4 cartes par ligne, le vent sur deux lignes
RANGEES_OBSERVATIONS = [['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm'],
                        ['Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm'],
                        ['WindGustSpeed', 'WindGustDir', 'WindSpeed9am', 'WindDir9am'],
                        ['WindSpeed3pm', 'WindDir3pm', 'Rainfall', 'RainToday']]
NOMS_MOIS = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre',
             'Novembre', 'Décembre']
METRIQUES = {'F1-score': 'F1', 'Précision': 'Précision', 'Rappel': 'Rappel', 'Accuracy': 'Accuracy',
             'AUC': 'AUC', "Temps d'entraînement": 'Temps entraînement (s)',
             'Temps de prédiction': 'Temps prédiction (s)', 'Taille du fichier': 'Taille du fichier (Mo)'}
INTERPRETABILITE = {'Régression logistique': 'Coefficients (odds ratios)', 'Random Forest': 'Importance des variables',
                    'XGBoost': 'Importance des variables et SHAP', 'KNN': 'Interprétation non calculée dans ce projet',
                    'Réseau de neurones': 'Interprétation non calculée dans ce projet'}
JOURS_HORIZON = {'J+1': 'demain', 'J+2': 'après-demain'}
DECISIONS = {True: 'Pluie prévue', False: 'Pas de pluie prévue'}
CLASSES_VERDICT = {True: 'verdict verdict-pluie', False: 'verdict verdict-sec'}
REALITES = {True: 'Il a plu', False: "Il n'a pas plu"}
CORRECTIONS = {True: 'prévision correcte', False: 'prévision incorrecte'}
CLASSES_CORRECTION = {True: 'prevision-correcte', False: 'prevision-incorrecte'}
# Taux de jours suivis d'une pluie de chaque station sur la période d'entraînement (données brutes antérieures au
# 10/11/2015, cible J+1 ou J+2), calculé par 02_preprocessing et sauvegardé dans les artefacts
TAUX_STATIONS = {'J+1': 'loc_rate_J1', 'J+2': 'loc_rate_J2'}

LECTURES = {
    'f1_par_pli': {
        'J+1': "Le classement des modèles reste stable d'un pli à l'autre : XGBoost et le réseau de neurones en tête, "
               "le KNN en retrait. Les variations entre plis (jusqu'à 0,08) dépassent les écarts entre les meilleurs "
               "modèles, d'où l'intérêt de raisonner sur la moyenne des cinq plis.",
        'J+2': "Les scores sont bien plus faibles et plus instables d'un pli à l'autre : à deux jours, le signal est "
               "faible et le classement des modèles change selon la période."},
    'evolution': {
        'J+1': "Chaque étape améliore le F1 de XGBoost, de 0,610 avec les paramètres par défaut à 0,660 sur le test. "
               "Le seuil de décision apporte le gain le plus net (+0,025).",
        'J+2': "Le F1 de XGBoost passe de 0,285 à 0,436 : à deux jours, l'optimisation des hyperparamètres (+0,054) "
               "et du seuil (+0,087) pèse bien davantage."},
    'comparaison': {
        'J+1': "XGBoost obtient le meilleur F1 sur le test (0,660), suivi du réseau de neurones (0,650). Tous les "
               "modèles dépassent largement le modèle naïf.",
        'J+2': "Les cinq modèles se tiennent en 0,05 de F1 ; le réseau de neurones obtient le meilleur score (0,471)."},
    'seuil': {
        'J+1': "Abaisser le seuil fait détecter plus de jours de pluie (rappel) au prix de fausses alertes plus "
               "nombreuses (précision). Le seuil retenu (0,302) est celui qui maximise le F1 en validation : 71 % des "
               "jours de pluie sont alors détectés, et 61 % des alertes sont justifiées.",
        'J+2': "Au seuil retenu (0,373), le modèle détecte 65 % des jours de pluie, mais seule une alerte sur trois "
               "est justifiée."},
    'shap': {
        'J+1': "L'humidité de l'après-midi est de loin la variable qui pèse le plus sur les prévisions, deux fois plus "
               "que la suivante. Viennent ensuite les rafales, la pression du matin et la température de "
               "l'après-midi : les signaux attendus d'un épisode pluvieux.",
        'J+2': "À deux jours, aucune variable ne domine : la pluie du jour et l'humidité de l'après-midi restent en "
               "tête, mais avec un poids bien plus faible, et le modèle s'appuie sur de nombreux signaux secondaires "
               "(directions du vent, station, saison)."},
    'odds': {
        'J+1': "Toutes choses égales par ailleurs, une humidité à 15 h élevée multiplie les chances de pluie par 2,6, "
               "des rafales fortes par 2,3, une pluie le jour même par 1,7. À l'inverse, une pression élevée à 9 h "
               "les réduit (×0,68).",
        'J+2': "À deux jours, les odds ratios restent proches de 1 : aucune variable n'a d'effet fort. La direction "
               "de la rafale (×1,45), l'humidité à 15 h (×1,40) et la pluie du jour (×1,35) augmentent les chances "
               "de pluie ; la hausse de pression entre 9 h et 15 h (×0,74) et la pression à 9 h (×0,78) les "
               "réduisent."},
    'stations': {
        'J+1': "Le F1 va de 0,42 (Newcastle) à 0,77 (Walpole). Il est le plus faible dans les stations sèches, où les "
               "jours de pluie sont rares, et à Newcastle, qui ne mesure ni la pression ni les rafales.",
        'J+2': "Le F1 chute dans toutes les stations ; 40 sur 49 passent sous 0,5."}
}
LECTURE_ESSAIM = ("Chaque point est une journée de test. À droite de zéro, la variable a poussé la prévision vers la "
                  "pluie ; à gauche, vers l'absence de pluie. La couleur indique la valeur de la variable : une humidité "
                  "élevée (points rouges) pousse vers la pluie, une pression élevée pousse vers l'absence de pluie.")
LECTURE_JAUGE = ("La probabilité de pluie estimée par le modèle ; au-delà du trait rouge (le seuil de décision), le "
                 "modèle prévoit de la pluie.")
LECTURE_CONTRIBUTION = ("Chaque barre montre ce qu'une mesure du jour a apporté à la prévision : vers la droite, elle "
                        "rend la pluie plus probable ; vers la gauche, moins probable.")
LECTURE_SENSIBILITE = ("Comment la probabilité de pluie évoluerait si cette seule mesure changeait, toutes les autres "
                       "restant celles du jour. Le trait vertical marque la valeur observée.")


# ---------------------------------------------------------------------------------------------------------------
# Fonctions d'affichage répétées
# ---------------------------------------------------------------------------------------------------------------

# Numéro de la prochaine carte de la page (remis à zéro à chaque exécution du script)
compteur_cartes = [0]


def nouvelle_carte(conteneur, hauteur):
    # Carte : conteneur à bordure, de hauteur fixe ou non (None), avec une clé « carte_N » qui lui donne son style
    compteur_cartes[0] = compteur_cartes[0] + 1
    if hauteur is None:
        return conteneur.container(border=True, key='carte_' + str(compteur_cartes[0]))
    return conteneur.container(border=True, height=hauteur, key='carte_' + str(compteur_cartes[0]))


def titre_onglet(conteneur, titre):
    # Nom de l'onglet en titre, au début de son contenu
    conteneur.markdown('<p class="titre-onglet">' + titre + '</p>', unsafe_allow_html=True)


def cartes_chiffres(conteneur, chiffres, hauteur):
    # Une carte par chiffre clé, côte à côte, de même largeur et de même hauteur ; chiffres : liste de
    # [libellé, valeur] ou [libellé, valeur, écart] (l'écart est affiché sous la valeur)
    colonnes = conteneur.columns(len(chiffres))
    for i in range(len(chiffres)):
        carte = nouvelle_carte(colonnes[i], hauteur)
        if len(chiffres[i]) == 3:
            carte.metric(chiffres[i][0], chiffres[i][1], delta=chiffres[i][2])
        else:
            carte.metric(chiffres[i][0], chiffres[i][1])


def cartes_etapes(conteneur, etapes, par_ligne, hauteur):
    # Cartes de même taille, par lignes de `par_ligne`, un titre en gras et une ligne de texte ;
    # etapes : liste de [titre, texte]
    for debut in range(0, len(etapes), par_ligne):
        colonnes = conteneur.columns(par_ligne)
        for j in range(par_ligne):
            carte = nouvelle_carte(colonnes[j], hauteur)
            carte.markdown('**' + etapes[debut + j][0] + '**')
            carte.write(etapes[debut + j][1])


def ecart_fr(valeur, decimales):
    # Écart signé (+0,012 ou -0,024), pour l'affichage sous un chiffre clé
    if valeur >= 0:
        return '+' + nombre_fr(valeur, decimales)
    return nombre_fr(valeur, decimales)


def surligner_retenu(ligne):
    # Ligne du modèle retenu (XGBoost) : fond de la teinte d'accent et texte en gras
    styles = []
    for valeur in ligne:
        styles.append('')
        if ligne['Modèle'] == 'XGBoost':
            styles[len(styles) - 1] = 'background-color: #DCEEFA; font-weight: 600'
    return styles


def carte_graphique(conteneur, fig, lecture, cle):
    # Un graphique dans sa carte, pleine largeur, avec sa lecture en dessous
    carte = nouvelle_carte(conteneur, None)
    carte.plotly_chart(fig, width='stretch', theme=None, key=cle)
    if lecture != '':
        carte.write(lecture)
    return carte


def textes_colonne(colonne, decimales):
    # Valeurs d'une colonne avec une virgule décimale ; tiret si la valeur n'existe pas (modèle naïf)
    textes = []
    manquantes = colonne.isnull().values
    for i in range(len(colonne)):
        textes.append('–')
        if manquantes[i] == False:
            textes[i] = nombre_fr(colonne.values[i], decimales)
    return textes


def texte_taille(valeur):
    # Taille de fichier en Mo (valeur existante) ; moins de 0,1 Mo écrit « < 0,1 »
    if valeur < 0.1:
        return '< 0,1'
    return nombre_fr(valeur, 2)


def texte_seuil(valeur):
    # Seuil écrit avec 2 décimales, ou 3 pour un seuil retenu qui n'est pas un multiple de 0,01
    if round(valeur, 2) == valeur:
        return nombre_fr(valeur, 2)
    return nombre_fr(valeur, 3)


def texte_observation(observation, variable):
    # Mesure du jour en français avec son unité ; « Non mesurée » si la valeur est manquante
    if observation[variable].isnull().values[0]:
        return 'Non mesurée'
    valeur = observation[variable].values[0]
    if variable in DIRECTIONS:
        return valeur.replace('W', 'O')
    if variable == 'RainToday':
        return {'No': 'Non', 'Yes': 'Oui'}[valeur]
    return nombre_fr(valeur, 1) + ' ' + UNITES_VARIABLES[variable]


def libelle_direction(code):
    # Point cardinal en français (W, ouest -> O)
    return code.replace('W', 'O')


def libelle_oui_non(code):
    return {'No': 'Non', 'Yes': 'Oui'}[code]


def libelle_mois(mois):
    return NOMS_MOIS[mois - 1]


def saisir(conteneur, variable, defauts, cle):
    # Champ de saisie d'une observation, prérempli avec la valeur par défaut de la station et du mois, sauf si une
    # journée tirée au hasard l'a déjà rempli ; mesures bornées par le minimum et le maximum observés, affichées avec
    # leur précision naturelle (entier ou une décimale)
    valeur = defauts[variable]
    if variable in MESURES:
        valeur = float(round(valeur, 1))
    if variable in MESURES_ENTIERES:
        valeur = float(round(valeur))
    if cle + variable not in st.session_state:
        st.session_state[cle + variable] = valeur
    if variable in DIRECTIONS:
        codes = list(charger_artefacts()['numero_direction'])
        return conteneur.selectbox(libelle(variable), codes, format_func=libelle_direction, key=cle + variable)
    if variable == 'RainToday':
        return conteneur.selectbox("Pluie aujourd'hui", ['No', 'Yes'], format_func=libelle_oui_non, key=cle + variable)
    plage = charger_csv('plages_variables.csv')
    plage = plage[plage['Variable'] == variable]
    pas = 0.1
    affichage = '%.1f'
    if variable in MESURES_ENTIERES:
        pas = 1.0
        affichage = '%.0f'
    return conteneur.number_input(libelle_axe(variable), min_value=float(plage['Minimum'].values[0]),
                                  max_value=float(plage['Maximum'].values[0]), step=pas, format=affichage,
                                  key=cle + variable)


def prevoir(observation):
    # Ligne brute -> lignes préparées (J+1 et J+2, non mises à l'échelle) et probabilités de pluie XGBoost
    artefacts = charger_artefacts()
    X_J1 = preparer_brut(observation, artefacts, False, artefacts['colonnes_J1'])
    X_J2 = preparer_brut(observation, artefacts, False, artefacts['colonnes_J2'])
    return {'X_J1': X_J1, 'J+1': charger_modele('J+1').predict_proba(X_J1)[0, 1],
            'J+2': charger_modele('J+2').predict_proba(X_J2)[0, 1]}


def prevision_jour_reel(observation, station, date):
    # Prévision d'un jour réel : probabilités, ce qui s'est réellement passé (cibles des fichiers de test, absentes si
    # la cible est inconnue) et contributions (valeurs SHAP précalculées pour ce jour du test, sinon calculées)
    prevision = prevoir(observation)
    prevision['station'] = station
    probas_J1 = charger_csv('probas_test_J1.csv')
    probas_J2 = charger_csv('probas_test_J2.csv')
    ligne_J1 = probas_J1[(probas_J1['Location'] == station) & (probas_J1['Date'] == str(date))]
    ligne_J2 = probas_J2[(probas_J2['Location'] == station) & (probas_J2['Date'] == str(date))]
    prevision['reel'] = {'J+1': 'indisponible', 'J+2': 'indisponible'}
    if len(ligne_J2) > 0:
        prevision['reel']['J+2'] = int(ligne_J2['Cible'].values[0])
    if len(ligne_J1) > 0:
        prevision['reel']['J+1'] = int(ligne_J1['Cible'].values[0])
        shap_ligne = charger_csv('shap_xgboost_J1.csv').iloc[ligne_J1.index[0]]
        colonnes_J1 = prevision['X_J1'].columns
        prevision['contributions'] = pd.DataFrame({'Variable': colonnes_J1, 'Valeur': prevision['X_J1'].values[0],
                                                   'Contribution': shap_ligne[colonnes_J1].values})
        prevision['base'] = shap_ligne['Valeur de base']
    else:
        prevision['contributions'], prevision['base'] = contributions_shap(prevision['X_J1'])
    return prevision


def contributions_shap(X_J1):
    # Valeurs SHAP de la prédiction J+1 calculées à la demande, et valeur de base du modèle
    explainer = charger_explainer()
    valeurs = explainer.shap_values(X_J1)
    contributions = pd.DataFrame({'Variable': X_J1.columns, 'Valeur': X_J1.values[0], 'Contribution': valeurs[0]})
    return contributions, explainer.expected_value


def bandeau_verdict(conteneur, prevision, horizon):
    # Bandeau de la décision sur toute la largeur de la colonne : teinte d'accent (pluie prévue) ou or, décision et
    # probabilité, ce qui s'est réellement passé (jour réel), seuil de décision et fréquence habituelle de la pluie à
    # la station (période d'entraînement)
    seuil = charger_seuils()[horizon]['XGBoost']
    prevue = prevision[horizon] >= seuil
    taux = charger_artefacts()[TAUX_STATIONS[horizon]][prevision['station']]
    reel = prevision['reel'][horizon]
    resultat = ''
    if reel == 'indisponible':
        resultat = '<p class="prevision-indisponible">Observation à ' + horizon + ' indisponible</p>'
    elif reel != 'simulation':
        correcte = (reel == 1) == prevue
        resultat = ('<p class="' + CLASSES_CORRECTION[correcte] + '">' + REALITES[reel == 1] + ' : '
                    + CORRECTIONS[correcte] + '</p>')
    conteneur.markdown('<div class="' + CLASSES_VERDICT[prevue] + '">'
                       + '<p class="verdict-titre">' + DECISIONS[prevue] + ' ' + JOURS_HORIZON[horizon] + '</p>'
                       + '<p class="verdict-proba">' + nombre_fr(prevision[horizon] * 100, 1) + ' %</p>' + resultat
                       + '<p class="verdict-contexte">Seuil de décision : ' + entier_fr(seuil * 100) + ' % · '
                       + 'Fréquence habituelle de la pluie à ' + prevision['station'] + ' : ' + entier_fr(taux * 100)
                       + ' %</p></div>', unsafe_allow_html=True)


def carte_horizon(conteneur, prevision, horizon, cle):
    # Bandeau de la décision (avec ce qui s'est réellement passé), jauge et fiabilité de XGBoost à la station (test)
    seuil = charger_seuils()[horizon]['XGBoost']
    bandeau_verdict(conteneur, prevision, horizon)
    carte = nouvelle_carte(conteneur, None)
    carte.plotly_chart(graphique_jauge(prevision[horizon], seuil), width='stretch', theme=None, key=cle + horizon)
    stations = charger_csv('stations_' + SUFFIXES[horizon] + '.csv')
    fiabilite = stations[(stations['Modèle'] == 'XGBoost') & (stations['Station'] == prevision['station'])]
    conteneur.caption("Fiabilité du modèle à cette station, sur la période de test :")
    chiffres = [['F1', nombre_fr(fiabilite['F1'].values[0], 2)],
                ['Précision', nombre_fr(fiabilite['Précision'].values[0], 2)],
                ['Rappel', nombre_fr(fiabilite['Rappel'].values[0], 2)]]
    # À J+2 : écart avec la fiabilité à J+1 à la même station
    if horizon == 'J+2':
        stations_J1 = charger_csv('stations_J1.csv')
        fiabilite_J1 = stations_J1[(stations_J1['Modèle'] == 'XGBoost') & (stations_J1['Station'] == prevision['station'])]
        for i in range(len(chiffres)):
            colonne = chiffres[i][0]
            chiffres[i].append(ecart_fr(fiabilite[colonne].values[0] - fiabilite_J1[colonne].values[0], 2) + ' vs J+1')
    # Une rangée compacte de trois chiffres clés
    cartes_chiffres(conteneur.container(key='compact_fiabilite_' + cle + SUFFIXES[horizon]), chiffres, 110)


def afficher_prevision(conteneur, observation, prevision, cle):
    # Probabilités et décisions (J+1 et J+2 côte à côte), contribution des variables (J+1), analyse de sensibilité
    # (J+1) ; bloc à clé « prevision_ » : graphiques sans barre d'outils Plotly
    bloc = conteneur.container(key='prevision_' + cle)
    bloc.subheader("Probabilité de pluie")
    colonnes = bloc.columns(2)
    carte_horizon(colonnes[0], prevision, 'J+1', cle)
    carte_horizon(colonnes[1], prevision, 'J+2', cle)
    bloc.write(LECTURE_JAUGE)
    bloc.subheader("Contribution des variables")
    carte_graphique(bloc, graphique_contribution(prevision['contributions'], prevision['base'], 10),
                    LECTURE_CONTRIBUTION, cle + 'contribution')
    bloc.subheader("Analyse de sensibilité")
    carte = nouvelle_carte(bloc, None)
    # Variables mesurées ce jour-là uniquement (une valeur manquante n'a pas de valeur observée à marquer)
    variables = []
    for variable in VARIABLES_SENSIBILITE:
        if observation[variable].isnull().values[0] == False:
            variables.append(variable)
    position = 0
    if 'Humidity3pm' in variables:
        position = variables.index('Humidity3pm')
    variable = carte.selectbox("Variable", variables, index=position, format_func=libelle, key=cle + 'variable')
    artefacts = charger_artefacts()
    demo = {'modele': charger_modele('J+1'), 'artefacts': artefacts, 'mise_a_echelle': False,
            'colonnes': artefacts['colonnes_J1']}
    courbe = calculer_sensibilite(observation, demo, variable, charger_csv('plages_variables.csv'))
    fig = graphique_sensibilite(courbe, variable, charger_seuils()['J+1']['XGBoost'], observation[variable].values[0])
    carte.plotly_chart(fig, width='stretch', theme=None, key=cle + 'sensibilite')
    carte.write(LECTURE_SENSIBILITE)


def jour_au_hasard():
    # Jour du test tiré au hasard (graine = nombre de clics sur le bouton)
    st.session_state['compteur_hasard'] = st.session_state['compteur_hasard'] + 1
    ligne = charger_csv('probas_test_J1.csv').sample(n=1, random_state=st.session_state['compteur_hasard'])
    st.session_state['station_reelle'] = ligne['Location'].values[0]
    st.session_state['date_reelle'] = pd.to_datetime(ligne['Date'].values[0]).date()


def journee_au_hasard():
    # Jour de la période d'entraînement tiré au hasard parmi les jours où toutes les mesures existent (graine = nombre
    # de clics sur le bouton) : station, mois et chaque champ de la simulation reçoivent les valeurs de ce jour,
    # puis la prévision est lancée
    st.session_state['compteur_simulation'] = st.session_state['compteur_simulation'] + 1
    brut = charger_brut()
    jours = brut[brut['Date'] < '2015-11-10'].dropna(subset=MESURES + DIRECTIONS + ['RainToday'])
    ligne = jours.sample(n=1, random_state=st.session_state['compteur_simulation'])
    station = ligne['Location'].values[0]
    mois = int(ligne['Mois'].values[0])
    st.session_state['station_simulation'] = station
    st.session_state['mois_simulation'] = mois
    cle = 'simulation_' + station + '_' + str(mois) + '_'
    for variable in MESURES:
        st.session_state[cle + variable] = float(ligne[variable].values[0])
    for variable in DIRECTIONS + ['RainToday']:
        st.session_state[cle + variable] = ligne[variable].values[0]
    st.session_state['simulation_lancee'] = True


def revenir_au_seuil():
    # Curseur du seuil remis au seuil retenu du modèle et de l'horizon affichés
    horizon = st.session_state['horizon']
    modele = st.session_state['modele_seuil']
    st.session_state['seuil_' + horizon + modele] = float(charger_seuils()[horizon][modele])


# ---------------------------------------------------------------------------------------------------------------
# En-tête et navigation
# ---------------------------------------------------------------------------------------------------------------

page = st.sidebar.radio("Navigation", PAGES, key='page')
# En-tête fixe au défilement : titre, auteurs et, sur la page « Modélisation », le choix de l'horizon
entete = st.container(key='entete')
colonnes_entete = entete.columns([5, 3, 1], vertical_alignment='center')
colonnes_entete[0].markdown('<p class="titre-app">Prévision de la pluie en Australie</p>'
                            '<p class="auteurs">Analyse réalisée par : Axelle Komano, Nicolas Jourdaine, '
                            'Yan Huang</p>', unsafe_allow_html=True)
if page == PAGES[3]:
    horizon = colonnes_entete[1].radio("Horizon de prévision", ['J+1', 'J+2'], horizontal=True, key='horizon')

# ---------------------------------------------------------------------------------------------------------------
# Page « Présentation »
# ---------------------------------------------------------------------------------------------------------------

if page == PAGES[0]:
    st.header("Contexte et problématique")
    st.write("L'agriculture représente 62 % de la consommation d'eau en Australie. Anticiper la pluie permet d'éviter "
             "d'arroser avant une journée pluvieuse, sans priver les cultures avant une journée sèche. Ce projet "
             "cherche à prévoir la pluie à un jour (J+1) et à deux jours (J+2) à partir des seules observations "
             "météorologiques du jour. Les deux erreurs (fausse alerte et pluie manquée) ayant un coût, la "
             "performance est mesurée par le F1-score.")
    st.header("Le jeu de données")
    cartes_chiffres(st, [['Observations', '145 460'], ['Stations', '49'], ['Période', '2007-2017'],
                         ["Jours suivis d'une pluie", '22,4 %']], 112)
    st.header("Démarche du projet")
    cartes_etapes(st, [['Exploration des données', "Comprendre les données et ce qui annonce la pluie"],
                       ['Pré-traitement', "Préparer les données sans jamais utiliser la période de test"],
                       ['Modélisation', "Comparer cinq modèles et retenir le meilleur"],
                       ['Prédiction', "Prévoir la pluie à partir des observations d'une journée"]], 4, 156)

# ---------------------------------------------------------------------------------------------------------------
# Page « Exploration des données »
# ---------------------------------------------------------------------------------------------------------------

if page == PAGES[1]:
    st.header("Déséquilibre de la variable cible")
    carte_graphique(st, graphique_repartition_cible(charger_csv('exploration_repartition_cible.csv')),
                    "Un peu plus d'un jour sur cinq (22,4 %) est suivi d'une pluie. Un modèle qui prédirait toujours "
                    "« pas de pluie » aurait raison 77,6 % du temps sans détecter une seule pluie : l'accuracy est "
                    "donc trompeuse ici, d'où le choix du F1-score.", 'repartition')

    st.header("Valeurs manquantes")
    carte_graphique(st, graphique_valeurs_manquantes(charger_csv('exploration_valeurs_manquantes.csv')),
                    "Quatre variables (ensoleillement, évaporation, nébulosité à 9 h et à 15 h) dépassent largement le "
                    "seuil de 20 %, avec 38 à 48 % de valeurs manquantes : elles sont supprimées. Toutes les autres "
                    "restent sous 11 % et sont imputées.", 'manquants')

    st.header("Distribution des variables selon la pluie")
    carte = nouvelle_carte(st, None)
    variables_numeriques = list(charger_csv('plages_variables.csv')['Variable'])
    variable = carte.selectbox("Variable", variables_numeriques, index=variables_numeriques.index('Humidity3pm'),
                               format_func=libelle, key='variable_distribution')
    carte.plotly_chart(graphique_distribution_pluie(charger_brut(), variable), width='stretch', theme=None,
                       key='distribution')
    if variable == 'Humidity3pm':
        carte.write("Les jours suivis d'une pluie ont une humidité à 15 h nettement plus élevée : 70 % en médiane, "
                    "contre 47 % pour les autres.")
    carte.write("Plus les deux boîtes se séparent, plus la variable aide à distinguer les jours de pluie.")

    st.header("Corrélations avec la variable cible")
    carte_graphique(st, graphique_correlation_cible(charger_csv('exploration_correlations_cible.csv')),
                    "L'humidité de l'après-midi (+0,45) et l'ensoleillement (-0,45) sont les variables les plus liées "
                    "à la pluie du lendemain, suivies de la nébulosité. Une pression élevée va avec une pluie moins "
                    "probable (-0,25 à 9 h).", 'correlations')

    st.header("Fréquence de la pluie par station")
    carte_graphique(st, graphique_pluie_par_station(charger_csv('exploration_pluie_par_station.csv')),
                    "La fréquence de la pluie varie d'un facteur cinq selon la station : de 6,8 % des jours à "
                    "Woomera, en zone désertique, à 36,5 % à Portland, sur la côte sud. La localisation est donc une "
                    "information essentielle pour le modèle.", 'stations_exploration')

# ---------------------------------------------------------------------------------------------------------------
# Page « Pré-traitement »
# ---------------------------------------------------------------------------------------------------------------

if page == PAGES[2]:
    st.write("Les données brutes contiennent des valeurs manquantes, des classes déséquilibrées et une forte dimension "
             "temporelle. Chaque transformation est calculée sur la seule période d'entraînement, puis appliquée à la "
             "période de test.")
    cartes_chiffres(st, [['Données brutes', '145 460 × 23'], ['Variables préparées', '24'],
                         ['Jeux de données (J+1 et J+2)', '2'], ['Découpage chronologique', '80 / 20']], 112)
    # Noms courts des onglets (une seule ligne) ; le titre en tête de chaque onglet garde le nom complet
    onglets = st.tabs(["Nettoyage et cible J+2", "Découpage", "Imputation", "Variables et encodage",
                       "Sélection et échelle"])

    titre_onglet(onglets[0], "Nettoyage et construction de la cible J+2")
    onglets[0].write("Quatre variables dépassent 20 % de valeurs manquantes, seuil au-delà duquel une imputation n'est "
                     "plus fiable : l'ensoleillement (48,01 %), l'évaporation (43,17 %), la nébulosité à 15 h (40,81 %) "
                     "et à 9 h (38,42 %). Elles sont supprimées, bien que l'ensoleillement et la nébulosité séparent "
                     "nettement les jours de pluie : imputer près de la moitié de leurs valeurs reviendrait à les "
                     "fabriquer.")
    onglets[0].write("La cible J+2 reprend la pluie du jour relevée exactement deux jours plus tard à la même station ; "
                     "elle n'est jamais imputée. Appliquée avec un décalage d'un jour, cette construction retrouve la "
                     "cible J+1 sur 100 % des 142 017 lignes comparables.")
    onglets[0].write("Les 3 267 lignes sans cible J+1 sont supprimées, car imputer la cible reviendrait à fabriquer la "
                     "réponse : il reste 142 193 lignes, dont 1 762 (1,24 %) sans cible J+2.")
    cartes_chiffres(onglets[0], [['Variables supprimées', '4'], ['Jours sans cible retirés', '3 267'],
                                 ['Jours conservés', '142 193'], ['Jours sans cible J+2 (jeu J+2)', '1,24 %']], 156)

    titre_onglet(onglets[1], "Découpage temporel")
    onglets[1].write("Le découpage est chronologique : les dates les plus anciennes servent à l'entraînement, les plus "
                     "récentes au test, comme en conditions réelles où l'on prédit le futur à partir du passé. Un "
                     "découpage aléatoire placerait des journées consécutives, presque identiques, de part et d'autre, et "
                     "rendrait les scores artificiellement optimistes.")
    onglets[1].write("La coupure du 10/11/2015 place 80 % des lignes en entraînement (113 748, du 01/11/2007 au "
                     "09/11/2015) et 20 % en test (28 445, du 10/11/2015 au 25/06/2017) ; les 49 stations sont présentes "
                     "dans les deux ensembles.")
    onglets[1].write("Un découpage chronologique ne permet pas de stratifier, mais la proportion de jours suivis d'une "
                     "pluie reste très proche : 22,44 % à l'entraînement, 22,32 % au test.")
    cartes_chiffres(onglets[1], [['Date de coupure', '10/11/2015'], ["Jours d'entraînement (2007-2015)", '113 748'],
                                 ['Jours de test (2015-2017)', '28 445'],
                                 ['Jours suivis de pluie (entraînement / test)', '22,4 % / 22,3 %']], 156)
    carte_graphique(onglets[1], graphique_taux_mensuel(charger_csv('exploration_taux_pluie_mensuel.csv'),
                                                       charger_artefacts()['date_coupure']),
                    "La fréquence de la pluie suit un cycle saisonnier, plus élevée en hiver austral (27 % des jours en "
                    "juillet, contre 19 % en janvier). L'entraînement porte sur les jours antérieurs au 10/11/2015, le "
                    "test sur la période suivante : le modèle est évalué sur un futur qu'il n'a jamais vu.",
                    'taux_mensuel')

    titre_onglet(onglets[2], "Imputation des valeurs manquantes")
    onglets[2].write("Les mesures de 9 h et de 15 h d'une même grandeur sont fortement liées (corrélation de 0,86 pour "
                     "la température, 0,66 pour l'humidité, 0,96 pour la pression) : quand une seule manque, elle est "
                     "imputée à partir de l'autre mesure du même jour et de l'écart habituel entre 9 h et 15 h.")
    onglets[2].write("Les autres valeurs manquantes reçoivent la médiane (distributions nettement asymétriques), la "
                     "moyenne (distributions quasi symétriques) ou le mode (directions du vent, pluie du jour), calculés "
                     "par station sur l'entraînement, avec repli sur la valeur de l'ensemble des stations pour les "
                     "stations sans aucune mesure.")
    onglets[2].write("Les valeurs extrêmes, physiquement plausibles (jusqu'à 371 mm de pluie, des rafales de 135 km/h), "
                     "sont conservées : ce sont des phénomènes météorologiques réels, et non des erreurs.")
    cartes_chiffres(onglets[2], [['Écarts de pression aberrants (moyenne de la station)', '48,6 %'],
                                 ['Écarts de pression aberrants (mesurés)', '2,7 %'],
                                 ['Modèles meilleurs avec l\'imputation par station', '3 sur 5'],
                                 ['Valeurs manquantes après imputation', '0']], 156)
    carte_graphique(onglets[2], graphique_biais_evite(charger_csv('pretraitement_biais_evite.csv')),
                    "Remplir une mesure manquante par la moyenne de la station crée des écarts entre 9 h et 15 h "
                    "jamais observés : pour la pression, 48,6 % de ces écarts seraient aberrants, contre 2,7 % pour "
                    "les écarts réellement mesurés. D'où l'imputation à partir de l'autre mesure du même jour.",
                    'biais')
    carte_graphique(onglets[2], graphique_imputation(charger_csv('pretraitement_imputation_globale_locale.csv')),
                    "Les deux méthodes donnent des F1 très proches (écarts inférieurs à 0,003). L'imputation par "
                    "station l'emporte pour trois modèles sur cinq et est retenue.", 'imputation')

    titre_onglet(onglets[3], "Création et encodage des variables")
    onglets[3].write("Cinq variables sont créées : le mois, la saison de l'hémisphère sud et les écarts entre 15 h et "
                     "9 h de la température, de l'humidité et de la pression. Une journée qui se réchauffe peu, ou dont "
                     "l'humidité baisse peu entre 9 h et 15 h, est plus souvent suivie d'une pluie.")
    onglets[3].write("Le mois, la saison et les trois directions du vent reçoivent un encodage trigonométrique (sinus et "
                     "cosinus), avec deux colonnes par variable au lieu d'une colonne par modalité.")
    onglets[3].write("La station est remplacée par son taux de jours suivis d'une pluie, calculé sur l'entraînement "
                     "seul : de 5,87 % à Uluru à 36,80 % à Portland, avec une version pour J+1 et une pour J+2.")
    cartes_chiffres(onglets[3], [['Variables créées', '5'], ['Colonnes sinus / cosinus', '10'],
                                 ["Corrélation de l'écart de température avec la cible", '-0,33'],
                                 ['Stations remplacées par leur taux de pluie', '49']], 156)
    carte = nouvelle_carte(onglets[3], None)
    choix = carte.radio("Encodage affiché", ["Mois", "Directions du vent"], horizontal=True, key='encodage')
    types_encodage = {'Mois': 'Mois', 'Directions du vent': 'Direction du vent'}
    carte.plotly_chart(graphique_encodage_cyclique(charger_csv('pretraitement_encodage_cyclique.csv'),
                                                   types_encodage[choix]), width='stretch', theme=None,
                       key='graphique_encodage')
    carte.write("Placées sur un cercle par leur sinus et cosinus, les directions voisines restent voisines : le "
                "nord-nord-ouest est à côté du nord, ce qu'un simple numéro de 0 à 15 ne permettrait pas. Le même "
                "principe place décembre à côté de janvier.")

    titre_onglet(onglets[4], "Sélection et mise à l'échelle")
    onglets[4].write("Pour chaque paire de variables corrélées à |r| ≥ 0,8, la variable la plus corrélée à la cible est "
                     "conservée ; recalculé dans chaque pli, le filtre retire toujours les trois mêmes variables. Le jeu "
                     "réduit est retenu de justesse : il l'emporte pour trois modèles sur cinq, avec des écarts de "
                     "-0,0014 à +0,0052.")
    onglets[4].write("Chaque variable reçoit le scaler adapté à sa distribution, ajusté sur l'entraînement : RobustScaler "
                     "si au moins 1 % de ses valeurs mesurées sont aberrantes, sinon StandardScaler si elle est quasi "
                     "normale (|skewness| < 0,5), sinon MinMaxScaler ; les sinus et cosinus, de distribution en U, "
                     "reçoivent le MinMaxScaler.")
    onglets[4].write("Random Forest et XGBoost, à base d'arbres, n'ont pas besoin de mise à l'échelle et reçoivent une "
                     "copie non mise à l'échelle des données.")
    cartes_chiffres(onglets[4], [['Variables retirées', '3'], ['Colonnes RobustScaler', '7'],
                                 ['Colonnes StandardScaler (par jeu)', '6'], ['Colonnes MinMaxScaler', '10']], 156)
    carte_graphique(onglets[4], graphique_paires_redondantes(charger_csv('pretraitement_paires_redondantes.csv')),
                    "Trois paires de variables sont corrélées à plus de 0,8. Pour chacune, la variable la moins liée à "
                    "la pluie est retirée : température maximale, pression à 15 h et température à 9 h.", 'paires')
    carte_graphique(onglets[4], graphique_jeu_reduit(charger_csv('pretraitement_jeu_complet_reduit.csv')),
                    "Retirer les trois variables redondantes ne dégrade pas les performances : le jeu réduit fait au "
                    "moins aussi bien pour trois modèles sur cinq, et améliore nettement le KNN, sensible aux variables "
                    "en double.", 'jeu_reduit')

# ---------------------------------------------------------------------------------------------------------------
# Page « Modélisation »
# ---------------------------------------------------------------------------------------------------------------

if page == PAGES[3]:
    suffixe = SUFFIXES[horizon]
    resultats = charger_csv('resultats_test_' + suffixe + '.csv')
    # Noms courts des onglets (une seule ligne) ; le titre en tête de chaque onglet garde le nom complet
    onglets = st.tabs(["Démarche", "Évolution", "Comparaison", "Seuil", "Interprétabilité", "Par station",
                       "Modèle retenu"])

    # Démarche
    titre_onglet(onglets[0], "Démarche")
    cartes_etapes(onglets[0], [
        ['Modèle naïf et cinq modèles', "Un modèle naïf (toujours « pas de pluie ») sert de référence aux cinq modèles "
                                        "comparés."],
        ['Validation chronologique', "Cinq plis successifs : chaque modèle est validé sur une période postérieure à "
                                     "son entraînement."],
        ['Déséquilibre des classes', "Pondération des classes ou SMOTENC, retenue si elle améliore le F1 moyen."],
        ['Hyperparamètres', "Recherche par grille pour les trois meilleurs modèles : Random Forest, XGBoost et réseau "
                            "de neurones."],
        ['Seuil de décision', "Seuil de probabilité qui maximise le F1 moyen sur les plis de validation."],
        ['Évaluation unique', "Les choix faits en validation sont appliqués une seule fois à la période de test."]],
        3, 156)
    carte_graphique(onglets[0], graphique_f1_par_pli(charger_csv('scores_plis_' + suffixe + '.csv'), horizon),
                    LECTURES['f1_par_pli'][horizon], 'f1_par_pli')

    # Évolution des performances
    titre_onglet(onglets[1], "Évolution des performances")
    carte = nouvelle_carte(onglets[1], None)
    modele = carte.selectbox("Modèle", MODELES, index=MODELES.index('XGBoost'), key='modele_evolution')
    carte.plotly_chart(graphique_evolution(charger_csv('progression_' + suffixe + '.csv'), modele, horizon),
                       width='stretch', theme=None, key='evolution')
    if modele == 'XGBoost':
        carte.write(LECTURES['evolution'][horizon])

    # Comparaison des modèles
    titre_onglet(onglets[2], "Comparaison des modèles")
    carte = nouvelle_carte(onglets[2], None)
    libelle_metrique = carte.selectbox("Mesure", list(METRIQUES), key='metrique')
    carte.plotly_chart(graphique_comparaison_modeles(resultats, METRIQUES[libelle_metrique], horizon),
                       width='stretch', theme=None, key='comparaison')
    carte.write(LECTURES['comparaison'][horizon])
    tableau = pd.DataFrame({'Modèle': resultats['Modèle']})
    colonnes_decimales = [['Seuil', 3], ['F1', 3], ['Précision', 3], ['Rappel', 3], ['Accuracy', 3], ['AUC', 3],
                          ['Temps entraînement (s)', 2], ['Temps prédiction (s)', 2]]
    for colonne in colonnes_decimales:
        tableau[colonne[0]] = textes_colonne(resultats[colonne[0]], colonne[1])
    # Taille des fichiers : « < 0,1 » pour un fichier de moins de 0,1 Mo, tiret pour le modèle naïf
    tailles = textes_colonne(resultats['Taille du fichier (Mo)'], 2)
    for i in range(len(tailles)):
        if tailles[i] != '–':
            tailles[i] = texte_taille(resultats['Taille du fichier (Mo)'].values[i])
    tableau['Taille du fichier (Mo)'] = tailles
    tableau = tableau.rename(columns={'Temps entraînement (s)': "Temps d'entraînement (s)",
                                      'Temps prédiction (s)': 'Temps de prédiction (s)'})
    # Ligne du modèle retenu mise en évidence
    nouvelle_carte(onglets[2], None).dataframe(tableau.style.apply(surligner_retenu, axis=1), hide_index=True,
                                                width='stretch')

    # Seuil de décision
    titre_onglet(onglets[3], "Seuil de décision")
    carte = nouvelle_carte(onglets[3], None)
    modele = carte.selectbox("Modèle", MODELES, index=MODELES.index('XGBoost'), key='modele_seuil')
    seuil_retenu = float(charger_seuils()[horizon][modele])
    carte.write("Explorer l'effet du seuil de décision. Le seuil retenu pour le modèle reste "
                + texte_seuil(float(charger_seuils()['J+1'][modele])) + " (J+1) / "
                + texte_seuil(float(charger_seuils()['J+2'][modele])) + " (J+2).")
    valeurs_seuil = []
    for i in range(101):
        valeurs_seuil.append(round(i / 100, 2))
    if seuil_retenu not in valeurs_seuil:
        valeurs_seuil = sorted(valeurs_seuil + [seuil_retenu])
    # Curseur placé au seuil retenu à son premier affichage, puis remis au seuil retenu par le bouton
    cle_seuil = 'seuil_' + horizon + modele
    if cle_seuil not in st.session_state:
        st.session_state[cle_seuil] = seuil_retenu
    seuil = carte.select_slider("Seuil de décision", options=valeurs_seuil, format_func=texte_seuil, key=cle_seuil)
    carte.button("Revenir au seuil retenu", on_click=revenir_au_seuil, key='revenir_seuil')
    probas = charger_csv('probas_test_' + suffixe + '.csv')
    matrice, precision, rappel, f1 = scores_seuil(probas['Cible'], probas[modele], seuil)
    # Écart avec les scores au seuil retenu
    matrice, precision_retenu, rappel_retenu, f1_retenu = scores_seuil(probas['Cible'], probas[modele], seuil_retenu)
    cartes_chiffres(carte, [['Précision', nombre_fr(precision, 3),
                             ecart_fr(precision - precision_retenu, 3) + ' / seuil retenu'],
                            ['Rappel', nombre_fr(rappel, 3), ecart_fr(rappel - rappel_retenu, 3) + ' / seuil retenu'],
                            ['F1', nombre_fr(f1, 3), ecart_fr(f1 - f1_retenu, 3) + ' / seuil retenu']], 132)
    carte.plotly_chart(graphique_seuil(probas, modele, seuil, horizon), width='stretch', theme=None, key='seuil')
    carte.write(LECTURES['seuil'][horizon])

    # Interprétabilité du modèle
    titre_onglet(onglets[4], "Interprétabilité du modèle")
    valeurs_shap = charger_csv('shap_xgboost_' + suffixe + '.csv')
    carte_graphique(onglets[4], graphique_shap_synthese(valeurs_shap, horizon), LECTURES['shap'][horizon], 'shap')
    carte_graphique(onglets[4], graphique_shap_beeswarm(valeurs_shap, charger_variables_test(suffixe), horizon),
                    LECTURE_ESSAIM, 'shap_essaim')
    carte_graphique(onglets[4], graphique_odds_ratios(charger_csv('odds_ratios_' + suffixe + '.csv'), horizon),
                    LECTURES['odds'][horizon], 'odds')

    # Performance par station
    titre_onglet(onglets[5], "Performance par station")
    carte = nouvelle_carte(onglets[5], None)
    modele = carte.selectbox("Modèle", MODELES, index=MODELES.index('XGBoost'), key='modele_stations')
    carte.plotly_chart(graphique_stations(charger_csv('stations_' + suffixe + '.csv'), modele, horizon),
                       width='stretch', theme=None, key='stations')
    carte.write(LECTURES['stations'][horizon])

    # Choix du modèle : F1 des deux horizons côte à côte ; AUC, temps de prédiction et taille du fichier de l'horizon
    # choisi ; ligne du modèle retenu mise en évidence
    titre_onglet(onglets[6], "Choix du modèle")
    resultats_J1 = charger_csv('resultats_test_J1.csv')
    resultats_J2 = charger_csv('resultats_test_J2.csv')
    lignes = []
    for nom in MODELES:
        r1 = resultats_J1[resultats_J1['Modèle'] == nom]
        r2 = resultats_J2[resultats_J2['Modèle'] == nom]
        r = resultats[resultats['Modèle'] == nom]
        lignes.append({'Modèle': nom, 'F1 J+1': nombre_fr(r1['F1'].values[0], 3),
                       'F1 J+2': nombre_fr(r2['F1'].values[0], 3), 'AUC ' + horizon: nombre_fr(r['AUC'].values[0], 3),
                       'Temps de prédiction ' + horizon + ' (s)': nombre_fr(r['Temps prédiction (s)'].values[0], 2),
                       'Taille du fichier ' + horizon + ' (Mo)': texte_taille(r['Taille du fichier (Mo)'].values[0]),
                       'Interprétabilité': INTERPRETABILITE[nom]})
    nouvelle_carte(onglets[6], None).dataframe(pd.DataFrame(lignes).style.apply(surligner_retenu, axis=1),
                                                hide_index=True, width='stretch')
    onglets[6].write("XGBoost est retenu pour les deux horizons : meilleur F1 à J+1 (0,660) et meilleure AUC (0,882), "
                     "entraînement en quelques secondes, prédiction en quelques centièmes de seconde, fichier de 7 Mo "
                     "(contre 353 Mo pour la Random Forest), et interprétable avec SHAP. À J+2, le réseau de neurones "
                     "fait légèrement mieux (0,471 contre 0,436), au prix d'un entraînement bien plus long.")

# ---------------------------------------------------------------------------------------------------------------
# Page « Prédiction »
# ---------------------------------------------------------------------------------------------------------------

if page == PAGES[4]:
    brut = charger_brut()
    stations = list(np.sort(brut['Location'].unique()))
    onglets = st.tabs(["Prédiction sur données réelles", "Simulation"])

    # Prédiction sur données réelles : un jour de la période de test
    titre_onglet(onglets[0], "Prédiction sur données réelles")
    # Valeurs initiales (jour de référence), remises en place si elles ont été effacées en quittant la page
    if 'compteur_hasard' not in st.session_state:
        st.session_state['compteur_hasard'] = 0
    if 'station_reelle' not in st.session_state:
        st.session_state['station_reelle'] = 'Sydney'
    if 'date_reelle' not in st.session_state:
        st.session_state['date_reelle'] = dt.date(2015, 11, 13)
    colonnes = onglets[0].columns([2, 2, 1], vertical_alignment='bottom')
    station = colonnes[0].selectbox("Station", stations, key='station_reelle')
    date = colonnes[1].date_input("Date (période de test)", min_value=dt.date(2015, 11, 10),
                                  max_value=dt.date(2017, 6, 25), format="DD/MM/YYYY", key='date_reelle')
    colonnes[2].button("Station et jour au hasard", on_click=jour_au_hasard, key='hasard')
    observation = brut[(brut['Location'] == station) & (brut['Date'] == str(date))]
    if len(observation) == 0:
        onglets[0].info("Aucun relevé pour cette station à cette date : choisissez un autre jour.")
    else:
        # Grille compacte : 4 cartes de même hauteur par ligne
        onglets[0].subheader("Observations du jour")
        grille = onglets[0].container(key='compact_observations')
        for rangee in RANGEES_OBSERVATIONS:
            chiffres = []
            for variable in rangee:
                chiffres.append([libelle(variable), texte_observation(observation, variable)])
            cartes_chiffres(grille, chiffres, 100)
        # Indicateur d'attente pendant le calcul de la prévision et l'affichage de ses graphiques
        with onglets[0].spinner("Calcul de la prévision..."):
            prevision = prevision_jour_reel(observation, station, date)
            afficher_prevision(onglets[0], observation, prevision, 'reel')

    # Simulation : observations saisies, préremplies avec les valeurs habituelles de la station et du mois
    titre_onglet(onglets[1], "Simulation")
    if 'compteur_simulation' not in st.session_state:
        st.session_state['compteur_simulation'] = 0
    if 'station_simulation' not in st.session_state:
        st.session_state['station_simulation'] = 'Sydney'
    if 'mois_simulation' not in st.session_state:
        st.session_state['mois_simulation'] = 1
    colonnes = onglets[1].columns([2, 2, 1], vertical_alignment='bottom')
    station_simulee = colonnes[0].selectbox("Station", stations, key='station_simulation')
    mois = colonnes[1].selectbox("Mois", list(range(1, 13)), format_func=libelle_mois, key='mois_simulation')
    colonnes[2].button("Journée type au hasard", on_click=journee_au_hasard, key='hasard_simulation')
    defauts = valeurs_par_defaut(station_simulee, mois)
    onglets[1].caption("Valeurs proposées : médiane (ou valeur la plus fréquente) de la station pour ce mois, sur la "
                       "période d'entraînement, ou mesures d'une journée de cette période tirée au hasard.")
    # Trois cartes de même hauteur ; dans chacune, les champs sont répartis sur toute la hauteur
    groupes_saisie = [['Températures et précipitations', ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'Rainfall',
                                                          'RainToday']],
                      ['Humidité et pression', ['Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']],
                      ['Vent', ['WindGustSpeed', 'WindGustDir', 'WindSpeed9am', 'WindDir9am', 'WindSpeed3pm',
                                'WindDir3pm']]]
    colonnes = onglets[1].columns(3)
    cle = 'simulation_' + station_simulee + '_' + str(mois) + '_'
    # Date fictive : seul le mois sert (mois et saison)
    ligne = {'Date': [f"2016-{mois:02d}-15"], 'Location': [station_simulee]}
    for i in range(len(groupes_saisie)):
        carte = colonnes[i].container(border=True, height=540, vertical_alignment='distribute',
                                      key='carte_simulation_' + str(i))
        carte.markdown('**' + groupes_saisie[i][0] + '**')
        for variable in groupes_saisie[i][1]:
            ligne[variable] = [saisir(carte, variable, defauts, cle)]
    if onglets[1].button("Prévoir la pluie", key='prevoir'):
        st.session_state['simulation_lancee'] = True
    if 'simulation_lancee' in st.session_state:
        # Indicateur d'attente pendant le calcul de la prévision et l'affichage de ses graphiques
        with onglets[1].spinner("Calcul de la prévision..."):
            observation_simulee = pd.DataFrame(ligne)
            prevision = prevoir(observation_simulee)
            prevision['station'] = station_simulee
            prevision['reel'] = {'J+1': 'simulation', 'J+2': 'simulation'}
            prevision['contributions'], prevision['base'] = contributions_shap(prevision['X_J1'])
            afficher_prevision(onglets[1], observation_simulee, prevision, 'simulation')
