# graphiques.py : fonctions des graphiques interactifs (Plotly) communes au notebook 04 et à l'application Streamlit
# Chaque fonction de graphique reçoit ses données (DataFrame) et au plus 3 autres paramètres, et renvoie une figure Plotly.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Ordre des modèles et palette : une couleur et un symbole par modèle, les mêmes dans tous les graphiques
MODELES = ['Régression logistique', 'Random Forest', 'XGBoost', 'KNN', 'Réseau de neurones']
SYMBOLES_MODELES = {'Régression logistique': 'circle', 'Random Forest': 'square', 'XGBoost': 'diamond',
                    'KNN': 'triangle-up', 'Réseau de neurones': 'cross'}

# Thème visuel (clair), le même que celui de l'application (.streamlit/config.toml et feuille de style de app.py)
COULEUR_FOND_PAGE = '#F6F9FC'
COULEUR_FOND_CARTE = '#FFFFFF'
COULEUR_FOND_DISCRET = '#EEF4FA'
COULEUR_TEXTE = '#10233A'
COULEUR_TEXTE_SECONDAIRE = '#566B80'
COULEUR_GRILLE = '#D6E1EC'
COULEUR_ACCENT = '#0B4F8A'
COULEUR_ACCENT_TEINTE = '#DCEEFA'
COULEUR_AVERTISSEMENT = '#E8A33D'
COULEUR_AVERTISSEMENT_TEINTE = '#FBEBD2'
COULEUR_DANGER = '#BA323C'
COULEUR_SUCCES = '#24785B'
COULEUR_VIOLET = '#7855A9'
POLICE = 'Geist, system-ui, -apple-system, "Segoe UI", sans-serif'
POLICE_TITRES = '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif'

# Une couleur par modèle, la même dans tous les graphiques
COULEURS_MODELES = {'Régression logistique': COULEUR_AVERTISSEMENT, 'Random Forest': COULEUR_SUCCES,
                    'XGBoost': COULEUR_ACCENT, 'KNN': COULEUR_TEXTE_SECONDAIRE, 'Réseau de neurones': COULEUR_VIOLET}
COULEUR_NAIF = '#B2B8BD'

# Une couleur pour la pluie (la couleur d'accent du thème), une pour l'absence de pluie
COULEUR_PLUIE = COULEUR_ACCENT
COULEUR_PAS_DE_PLUIE = COULEUR_AVERTISSEMENT
# Barres sans identité particulière, lignes de référence (seuils, coupure), deux méthodes comparées
COULEUR_NEUTRE = COULEUR_TEXTE_SECONDAIRE
COULEUR_REFERENCE = COULEUR_DANGER
COULEURS_COMPARAISON = ['#9DB4CA', COULEUR_TEXTE]
ECHELLE_MATRICE = [[0, COULEUR_FOND_DISCRET], [1, '#8FB3D6']]
# Échelle de couleur des valeurs des variables (graphique SHAP en essaim) : faible -> moyenne -> élevée
ECHELLE_VALEURS = [[0, COULEUR_ACCENT], [0.5, COULEUR_ACCENT_TEINTE], [1, COULEUR_DANGER]]
# Dégradés des barres à une seule mesure, de la plus petite à la plus grande valeur affichée : bleu (valeurs
# positives, odds ratios au-dessus de 1) et or (valeurs négatives, odds ratios au-dessous de 1, selon leur ampleur)
ECHELLE_DEGRADE = [[0, COULEUR_ACCENT_TEINTE], [1, COULEUR_ACCENT]]
ECHELLE_NEGATIVE = [[0, COULEUR_AVERTISSEMENT_TEINTE], [1, COULEUR_AVERTISSEMENT]]

# Étapes de la progression des performances (colonnes des fichiers progression_J1.csv et progression_J2.csv)
ETAPES = ['Paramètres par défaut', 'Option de déséquilibre', 'Hyperparamètres', 'Seuil de décision', 'Test']

# Nom français et unité de chaque variable (brute ou préparée)
NOMS_VARIABLES = {
    'MinTemp': 'Température minimale', 'MaxTemp': 'Température maximale', 'Rainfall': 'Précipitations du jour',
    'Evaporation': 'Évaporation', 'Sunshine': 'Ensoleillement', 'WindGustDir': 'Direction de la rafale maximale',
    'WindGustSpeed': 'Vitesse de la rafale maximale', 'WindDir9am': 'Direction du vent à 9 h',
    'WindDir3pm': 'Direction du vent à 15 h', 'WindSpeed9am': 'Vitesse du vent à 9 h',
    'WindSpeed3pm': 'Vitesse du vent à 15 h', 'Humidity9am': 'Humidité à 9 h', 'Humidity3pm': 'Humidité à 15 h',
    'Pressure9am': 'Pression à 9 h', 'Pressure3pm': 'Pression à 15 h', 'Cloud9am': 'Nébulosité à 9 h',
    'Cloud3pm': 'Nébulosité à 15 h', 'Temp9am': 'Température à 9 h', 'Temp3pm': 'Température à 15 h',
    'RainToday': 'Pluie le jour même', 'RainTomorrow': 'Pluie le lendemain', 'Date': 'Date', 'Location': 'Station',
    'Temp_diff': 'Écart de température 15 h - 9 h', 'Humidity_diff': "Écart d'humidité 15 h - 9 h",
    'Pressure_diff': 'Écart de pression 15 h - 9 h', 'Month': 'Mois', 'Season': 'Saison',
    'Month_sin': 'Mois (sinus)', 'Month_cos': 'Mois (cosinus)', 'Season_sin': 'Saison (sinus)',
    'Season_cos': 'Saison (cosinus)', 'WindGustDir_sin': 'Direction de la rafale (sinus)',
    'WindGustDir_cos': 'Direction de la rafale (cosinus)', 'WindDir9am_sin': 'Direction du vent à 9 h (sinus)',
    'WindDir9am_cos': 'Direction du vent à 9 h (cosinus)', 'WindDir3pm_sin': 'Direction du vent à 15 h (sinus)',
    'WindDir3pm_cos': 'Direction du vent à 15 h (cosinus)', 'Location_encoded': 'Taux de pluie de la station (J+1)',
    'Location_encoded_J2': 'Taux de pluie de la station (J+2)'
}
UNITES_VARIABLES = {
    'MinTemp': '°C', 'MaxTemp': '°C', 'Temp9am': '°C', 'Temp3pm': '°C', 'Temp_diff': '°C', 'Rainfall': 'mm',
    'Evaporation': 'mm', 'Sunshine': 'h', 'WindGustSpeed': 'km/h', 'WindSpeed9am': 'km/h', 'WindSpeed3pm': 'km/h',
    'Humidity9am': '%', 'Humidity3pm': '%', 'Humidity_diff': 'points de %', 'Pressure9am': 'hPa',
    'Pressure3pm': 'hPa', 'Pressure_diff': 'hPa', 'Cloud9am': 'octas', 'Cloud3pm': 'octas'
}

# Saison de chaque mois (hémisphère sud), comme dans 02_preprocessing
SAISON_DU_MOIS = {12: 'Ete', 1: 'Ete', 2: 'Ete', 3: 'Automne', 4: 'Automne', 5: 'Automne',
                  6: 'Hiver', 7: 'Hiver', 8: 'Hiver', 9: 'Printemps', 10: 'Printemps', 11: 'Printemps'}


def libelle(variable):
    # Nom français de la variable (nom d'origine s'il n'est pas dans le dictionnaire)
    if variable in NOMS_VARIABLES:
        return NOMS_VARIABLES[variable]
    return variable


def libelle_axe(variable):
    # Nom français suivi de l'unité entre parenthèses, pour un titre d'axe
    if variable in UNITES_VARIABLES:
        return libelle(variable) + ' (' + UNITES_VARIABLES[variable] + ')'
    return libelle(variable)


def nombre_fr(valeur, decimales):
    # Nombre écrit avec 1, 2 ou 3 décimales, une virgule décimale et une espace entre les milliers (1 023,0)
    texte = f"{abs(valeur):.3f}"
    if decimales == 1:
        texte = f"{abs(valeur):.1f}"
    elif decimales == 2:
        texte = f"{abs(valeur):.2f}"
    morceaux = texte.split('.')
    resultat = entier_fr(int(morceaux[0])) + ',' + morceaux[1]
    if valeur < 0:
        resultat = '-' + resultat
    return resultat


# Pas des graduations (multiples de 1, 2 ou 5) et nombre de décimales de chacun, du plus grand au plus petit
PAS_GRADUATIONS = [[50000, 0], [20000, 0], [10000, 0], [5000, 0], [2000, 0], [1000, 0], [500, 0], [200, 0], [100, 0],
                   [50, 0], [20, 0], [10, 0], [5, 0], [2, 0], [1, 0], [0.5, 1], [0.2, 1], [0.1, 1], [0.05, 2],
                   [0.02, 2], [0.01, 2], [0.005, 3], [0.002, 3], [0.001, 3]]


def entier_fr(nombre):
    # Nombre entier écrit avec une espace entre les milliers (145 460)
    texte = str(int(round(nombre)))
    resultat = ''
    for i in range(len(texte)):
        if i > 0 and (len(texte) - i) % 3 == 0:
            resultat = resultat + ' '
        resultat = resultat + texte[i]
    return resultat


def regler_axe(fig, axe, minimum, maximum):
    # Graduations d'un axe : pas multiple de 1, 2 ou 5 (au plus 8 intervalles) et même nombre de décimales pour
    # toutes les graduations (le nombre de décimales du pas) ; axe : 'x' ou 'y'
    # Pas possibles, du plus grand au plus petit, avec leur nombre de décimales ; le dernier pas retenu est le plus
    # petit qui donne au plus 8 intervalles
    etendue = maximum - minimum
    choix = PAS_GRADUATIONS[0]
    for pas in PAS_GRADUATIONS:
        if etendue / pas[0] <= 8:
            choix = pas
    if axe == 'x':
        fig.update_xaxes(dtick=choix[0], tickformat=',.' + str(choix[1]) + 'f')
    else:
        fig.update_yaxes(dtick=choix[0], tickformat=',.' + str(choix[1]) + 'f')
    return fig


def appliquer_theme(fig, titre):
    # Thème commun à toutes les figures : fond transparent autour du graphique, zone de tracé blanche, une seule
    # police, couleurs de texte du thème, titre aligné à gauche, virgule décimale, marges identiques
    fig.update_layout(title={'text': titre, 'x': 0, 'xanchor': 'left',
                             'font': {'family': POLICE_TITRES, 'size': 18, 'weight': 600, 'color': COULEUR_TEXTE}},
                      template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=COULEUR_FOND_CARTE,
                      separators=', ', font={'family': POLICE, 'size': 14, 'color': COULEUR_TEXTE},
                      hoverlabel={'bgcolor': COULEUR_FOND_CARTE, 'font': {'family': POLICE, 'color': COULEUR_TEXTE}},
                      legend={'orientation': 'h', 'x': 0, 'y': 1.02, 'yanchor': 'bottom',
                              'font': {'color': COULEUR_TEXTE_SECONDAIRE}},
                      margin={'l': 80, 'r': 40, 't': 110, 'b': 70})
    return fig


def marqueur_degrade(valeurs, echelle):
    # Couleur de chaque barre selon sa valeur (ou son ampleur) : couleur claire de l'échelle pour la plus petite
    # valeur affichée, couleur foncée pour la plus grande
    return {'color': valeurs, 'colorscale': echelle, 'cmin': valeurs.min(), 'cmax': valeurs.max()}


def mise_en_forme(fig, titre, titre_x, titre_y):
    # Thème commun, puis axes : grille claire, graduations en texte secondaire, titres d'axes en couleur du texte
    fig = appliquer_theme(fig, titre)
    fig.update_xaxes(title_text=titre_x, gridcolor=COULEUR_GRILLE, zerolinecolor=COULEUR_GRILLE,
                     linecolor=COULEUR_GRILLE, tickfont={'color': COULEUR_TEXTE_SECONDAIRE},
                     title_font={'color': COULEUR_TEXTE})
    fig.update_yaxes(title_text=titre_y, ticksuffix='  ', gridcolor=COULEUR_GRILLE, zerolinecolor=COULEUR_GRILLE,
                     linecolor=COULEUR_GRILLE, tickfont={'color': COULEUR_TEXTE_SECONDAIRE},
                     title_font={'color': COULEUR_TEXTE})
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Préparation d'observations brutes (même chaîne que 02_preprocessing, à partir des artefacts sauvegardés)
# ---------------------------------------------------------------------------------------------------------------

def imputer_brut(X, artefacts):
    # Paires 9 h / 15 h avec une seule valeur manquante : imputée à partir de l'autre valeur et de l'écart typique,
    # puis le reste : statistique de la station, puis statistique globale
    for var_diff in artefacts['paires_9h_15h']:
        col_9h = artefacts['paires_9h_15h'][var_diff][0]
        col_15h = artefacts['paires_9h_15h'][var_diff][1]
        ecart_type = X['Location'].map(artefacts['ecarts_types_local'][var_diff]).fillna(
            artefacts['ecarts_types_global'][var_diff])
        manque_9h = X[col_9h].isnull() & (X[col_15h].isnull() == False)
        manque_15h = X[col_15h].isnull() & (X[col_9h].isnull() == False)
        X[col_9h] = np.where(manque_9h, X[col_15h] - ecart_type, X[col_9h])
        X[col_15h] = np.where(manque_15h, X[col_9h] + ecart_type, X[col_15h])
    for col in artefacts['impute_local'].columns:
        X[col] = X[col].fillna(X['Location'].map(artefacts['impute_local'][col]))
        X[col] = X[col].fillna(artefacts['impute_global'][col])
    return X


def creer_variables_brut(X):
    # Mois, saison et écarts 15 h - 9 h
    X['Month'] = X['Date'].dt.month
    X['Season'] = X['Month'].map(SAISON_DU_MOIS)
    X['Temp_diff'] = X['Temp3pm'] - X['Temp9am']
    X['Humidity_diff'] = X['Humidity3pm'] - X['Humidity9am']
    X['Pressure_diff'] = X['Pressure3pm'] - X['Pressure9am']
    return X


def encoder_brut(X, artefacts):
    # RainToday en 0 / 1, encodage cyclique (sin / cos), taux de pluie des stations
    X['RainToday'] = X['RainToday'].map({'No': 0, 'Yes': 1})
    X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
    X['Season'] = X['Season'].map(artefacts['numero_saison'])
    X['Season_sin'] = np.sin(2 * np.pi * X['Season'] / 4)
    X['Season_cos'] = np.cos(2 * np.pi * X['Season'] / 4)
    for col in artefacts['colonnes_directions']:
        X[col] = X[col].map(artefacts['numero_direction'])
        X[col + '_sin'] = np.sin(2 * np.pi * X[col] / 16)
        X[col + '_cos'] = np.cos(2 * np.pi * X[col] / 16)
    X['Location_encoded'] = X['Location'].map(artefacts['loc_rate_J1']).fillna(artefacts['taux_global_J1'])
    X['Location_encoded_J2'] = X['Location'].map(artefacts['loc_rate_J2']).fillna(artefacts['taux_global_J2'])
    return X


def mettre_a_echelle_brut(X, artefacts):
    # Les 3 scalers ajustés sur X_train en 02, chacun sur ses colonnes
    for nom_scaler in artefacts['scalers']:
        colonnes = artefacts['colonnes_scalers'][nom_scaler]
        X[colonnes] = artefacts['scalers'][nom_scaler].transform(X[colonnes])
    return X


def preparer_brut(lignes, artefacts, mise_a_echelle, colonnes):
    # Lignes brutes de weatherAUS.csv -> lignes préparées (colonnes du jeu J+1 ou J+2), mises à l'échelle ou non
    X = lignes.copy()
    for col in artefacts['colonnes_supprimees_manquants'] + ['RainTomorrow']:
        if col in X.columns:
            X = X.drop(columns=[col])
    X['Date'] = pd.to_datetime(X['Date'])
    X = encoder_brut(creer_variables_brut(imputer_brut(X, artefacts)), artefacts)
    X = X[artefacts['ordre_colonnes_avant_scaling']]
    if mise_a_echelle:
        X = mettre_a_echelle_brut(X, artefacts)
    return X[colonnes]


# ---------------------------------------------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------------------------------------------

COULEURS_CLASSES = {'Pas de pluie': COULEUR_PAS_DE_PLUIE, 'Pluie': COULEUR_PLUIE}


def graphique_repartition_cible(repartition):
    # repartition : colonnes Classe, Nombre de jours, Pourcentage
    fig = go.Figure()
    fig.add_trace(go.Bar(x=repartition['Classe'], y=repartition['Nombre de jours'],
                         marker_color=repartition['Classe'].map(COULEURS_CLASSES),
                         customdata=repartition['Pourcentage'], texttemplate='%{customdata:,.2f} %',
                         textposition='outside',
                         hovertemplate='%{x}<br>%{y:,d} jours<br>%{customdata:,.2f} % des jours<extra></extra>'))
    fig = mise_en_forme(fig, "Répartition de la cible (pluie le lendemain)", "Pluie le lendemain (RainTomorrow)",
                        "Nombre de jours")
    fig.update_yaxes(range=[0, repartition['Nombre de jours'].max() * 1.15])
    fig = regler_axe(fig, 'y', 0, repartition['Nombre de jours'].max() * 1.15)
    return fig


def graphique_valeurs_manquantes(manquants):
    # manquants : colonnes Variable, Pourcentage ; seuil de suppression à 20 % ; variables conservées en dégradé,
    # variables supprimées (au-delà du seuil) en gris
    table = manquants.sort_values('Pourcentage', ascending=True)
    noms = []
    for variable in table['Variable']:
        noms.append(libelle(variable))
    table['Nom'] = noms
    conservees = table[table['Pourcentage'] <= 20]
    supprimees = table[table['Pourcentage'] > 20]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=conservees['Pourcentage'], y=conservees['Nom'], orientation='h',
                         marker=marqueur_degrade(conservees['Pourcentage'], ECHELLE_DEGRADE),
                         customdata=conservees['Variable'],
                         name='Variables conservées (imputées)',
                         hovertemplate='%{y} (%{customdata})<br>%{x:,.2f} % de valeurs manquantes<extra></extra>'))
    fig.add_trace(go.Bar(x=supprimees['Pourcentage'], y=supprimees['Nom'], orientation='h', marker_color=COULEUR_NEUTRE,
                         customdata=supprimees['Variable'], name='Variables supprimées',
                         hovertemplate='%{y} (%{customdata})<br>%{x:,.2f} % de valeurs manquantes<extra></extra>'))
    # Ordre des variables : celui du tri (la plus incomplète en haut), quelle que soit la série
    fig.update_yaxes(categoryorder='array', categoryarray=noms)
    fig.add_shape(type='line', x0=20, x1=20, y0=0, y1=1, yref='paper',
                  line={'color': COULEUR_REFERENCE, 'dash': 'dash', 'width': 2})
    fig.add_annotation(x=20, y=1, yref='paper', text='Seuil de suppression : 20 %', showarrow=False,
                       xanchor='left', yanchor='bottom', font={'color': COULEUR_REFERENCE})
    fig = mise_en_forme(fig, "Valeurs manquantes par variable", "Valeurs manquantes (%)", "Variable")
    fig = regler_axe(fig, 'x', 0, table['Pourcentage'].max())
    fig.update_layout(height=700)
    return fig


def quartiles_classe(valeurs):
    # Quartiles et moustaches (valeurs extrêmes situées à moins de 1,5 IQR des quartiles)
    q1 = valeurs.quantile(0.25)
    q3 = valeurs.quantile(0.75)
    iqr = q3 - q1
    basse = valeurs[valeurs >= q1 - 1.5 * iqr].min()
    haute = valeurs[valeurs <= q3 + 1.5 * iqr].max()
    return [q1, valeurs.median(), q3, basse, haute]


def ajouter_boite(fig, valeurs, classe):
    # Boîte d'une classe à partir de ses quartiles, et point invisible portant le texte de survol en français
    q = quartiles_classe(valeurs)
    fig.add_trace(go.Box(x=[classe], q1=[q[0]], median=[q[1]], q3=[q[2]], lowerfence=[q[3]], upperfence=[q[4]],
                         name=classe, marker_color=COULEURS_CLASSES[classe], hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[classe], y=[q[1]], mode='markers', marker={'opacity': 0, 'size': 30},
                             showlegend=False, customdata=[q],
                             hovertemplate=classe + '<br>Moustache haute : %{customdata[4]:,.1f}<br>'
                             + '3e quartile : %{customdata[2]:,.1f}<br>Médiane : %{customdata[1]:,.1f}<br>'
                             + '1er quartile : %{customdata[0]:,.1f}<br>Moustache basse : %{customdata[3]:,.1f}'
                             + '<br>' + entier_fr(len(valeurs)) + ' jours<extra></extra>'))
    return fig


def graphique_distribution_pluie(donnees, variable):
    # donnees : données brutes (weatherAUS.csv) ; une boîte par classe de RainTomorrow, valeurs manquantes exclues
    fig = go.Figure()
    fig = ajouter_boite(fig, donnees.loc[donnees['RainTomorrow'] == 'No', variable].dropna(), 'Pas de pluie')
    fig = ajouter_boite(fig, donnees.loc[donnees['RainTomorrow'] == 'Yes', variable].dropna(), 'Pluie')
    fig = mise_en_forme(fig, libelle(variable) + " selon la pluie du lendemain", "Pluie le lendemain (RainTomorrow)",
                        libelle_axe(variable))
    # Axe vertical : des moustaches basses aux moustaches hautes des deux classes
    q_non = quartiles_classe(donnees.loc[donnees['RainTomorrow'] == 'No', variable].dropna())
    q_oui = quartiles_classe(donnees.loc[donnees['RainTomorrow'] == 'Yes', variable].dropna())
    fig = regler_axe(fig, 'y', min(q_non[3], q_oui[3]), max(q_non[4], q_oui[4]))
    fig.update_layout(showlegend=False)
    return fig


def graphique_correlation_cible(correlations):
    # correlations : colonnes Variable, Corrélation (Pearson avec RainTomorrow codée 0 / 1) ; corrélations positives
    # en dégradé bleu, corrélations négatives en dégradé or selon leur valeur absolue
    table = correlations.sort_values('Corrélation', ascending=True)
    noms = []
    for variable in table['Variable']:
        noms.append(libelle(variable))
    table['Nom'] = noms
    positives = table[table['Corrélation'] > 0]
    negatives = table[table['Corrélation'] <= 0]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=positives['Corrélation'], y=positives['Nom'], orientation='h',
                         marker=marqueur_degrade(positives['Corrélation'], ECHELLE_DEGRADE),
                         customdata=positives['Variable'], showlegend=False,
                         hovertemplate='%{y} (%{customdata})<br>Corrélation : %{x:,.3f}<extra></extra>'))
    fig.add_trace(go.Bar(x=negatives['Corrélation'], y=negatives['Nom'], orientation='h',
                         marker=marqueur_degrade(-negatives['Corrélation'], ECHELLE_NEGATIVE),
                         customdata=negatives['Variable'], showlegend=False,
                         hovertemplate='%{y} (%{customdata})<br>Corrélation : %{x:,.3f}<extra></extra>'))
    # Ordre des variables : celui du tri, quelle que soit la série
    fig.update_yaxes(categoryorder='array', categoryarray=noms)
    fig = mise_en_forme(fig, "Corrélation des variables numériques avec la pluie du lendemain",
                        "Corrélation de Pearson avec la pluie du lendemain (RainTomorrow = 1)", "Variable")
    fig = regler_axe(fig, 'x', min(0, table['Corrélation'].min()), max(0, table['Corrélation'].max()))
    fig.update_layout(height=700)
    return fig


def graphique_pluie_par_station(stations):
    # stations : colonnes Station, Taux de pluie (%), Jours ; stations triées, la plus pluvieuse en haut
    table = stations.sort_values('Taux de pluie (%)', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table['Taux de pluie (%)'], y=table['Station'], orientation='h',
                         marker=marqueur_degrade(table['Taux de pluie (%)'], ECHELLE_DEGRADE), customdata=table['Jours'],
                         hovertemplate='%{y}<br>Pluie le lendemain : %{x:,.1f} % des jours<br>'
                         + '%{customdata:,d} jours avec la cible connue<extra></extra>'))
    fig = mise_en_forme(fig, "Fréquence de la pluie par station", "Jours de pluie le lendemain (%)", "Station")
    fig = regler_axe(fig, 'x', 0, table['Taux de pluie (%)'].max())
    fig.update_layout(height=1100)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Pré-traitement
# ---------------------------------------------------------------------------------------------------------------

def graphique_taux_mensuel(taux, date_coupure):
    # taux : colonnes Mois (date du premier jour du mois), Taux de pluie (%), Jours ; date_coupure : début du test
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=taux['Mois'], y=taux['Taux de pluie (%)'], mode='lines+markers',
                             line={'color': COULEUR_PLUIE, 'width': 2}, marker={'size': 5},
                             customdata=taux['Jours'], name='Taux de pluie mensuel',
                             hovertemplate='%{x|%m/%Y}<br>Pluie le lendemain : %{y:,.1f} % des jours<br>'
                             + '%{customdata:,d} jours<extra></extra>'))
    fig.add_shape(type='line', x0=date_coupure, x1=date_coupure, y0=0, y1=1, yref='paper',
                  line={'color': COULEUR_REFERENCE, 'dash': 'dash', 'width': 2})
    texte_date = f"{date_coupure.day:02d}/{date_coupure.month:02d}/{date_coupure.year}"
    fig.add_annotation(x=date_coupure, y=1, yref='paper', text='Coupure : ' + texte_date, showarrow=False,
                       xanchor='left', yanchor='bottom', font={'color': COULEUR_REFERENCE})
    fig.add_annotation(x=date_coupure, y=0.95, yref='paper', text='Entraînement (80 %) ', showarrow=False,
                       xanchor='right')
    fig.add_annotation(x=date_coupure, y=0.95, yref='paper', text=' Test (20 %)', showarrow=False, xanchor='left')
    fig = mise_en_forme(fig, "Taux de pluie mensuel et découpage temporel", "Mois",
                        "Jours de pluie le lendemain (%)")
    fig.update_xaxes(tickformat='%Y')
    fig = regler_axe(fig, 'y', taux['Taux de pluie (%)'].min(), taux['Taux de pluie (%)'].max())
    return fig


def graphique_biais_evite(biais):
    # biais : une ligne par paire 9 h / 15 h, pourcentages d'écarts 15 h - 9 h hors des bornes IQR
    fig = go.Figure()
    fig.add_trace(go.Bar(x=biais['Paire'], y=biais['Écarts hypothétiques hors bornes (%)'],
                         name='Écarts obtenus en remplissant par la moyenne de la station',
                         marker_color=COULEURS_COMPARAISON[0], customdata=biais['Lignes avec une valeur manquante'],
                         texttemplate='%{y:,.2f} %', textposition='outside',
                         hovertemplate='%{x}<br>%{y:,.2f} % hors bornes<br>%{customdata:,d} lignes avec une seule valeur '
                         + 'manquante<extra></extra>'))
    fig.add_trace(go.Bar(x=biais['Paire'], y=biais['Écarts mesurés hors bornes (%)'],
                         name='Écarts réellement mesurés (paires complètes)', marker_color=COULEURS_COMPARAISON[1],
                         customdata=biais['Corrélation 9 h / 15 h'], texttemplate='%{y:,.2f} %', textposition='outside',
                         hovertemplate='%{x}<br>%{y:,.2f} % hors bornes<br>Corrélation 9 h / 15 h : %{customdata:,.4f}'
                         + '<extra></extra>'))
    fig = mise_en_forme(fig, "Écarts 15 h - 9 h hors des bornes IQR des écarts mesurés",
                        "Paire de mesures 9 h / 15 h", "Écarts hors des bornes IQR (%)")
    fig.update_layout(barmode='group')
    fig.update_yaxes(range=[0, biais['Écarts hypothétiques hors bornes (%)'].max() * 1.15])
    fig = regler_axe(fig, 'y', 0, biais['Écarts hypothétiques hors bornes (%)'].max() * 1.15)
    return fig


def points_comparaison(comparaison, colonne_1, colonne_2, titre):
    # Un segment par modèle reliant les F1 moyens des deux méthodes, un point de couleur par méthode
    fig = go.Figure()
    for i in range(len(comparaison)):
        fig.add_trace(go.Scatter(x=[comparaison[colonne_1].values[i], comparaison[colonne_2].values[i]],
                                 y=[comparaison['Modèle'].values[i], comparaison['Modèle'].values[i]], mode='lines',
                                 line={'color': COULEUR_GRILLE, 'width': 3}, showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=comparaison[colonne_1], y=comparaison['Modèle'], mode='markers', name=colonne_1,
                             marker={'color': COULEURS_COMPARAISON[0], 'size': 20, 'symbol': 'circle-open',
                                     'line': {'width': 3}},
                             hovertemplate='%{y}<br>' + colonne_1 + ' : F1 moyen = %{x:,.4f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=comparaison[colonne_2], y=comparaison['Modèle'], mode='markers', name=colonne_2,
                             marker={'color': COULEURS_COMPARAISON[1], 'size': 10},
                             hovertemplate='%{y}<br>' + colonne_2 + ' : F1 moyen = %{x:,.4f}<extra></extra>'))
    fig = mise_en_forme(fig, titre, "F1 moyen sur les 5 plis de validation (classe pluie, J+1)", "Modèle")
    fig = regler_axe(fig, 'x', min(comparaison[colonne_1].min(), comparaison[colonne_2].min()),
                     max(comparaison[colonne_1].max(), comparaison[colonne_2].max()))
    fig.update_yaxes(autorange='reversed')
    return fig


def graphique_imputation(comparaison):
    # comparaison : colonnes Modèle, Imputation globale, Imputation locale
    return points_comparaison(comparaison, 'Imputation globale', 'Imputation locale',
                              "Imputation globale et imputation locale (par station)")


def graphique_jeu_reduit(comparaison):
    # comparaison : colonnes Modèle, Jeu complet, Jeu réduit
    return points_comparaison(comparaison, 'Jeu complet', 'Jeu réduit',
                              "Jeu complet et jeu réduit (filtre de redondance)")


def graphique_encodage_cyclique(encodage, type_encodage):
    # encodage : colonnes Type, Libellé, Code, Numéro, sin, cos ; type_encodage : 'Mois' ou 'Direction du vent'
    table = encodage[encodage['Type'] == type_encodage]
    angles = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.sin(angles), y=np.cos(angles), mode='lines', line={'color': COULEUR_GRILLE},
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=table['sin'], y=table['cos'], mode='markers+text', text=table['Libellé'],
                             textposition='top center', marker={'color': COULEUR_NEUTRE, 'size': 10},
                             showlegend=False, customdata=table[['Code', 'Numéro']],
                             hovertemplate='%{text} (%{customdata[0]}), numéro %{customdata[1]}<br>'
                             + 'sin = %{x:,.3f}<br>cos = %{y:,.3f}<extra></extra>'))
    n = str(len(table))
    fig = mise_en_forme(fig, "Encodage cyclique : " + type_encodage.lower() + " sur le cercle",
                        "sin(2π × numéro / " + n + ")", "cos(2π × numéro / " + n + ")")
    fig.update_xaxes(range=[-1.3, 1.3])
    fig.update_yaxes(range=[-1.3, 1.3], scaleanchor='x', scaleratio=1)
    fig = regler_axe(fig, 'x', -1, 1)
    fig = regler_axe(fig, 'y', -1, 1)
    fig.update_layout(height=700)
    return fig


def graphique_paires_redondantes(paires):
    # paires : colonnes Variable 1, Variable 2, |r|, Décision ; seuil de redondance |r| = 0,8 ; barres en dégradé bleu
    # (la plus claire pour la plus petite corrélation affichée)
    table = paires.sort_values('|r|', ascending=True)
    noms = []
    decisions = []
    for i in range(len(table)):
        variable_1 = table['Variable 1'].values[i]
        variable_2 = table['Variable 2'].values[i]
        noms.append(libelle(variable_1) + ' / ' + libelle(variable_2))
        # Noms d'origine des deux variables remplacés par leurs noms français dans le texte de la décision
        decisions.append(table['Décision'].values[i].replace(variable_1, libelle(variable_1)).replace(
            variable_2, libelle(variable_2)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table['|r|'], y=noms, orientation='h', marker=marqueur_degrade(table['|r|'], ECHELLE_DEGRADE),
                         text=decisions, textposition='inside', insidetextanchor='start',
                         hovertemplate='%{y}<br>|r| = %{x:,.4f}<br>%{text}<extra></extra>'))
    fig.add_shape(type='line', x0=0.8, x1=0.8, y0=0, y1=1, yref='paper',
                  line={'color': COULEUR_REFERENCE, 'dash': 'dash', 'width': 2})
    fig.add_annotation(x=0.8, y=1, yref='paper', text='Seuil de redondance : |r| = 0,8', showarrow=False,
                       xanchor='right', yanchor='bottom', font={'color': COULEUR_REFERENCE})
    fig = mise_en_forme(fig, "Paires de variables redondantes et décisions",
                        "Corrélation de Pearson entre les deux variables (valeur absolue |r|)", "Paire de variables")
    fig.update_xaxes(range=[0, 1])
    fig = regler_axe(fig, 'x', 0, 1)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Modélisation
# ---------------------------------------------------------------------------------------------------------------

def graphique_f1_par_pli(scores, horizon):
    # scores : colonnes Pli et un F1 par modèle (paramètres par défaut) ; horizon : 'J+1' ou 'J+2'
    fig = go.Figure()
    for nom in MODELES:
        fig.add_trace(go.Scatter(x=scores['Pli'], y=scores[nom], mode='lines+markers', name=nom,
                                 line={'color': COULEURS_MODELES[nom], 'width': 2},
                                 marker={'symbol': SYMBOLES_MODELES[nom], 'size': 10},
                                 hovertemplate=nom + '<br>Pli %{x}<br>F1 = %{y:,.4f}<extra></extra>'))
    fig = mise_en_forme(fig, "F1 de chaque pli, paramètres par défaut (" + horizon + ")",
                        "Pli de validation chronologique", "F1 (classe pluie, " + horizon + ")")
    fig.update_xaxes(tickvals=scores['Pli'])
    fig = regler_axe(fig, 'y', scores[MODELES].min().min(), scores[MODELES].max().max())
    return fig


def graphique_evolution(progression, modele, horizon):
    # progression : colonnes Modèle et une colonne par étape (ETAPES) ; modele : nom du modèle
    valeurs = progression[progression['Modèle'] == modele][ETAPES].values[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ETAPES, y=valeurs, mode='lines+markers+text', name=modele,
                             line={'color': COULEURS_MODELES[modele], 'width': 2},
                             marker={'symbol': SYMBOLES_MODELES[modele], 'size': 12},
                             texttemplate='%{y:,.3f}', textposition='top center',
                             hovertemplate='%{x}<br>F1 = %{y:,.4f}<extra></extra>'))
    fig.add_shape(type='line', x0=3.5, x1=3.5, y0=0, y1=1, yref='paper',
                  line={'color': COULEUR_NEUTRE, 'dash': 'dot', 'width': 1})
    fig.add_annotation(x=3.5, y=1, yref='paper', text='Validation (moyenne sur 5 plis) ', showarrow=False,
                       xanchor='right', yanchor='bottom')
    fig.add_annotation(x=3.5, y=1, yref='paper', text=' Ensemble de test', showarrow=False, xanchor='left',
                       yanchor='bottom')
    fig = mise_en_forme(fig, modele + " : F1 à chaque étape (" + horizon + ")", "Étape",
                        "F1 (classe pluie, " + horizon + ")")
    fig.update_yaxes(range=[min(valeurs) - 0.05, max(valeurs) + 0.05])
    fig = regler_axe(fig, 'y', min(valeurs) - 0.05, max(valeurs) + 0.05)
    return fig


def graphique_comparaison_modeles(resultats, metrique, horizon):
    # resultats : une ligne par modèle (et le modèle naïf), métriques sur le test ; metrique : nom de la colonne
    table = resultats[resultats[metrique].isnull() == False].sort_values(metrique, ascending=False)
    couleurs = table['Modèle'].map(COULEURS_MODELES).fillna(COULEUR_NAIF)
    decimales = 3
    if metrique in ['Temps entraînement (s)', 'Temps prédiction (s)', 'Taille du fichier (Mo)']:
        decimales = 2
    # Valeur écrite au-dessus de chaque barre ; un fichier de moins de 0,1 Mo est écrit « < 0,1 »
    textes = []
    for valeur in table[metrique]:
        textes.append(nombre_fr(valeur, decimales))
        if metrique == 'Taille du fichier (Mo)' and valeur < 0.1:
            textes[len(textes) - 1] = '< 0,1'
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table['Modèle'], y=table[metrique], marker_color=couleurs, text=textes,
                         textposition='outside', hovertemplate='%{x}<br>' + metrique + ' : %{text}<extra></extra>'))
    fig = mise_en_forme(fig, metrique + " de chaque modèle sur l'ensemble de test (" + horizon + ")", "Modèle",
                        metrique + " (test, " + horizon + ")")
    fig.update_yaxes(range=[0, table[metrique].max() * 1.15])
    fig = regler_axe(fig, 'y', 0, table[metrique].max() * 1.15)
    return fig


def scores_seuil(cible, proba, seuil):
    # Matrice de confusion, précision, rappel et F1 de la classe pluie au seuil donné
    # (aucune prédiction positive : précision fixée à 0, même règle que 03_modelisation)
    pred = (proba >= seuil).astype(int)
    precision = 0.0
    if pred.sum() > 0:
        precision = precision_score(cible, pred)
    return confusion_matrix(cible, pred), precision, recall_score(cible, pred), f1_score(cible, pred)


def texte_case(nombre, total, classe):
    # Nombre de jours d'une case et sa part dans la ligne (classe réelle) : « 4 537 jours (71 % des jours de pluie) »
    return entier_fr(nombre) + ' jours<br>(' + entier_fr(nombre / total * 100) + ' % des jours ' + classe + ')'


def graphique_seuil(probas, modele, seuil, horizon):
    # probas : colonnes Cible et une probabilité de pluie par modèle (test) ; seuil : seuil de décision appliqué ;
    # chaque case : nombre de jours et part de la ligne (classe réelle), couleur selon cette part
    matrice, precision, rappel, f1 = scores_seuil(probas['Cible'], probas[modele], seuil)
    classes = ['Pas de pluie', 'Pluie']
    jours_secs = matrice[0].sum()
    jours_pluie = matrice[1].sum()
    parts = [[matrice[0, 0] / jours_secs * 100, matrice[0, 1] / jours_secs * 100],
             [matrice[1, 0] / jours_pluie * 100, matrice[1, 1] / jours_pluie * 100]]
    textes = [[texte_case(matrice[0, 0], jours_secs, 'secs'), texte_case(matrice[0, 1], jours_secs, 'secs')],
              [texte_case(matrice[1, 0], jours_pluie, 'de pluie'), texte_case(matrice[1, 1], jours_pluie, 'de pluie')]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=parts, x=classes, y=classes, zmin=0, zmax=100, colorscale=ECHELLE_MATRICE, text=textes,
                             texttemplate='%{text}', textfont={'size': 16, 'color': COULEUR_TEXTE},
                             colorbar={'title': {'text': 'Part des jours de la classe réelle (%)', 'side': 'right'},
                                       'tickformat': ',d'},
                             hovertemplate='Réel : %{y}<br>Prédit : %{x}<br>%{text}<extra></extra>'))
    texte = ("Seuil = " + nombre_fr(seuil, 3) + "   |   Précision = " + nombre_fr(precision, 3)
             + "   |   Rappel = " + nombre_fr(rappel, 3) + "   |   F1 = " + nombre_fr(f1, 3))
    fig.add_annotation(x=0.5, y=1.02, xref='paper', yref='paper', text=texte, showarrow=False, yanchor='bottom')
    fig = mise_en_forme(fig, modele + " : matrice de confusion sur le test (" + horizon + ")",
                        "Classe prédite", "Classe réelle")
    fig.update_yaxes(autorange='reversed')
    return fig


def barres_variables(table, colonne, couleur, titre_x):
    # Barres horizontales des 15 premières variables d'une table triée par ordre décroissant (la première en haut) ;
    # couleur : une couleur, ou 'dégradé' pour une couleur selon la valeur
    table = table.head(15).sort_values(colonne, ascending=True)
    noms = []
    for variable in table['Variable']:
        noms.append(libelle(variable))
    marqueur = {'color': couleur}
    if couleur == 'dégradé':
        marqueur = marqueur_degrade(table[colonne], ECHELLE_DEGRADE)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table[colonne], y=noms, orientation='h', marker=marqueur,
                         customdata=table['Variable'],
                         hovertemplate='%{y} (%{customdata})<br>' + colonne + ' : %{x:,.4f}<extra></extra>'))
    fig = mise_en_forme(fig, "", titre_x, "Variable")
    fig = regler_axe(fig, 'x', 0, table[colonne].max())
    fig.update_layout(height=650)
    return fig


def graphique_importance(importances, horizon):
    # importances : colonnes Variable, Importance (feature_importances_ du modèle XGBoost)
    table = importances.sort_values('Importance', ascending=False)
    fig = barres_variables(table, 'Importance', COULEURS_MODELES['XGBoost'],
                           "Importance relative (gain, somme des importances = 1)")
    fig.update_layout(title={'text': "XGBoost : importance des variables, 15 premières (" + horizon + ")"})
    return fig


def graphique_shap_synthese(valeurs_shap, horizon):
    # valeurs_shap : une ligne par jour du test, une colonne de valeurs SHAP par variable, et la valeur de base
    variables = []
    moyennes = []
    for col in valeurs_shap.columns:
        if col != 'Valeur de base':
            variables.append(col)
            moyennes.append(np.abs(valeurs_shap[col]).mean())
    table = pd.DataFrame({'Variable': variables, 'SHAP absolu moyen': moyennes})
    table = table.sort_values('SHAP absolu moyen', ascending=False)
    fig = barres_variables(table, 'SHAP absolu moyen', 'dégradé', "Valeur SHAP absolue moyenne sur le test (log-odds)")
    fig.update_layout(title={'text': "XGBoost : synthèse SHAP, 15 premières variables (" + horizon + ")"})
    return fig


def couleurs_normalisees(valeurs):
    # Valeurs ramenées entre 0 (faible) et 1 (élevée) entre leurs 5e et 95e centiles (au-delà : 0 ou 1)
    bas = valeurs.quantile(0.05)
    haut = valeurs.quantile(0.95)
    if haut == bas:
        haut = bas + 1
    normalisees = (valeurs - bas) / (haut - bas)
    normalisees = np.where(normalisees < 0, 0, normalisees)
    return np.where(normalisees > 1, 1, normalisees)


def ajouter_essaim(fig, shap_variable, valeurs_variable, position):
    # Points d'une variable : valeur SHAP en abscisse, ligne de la variable avec un léger décalage vertical aléatoire,
    # couleur selon la valeur de la variable
    decalage = np.random.uniform(-0.3, 0.3, len(shap_variable))
    fig.add_trace(go.Scatter(x=shap_variable, y=position + decalage, mode='markers', showlegend=False,
                             marker={'size': 5, 'opacity': 0.7, 'color': couleurs_normalisees(valeurs_variable),
                                     'colorscale': ECHELLE_VALEURS, 'cmin': 0, 'cmax': 1},
                             customdata=valeurs_variable,
                             hovertemplate=libelle(shap_variable.name) + '<br>Valeur SHAP : %{x:+,.3f}<br>'
                             + 'Valeur de la variable : %{customdata:,.2f}<extra></extra>'))
    return fig


def graphique_shap_beeswarm(valeurs_shap, valeurs_variables, horizon):
    # valeurs_shap : valeurs SHAP du test (une colonne par variable, et la valeur de base) ; valeurs_variables : valeurs
    # des variables préparées du test (non mises à l'échelle), mêmes lignes ; les 15 variables à la plus forte valeur
    # SHAP absolue moyenne, la première en haut, un point par jour (3 000 jours du test tirés au hasard)
    moyennes = []
    for col in valeurs_variables.columns:
        moyennes.append(np.abs(valeurs_shap[col]).mean())
    table = pd.DataFrame({'Variable': valeurs_variables.columns, 'SHAP absolu moyen': moyennes})
    table = table.sort_values('SHAP absolu moyen', ascending=False).head(15)
    jours = valeurs_shap.sample(n=3000, random_state=42).index
    np.random.seed(42)
    fig = go.Figure()
    noms = []
    for i in range(len(table)):
        variable = table['Variable'].values[i]
        fig = ajouter_essaim(fig, valeurs_shap.loc[jours, variable], valeurs_variables.loc[jours, variable], -i)
        noms.append(libelle(variable))
    fig.data[0].marker.showscale = True
    fig.data[0].marker.colorbar = {'title': {'text': 'Valeur de la variable : faible → élevée', 'side': 'right'},
                                   'tickvals': [0, 1], 'ticktext': ['Faible', 'Élevée']}
    fig.add_shape(type='line', x0=0, x1=0, y0=0, y1=1, yref='paper', line={'color': COULEUR_TEXTE, 'width': 1})
    fig = mise_en_forme(fig, "XGBoost : valeurs SHAP de chaque jour, 15 premières variables (" + horizon + ")",
                        "Valeur SHAP (log-odds) ; à droite : vers la pluie", "Variable")
    fig.update_yaxes(tickvals=list(range(0, -len(table), -1)), ticktext=noms, showgrid=False, zeroline=False)
    fig = regler_axe(fig, 'x', valeurs_shap.loc[jours, table['Variable']].min().min(),
                     valeurs_shap.loc[jours, table['Variable']].max().max())
    fig.update_layout(height=750)
    return fig


def graphique_odds_ratios(odds, horizon):
    # odds : colonnes Variable, Odds ratio (exp des coefficients de la régression logistique, variables mises à l'échelle)
    # Barres partant de 1 : dégradé bleu au-dessus de 1, dégradé or au-dessous, selon l'écart à 1
    table = odds.sort_values('Odds ratio', ascending=True)
    noms = []
    for variable in table['Variable']:
        noms.append(libelle(variable))
    table['Nom'] = noms
    au_dessus = table[table['Odds ratio'] > 1]
    au_dessous = table[table['Odds ratio'] <= 1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=au_dessus['Odds ratio'] - 1, base=1, y=au_dessus['Nom'], orientation='h',
                         marker=marqueur_degrade(au_dessus['Odds ratio'] - 1, ECHELLE_DEGRADE), showlegend=False,
                         customdata=au_dessus[['Variable', 'Odds ratio']],
                         hovertemplate='%{y} (%{customdata[0]})<br>Odds ratio : %{customdata[1]:,.4f}<extra></extra>'))
    fig.add_trace(go.Bar(x=au_dessous['Odds ratio'] - 1, base=1, y=au_dessous['Nom'], orientation='h',
                         marker=marqueur_degrade(1 - au_dessous['Odds ratio'], ECHELLE_NEGATIVE), showlegend=False,
                         customdata=au_dessous[['Variable', 'Odds ratio']],
                         hovertemplate='%{y} (%{customdata[0]})<br>Odds ratio : %{customdata[1]:,.4f}<extra></extra>'))
    # Ordre des variables : celui du tri, quelle que soit la série
    fig.update_yaxes(categoryorder='array', categoryarray=noms)
    fig.add_shape(type='line', x0=1, x1=1, y0=0, y1=1, yref='paper', line={'color': COULEUR_TEXTE, 'width': 1})
    fig = mise_en_forme(fig, "Régression logistique : odds ratios (" + horizon + ")",
                        "Odds ratio (effet d'une hausse d'une unité de la variable mise à l'échelle)", "Variable")
    fig = regler_axe(fig, 'x', min(1, table['Odds ratio'].min()), max(1, table['Odds ratio'].max()))
    fig.update_layout(height=750)
    return fig


def graphique_stations(stations, modele, horizon):
    # stations : une ligne par modèle et par station (test) ; modele : nom du modèle affiché
    table = stations[stations['Modèle'] == modele].sort_values('F1', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table['F1'], y=table['Station'], orientation='h',
                         marker=marqueur_degrade(table['F1'], ECHELLE_DEGRADE),
                         customdata=table[['Précision', 'Rappel', 'Jours de pluie', 'Jours de test']],
                         hovertemplate='%{y}<br>F1 = %{x:,.3f}<br>Précision = %{customdata[0]:,.3f}<br>'
                         + 'Rappel = %{customdata[1]:,.3f}<br>%{customdata[2]} jours de pluie sur %{customdata[3]} '
                         + 'jours de test<extra></extra>'))
    fig = mise_en_forme(fig, modele + " : F1 par station sur le test (" + horizon + ")",
                        "F1 de la station (classe pluie, test, " + horizon + ")", "Station")
    fig.update_xaxes(range=[0, 1])
    fig = regler_axe(fig, 'x', 0, 1)
    fig.update_layout(height=1100)
    return fig


# ---------------------------------------------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------------------------------------------

def graphique_jauge(probabilite, seuil):
    # probabilite et seuil entre 0 et 1 ; arc de la couleur de la décision : accent si la probabilité atteint le seuil
    # (pluie prévue), ambre sinon ; jauge compacte, sans titre ni nombre (la décision, la probabilité et le seuil sont
    # écrits au-dessus)
    couleur = COULEUR_PAS_DE_PLUIE
    if probabilite >= seuil:
        couleur = COULEUR_PLUIE
    fig = go.Figure()
    fig.add_trace(go.Indicator(mode='gauge', value=probabilite * 100,
                               gauge={'axis': {'range': [0, 100], 'ticksuffix': ' %',
                                               'tickfont': {'color': COULEUR_TEXTE_SECONDAIRE}},
                                      'bar': {'color': couleur}, 'bgcolor': COULEUR_FOND_CARTE,
                                      'bordercolor': COULEUR_GRILLE,
                                      'threshold': {'line': {'color': COULEUR_REFERENCE, 'width': 4},
                                                    'thickness': 0.9, 'value': seuil * 100}}))
    fig = appliquer_theme(fig, "")
    # Marges latérales : graduations « 0 % » et « 100 % » entièrement visibles dans une demi-largeur
    fig.update_layout(height=220, margin={'l': 50, 'r': 50, 't': 30, 'b': 10})
    return fig


def graphique_contribution(contributions, valeur_base, nombre):
    # contributions : colonnes Variable, Valeur, Contribution (valeurs SHAP d'une prédiction, en log-odds) ;
    # les `nombre` plus fortes contributions sont détaillées, les autres regroupées
    table = contributions.copy()
    table['Contribution absolue'] = np.abs(table['Contribution'])
    table = table.sort_values('Contribution absolue', ascending=False)
    detail = table.head(nombre)
    noms = ['Valeur de base']
    textes = ['Moyenne du modèle']
    mesures = ['absolute']
    valeurs = [valeur_base]
    for i in range(len(detail)):
        noms.append(libelle(detail['Variable'].values[i]))
        textes.append('Valeur : ' + nombre_fr(detail['Valeur'].values[i], 2))
        mesures.append('relative')
        valeurs.append(detail['Contribution'].values[i])
    total = valeur_base + table['Contribution'].sum()
    noms = noms + ['Autres variables (' + str(len(table) - nombre) + ')', 'Prédiction']
    probabilite = nombre_fr(100 / (1 + np.exp(-total)), 1) + ' %'
    textes = textes + ['', 'Probabilité : ' + probabilite]
    mesures = mesures + ['relative', 'total']
    valeurs = valeurs + [table['Contribution'].values[nombre:].sum(), 0]
    fig = go.Figure()
    fig.add_trace(go.Waterfall(orientation='h', y=noms, x=valeurs, measure=mesures, customdata=textes,
                               increasing={'marker': {'color': COULEUR_PLUIE}},
                               decreasing={'marker': {'color': COULEUR_PAS_DE_PLUIE}},
                               totals={'marker': {'color': COULEUR_NEUTRE}}, connector={'line': {'color': COULEUR_GRILLE}},
                               hovertemplate='%{y}<br>%{customdata}<br>Effet : %{x:+.3f}<extra></extra>'))
    fig = mise_en_forme(fig, "Contribution des variables à la prédiction (XGBoost, valeurs SHAP)",
                        "Marge du modèle (log-odds) ; vers la droite : vers la pluie", "Variable")
    position = 0
    positions = [0]
    for valeur in valeurs[:len(valeurs) - 1]:
        position = position + valeur
        positions.append(position)
    # Probabilité finale écrite à droite de la barre « Prédiction » (qui va de 0 à la marge totale), dans la zone de
    # tracé : l'axe est prolongé à droite pour laisser la place au texte
    fin_barre = max(0, total)
    etendue = max(positions) - min(positions)
    debut_axe = min(positions) - 0.05 * etendue
    fin_axe = max(max(positions), fin_barre) + 0.6 * etendue
    fig.add_annotation(x=fin_barre, y='Prédiction', text='  Probabilité de pluie : ' + probabilite, showarrow=False,
                       xanchor='left', font={'color': COULEUR_TEXTE})
    fig.update_xaxes(range=[debut_axe, fin_axe])
    fig = regler_axe(fig, 'x', debut_axe, fin_axe)
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=650, showlegend=False)
    return fig


def calculer_sensibilite(observation, demo, variable, plages):
    # Probabilité de pluie quand `variable` parcourt sa plage observée (50 valeurs), les autres valeurs fixées ;
    # observation : une ligne brute ; demo : {'modele', 'artefacts', 'mise_a_echelle', 'colonnes'} ;
    # plages : colonnes Variable, Minimum, Maximum
    plage = plages[plages['Variable'] == variable]
    grille = np.linspace(plage['Minimum'].values[0], plage['Maximum'].values[0], 50)
    lignes = []
    for valeur in grille:
        ligne = observation.copy()
        ligne[variable] = valeur
        lignes.append(ligne)
    X = preparer_brut(pd.concat(lignes), demo['artefacts'], demo['mise_a_echelle'], demo['colonnes'])
    probas = demo['modele'].predict_proba(X)[:, 1]
    return pd.DataFrame({'Valeur': grille, 'Probabilité de pluie': probas})


def graphique_sensibilite(courbe, variable, seuil, valeur_observee):
    # courbe : colonnes Valeur, Probabilité de pluie (calculer_sensibilite) ; seuil et valeur observée marqués
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=courbe['Valeur'], y=courbe['Probabilité de pluie'] * 100, mode='lines',
                             line={'color': COULEURS_MODELES['XGBoost'], 'width': 3}, name='Probabilité de pluie',
                             hovertemplate=libelle(variable) + ' = %{x:,.1f}<br>Probabilité de pluie : %{y:,.1f} %'
                             + '<extra></extra>'))
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=seuil * 100, y1=seuil * 100,
                  line={'color': COULEUR_REFERENCE, 'dash': 'dash', 'width': 2})
    fig.add_annotation(x=0, xref='paper', y=seuil * 100, text='Seuil de décision : ' + nombre_fr(seuil * 100, 1) + ' %',
                       showarrow=False, xanchor='left', yanchor='bottom', font={'color': COULEUR_REFERENCE})
    fig.add_shape(type='line', x0=valeur_observee, x1=valeur_observee, y0=0, y1=1, yref='paper',
                  line={'color': COULEUR_NEUTRE, 'dash': 'dot', 'width': 2})
    fig.add_annotation(x=valeur_observee, y=1, yref='paper', text='Valeur observée : ' + nombre_fr(valeur_observee, 1),
                       showarrow=False, xanchor='left', yanchor='bottom')
    fig = mise_en_forme(fig, "Sensibilité de la prédiction à : " + libelle(variable), libelle_axe(variable),
                        "Probabilité de pluie (%)")
    fig.update_yaxes(range=[0, 100])
    fig = regler_axe(fig, 'y', 0, 100)
    fig = regler_axe(fig, 'x', courbe['Valeur'].min(), courbe['Valeur'].max())
    return fig
