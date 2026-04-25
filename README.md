# CycloPredict - Prédiction des Flux Cyclistes Bordeaux-Campus
## Présentation
Modélisation et prédiction des flux cyclistes sur l'axe Bordeaux (Barrière de Toulouse) — Campus (Pessac Haut-Lévêque). Ce MVP utilise un réseau de neurones pour estimer la "cyclabilité" de mobilité universitaire.

## Architecture des Scripts
    01_get_data.py : Extraction des données (Open Data Bordeaux Métropole).

    02_preprocess.py : Nettoyage et Feature Engineering (ex : variables cycliques, lags temporels).

    04_train_mlp.py : Entraînement du modèle MLPRegressor et normalisation.

    05_plot_results.py : Visualisation des performances sur le set de test.

## stack technique 
* Langage : Python 3.12+

* Machine Learning : Scikit-learn (MLPRegressor), Pandas.

* Visualisation : Matplotlib.

* Persistance : Joblib.

## Installation et usage
    pip install -r requirements.txt
    # Exécution du pipeline
    python scripts/01_get_data.py
    python scripts/02_preprocess.py
    python scripts/04_train_mlp.py
    python scripts/05_plot_results.py
