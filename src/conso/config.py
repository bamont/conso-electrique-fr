"""Constantes partagées par l'entraînement, le backtest et l'API."""

from __future__ import annotations

TZ = "Europe/Paris"
ISSUE_HOUR = 10          # prévision émise le jour D à 10 h (heure locale) pour le jour D+1
GAP_DAYS = 2             # écart entre fin d'entraînement et premier jour prédit (backtest)
HISTORY_DAYS = 45        # historique nécessaire pour calculer les variables d'un jour
FC_START = "2022-01-01"  # début de la couverture des prévisions météo historiques

# Météo : variables Open-Meteo -> colonnes du jeu de données
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"]
WCOLS = ["temp_nat", "hum_nat", "wind_nat", "cloud_nat", "rad_nat"]

# nom: (latitude, longitude, poids ~ population de l'agglomération, en millions)
CITIES = {
    "Paris": (48.8566, 2.3522, 10.9),
    "Lyon": (45.7640, 4.8357, 2.3),
    "Marseille": (43.2965, 5.3698, 1.9),
    "Toulouse": (43.6047, 1.4442, 1.4),
    "Lille": (50.6292, 3.0573, 1.2),
    "Bordeaux": (44.8378, -0.5792, 1.3),
    "Nantes": (47.2184, -1.5536, 1.0),
    "Strasbourg": (48.5734, 7.7521, 0.8),
    "Nice": (43.7102, 7.2620, 1.0),
    "Rennes": (48.1173, -1.6778, 0.75),
}
CITY_WEIGHTS = {name: w for name, (_, _, w) in CITIES.items()}

# Modèle
DEFAULT_PARAMS = {
    "objective": "regression", "metric": "l1", "learning_rate": 0.1, "num_leaves": 63,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "seed": 0,
}
DEFAULT_ROUNDS = 1000
QUANTILE_ROUNDS = 400

# Intervalle conforme : groupes de température (température lissée, en °C)
TEMP_GROUP_EDGES = (5.0, 15.0, 22.0)
TEMP_GROUP_NAMES = ("froid", "doux", "chaud", "tres_chaud")

# Sources de données
ODRE_EXPORT = "https://opendata.reseaux-energies.fr/api/explore/v2.1/catalog/datasets/{}/exports/csv"
ODRE_PARAMS = {"timezone": "UTC", "use_labels": "false", "delimiter": ";"}
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
