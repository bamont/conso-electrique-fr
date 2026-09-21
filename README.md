# Prévision de la consommation électrique française (J+1, pas de 30 min)

Projet de data science autour des séries temporelles : prévoir, le jour D à 10 h, la consommation nationale d'électricité des 48 demi-heures du jour D+1, avec un intervalle de prévision à 90 %, et la comparer à la prévision publiée par RTE.

**Ce que le projet démontre :** protocole d'évaluation sans fuite de données (testé), baselines rigoureuses, gestion des changements de niveau, modèle LightGBM, prévision avec **météo prévue** (et non observée), correction du biais de la météo, intervalles calibrés (prédiction conforme), code testé, API et Docker.

## Résultats

Période de test : **janvier 2025 → juin 2026** (26 202 demi-heures communes à tous les modèles, données consolidées), réentraînement mensuel, émission à 10 h le jour D.

| Modèle | MAE (MW) | MAPE | Biais (MW) |
|---|---|---|---|
| Naïf saisonnier (J-7) | 3 666 | 6,83 % | +154 |
| Régression linéaire par créneau (météo observée) | 1 505 | 3,00 % | −8 |
| **RTE J-1** (référence externe) | **1 379** | **2,74 %** | −1 055 |
| LightGBM, météo observée (borne haute) | 870 | 1,66 % | −124 |
| LightGBM, météo prévue brute | 1 053 | 2,00 % | −537 |
| LightGBM, entraîné sur prévisions météo | 971 | 1,83 % | −178 |
| **LightGBM, météo prévue + température débiaisée par heure** ¹ | **929** | **1,78 %** | −262 |

¹ Modèle entraîné une seule fois avant le test. Le notebook `04_validation_production` refait cette mesure avec le code de production (réentraînement mensuel, biais réestimé chaque mois) : **mettre à jour cette ligne avec ses chiffres.**

**Lecture honnête de ces chiffres**
- Avec la météo prévue débiaisée, la MAE est **33 % sous celle de RTE J-1**, et l'écart-type de l'erreur 9 % plus bas (1 236 contre 1 356 MW).
- RTE a un **biais systématique de −1 GW** (dont on ignore la cause : définition de la série ou dérive de la prévision). Une part importante de l'avantage en MAE vient de ce biais : en écart-type de l'erreur, l'avance est modeste, et nulle sans les informations de la matinée de D.
- **Sans la consommation de la matinée de D** (émission à la fin de D−1), la MAE en météo prévue passe de 1 064 à 1 147 MW : le résultat tient, mais RTE reprend l'avantage sur les créneaux de 0 h à 9 h 30.
- **Faiblesses :** froid intense (< 0 °C, 1 644 MW contre 1 505 MW pour RTE), jours fériés (peu d'exemples), fortes chaleurs pour l'intervalle.
- **Intervalle à 90 %** : la couverture passe de 66 % à ~88 % après calibration conforme (notebook 03) ; le notebook 04 mesure la version de production (4 groupes de température).

## Données

| Source | Contenu |
|---|---|
| RTE éCO2mix (`eco2mix-national-cons-def`, `-tr`) | consommation nationale 2013-2026 (définitive, consolidée, temps réel), prévisions RTE J-1 et J |
| Open-Meteo (archive) | météo observée (ERA5) de 10 agglomérations, agrégée en moyenne nationale pondérée |
| Open-Meteo (prévisions historiques) | météo prévue, depuis 2022 |
| `holidays`, `vacances-scolaires-france`, Éducation nationale | jours fériés, ponts, vacances scolaires par zone |

## Méthode

1. **Cible : l'écart au niveau récent.** Le niveau de consommation a baissé d'environ 5 GW en dix ans (efficacité, autoconsommation solaire, sobriété). On prédit `consommation − level7`, où `level7` est la moyenne des 7 jours complets T−8 … T−2 : la dérive est absorbée par construction.
2. **Variables (39)** : calendrier, météo à l'instant t et lissée, retards T−1 (créneaux avant 10 h), T−2, T−7, T−14, écart matinal de D, chauffage des 7 jours du niveau.
3. **Météo.** Le modèle est entraîné sur la météo **observée** (historique complet depuis 2013) et servi avec la météo **prévue**, dont la température est corrigée d'un **biais horaire** (~+1 °C la nuit et le soir, ~0 vers 10 h), réestimé sur les 24 derniers mois.
4. **Intervalle.** Régression quantile (5 % / 95 %) puis calibration conforme (CQR) par régime de température (froid, doux, chaud, très chaud).
5. **Protocole sans fuite.** Un test automatique efface toute la consommation postérieure à l'heure d'émission et vérifie que les variables du jour cible ne changent pas.

## Structure du dépôt

```
notebooks/            01 exploration · 02 baselines · 03 LightGBM · 04 validation de la production
src/conso/            package : features, météo, modèle, backtest, entraînement, API
tests/                33 tests (fuite de données, DST, calendrier, météo, modèle, API)
data/  models/        régénérés localement (ignorés par git)
Dockerfile · Makefile · .github/workflows/ci.yml
```

## Démarrage rapide

```powershell
# 1. Environnement
poetry install --with dev

# 2. Notebooks, dans l'ordre : 01 → 02 → 03 → 04
poetry run jupyter lab

# 3. Tests et qualité de code
poetry run pytest -q
poetry run ruff check src tests

# 4. Entraîner le modèle de production (le notebook 04 le fait aussi)
poetry run python -m conso.train --dataset data/processed/dataset_30min.parquet `
    --meteo-fc data/raw/meteo_forecast_openmeteo.parquet --out models/prod

# 5. Lancer l'API
$env:CONSO_MODEL="models/prod"
$env:CONSO_DATASET="data/processed/dataset_30min.parquet"
$env:CONSO_METEO_FC="data/raw/meteo_forecast_openmeteo.parquet"
poetry run uvicorn conso.api.main:app --reload
```

Documentation interactive : <http://localhost:8000/docs>.

### API

| Route | Rôle |
|---|---|
| `GET /health` | état du service |
| `GET /model` | version, période d'entraînement, biais horaire, élargissements de l'intervalle |
| `GET /forecast?date=2025-01-15&mode=replay` | rejoue la prévision d'un jour passé (hors ligne), avec le réel et la prévision RTE J-1 pour comparer |
| `GET /forecast?date=<demain>&mode=live` | prévision du lendemain à partir des données RTE et Open-Meteo en direct (nécessite Internet) |

```json
{
  "date": "2025-01-15", "mode": "replay", "issue_time": "2025-01-14T10:00:00+01:00",
  "model_version": "20260920-2125", "n_points": 48,
  "points": [{"time": "2025-01-15T00:00:00+01:00", "forecast_mw": 69041.2, "lower_mw": 67814.9,
              "upper_mw": 72955.6, "actual_mw": 69635.3, "rte_j1_mw": 67670.0}, "..."]
}
```

### Docker

```powershell
docker build -t conso-api .        # nécessite models/prod (produit par le notebook 04 ou `conso.train`)
docker run --rm -p 8000:8000 conso-api                       # mode live
docker run --rm -p 8000:8000 -v ${PWD}/data:/app/data `
    -e CONSO_DATASET=/app/data/processed/dataset_30min.parquet `
    -e CONSO_METEO_FC=/app/data/raw/meteo_forecast_openmeteo.parquet conso-api   # + mode replay
```

## Limites et pistes

- **Heure d'émission de RTE inconnue.** Les comparaisons supposent une émission à 10 h avec la matinée de D connue ; la variante sans ces informations est fournie.
- **Coût de la météo prévue sous-estimé.** Les prévisions historiques d'Open-Meteo ont des échéances courtes ; une vraie prévision J-1 est un peu moins précise. Le biais horaire est estimé sur la même source qu'en exploitation, et devra être réestimé si la source change.
- **Calendrier scolaire.** L'entraînement utilise les données de l'Éducation nationale depuis octobre 2017 (`vacances-scolaires-france` avant) ; l'API utilise ce package pour tous les jours. Vérifier leur accord sur la période commune.
- **Mode `live` non testé sur le réseau réel** dans la suite de tests : les appels RTE et Open-Meteo y sont simulés.
- **Pistes :** températures régionales et plusieurs modèles météo, correction de biais des autres variables (nébulosité, rayonnement, vent), recalibration glissante de l'intervalle, ré-entraînement mensuel automatisé, comparaison avec un modèle fondation (Chronos, TimesFM).
