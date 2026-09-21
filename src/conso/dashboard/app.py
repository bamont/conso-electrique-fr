"""Application Streamlit : prévision du lendemain, rejeu d'un jour passé, performances, modèle.

    streamlit run src/conso/dashboard/app.py
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from conso.config import ISSUE_HOUR, TZ
from conso.dashboard import charts, client, performance

st.set_page_config(page_title="Consommation électrique : prévision J+1", page_icon="⚡", layout="wide")

DATASET = os.environ.get("CONSO_DATASET", "data/processed/dataset_30min.parquet")
BACKTEST = os.environ.get("CONSO_BACKTEST", "data/processed/production_backtest_predictions.parquet")
PAGES = ["Demain (live)", "Rejouer un jour", "Performance", "Modèle"]
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


# Accès aux données (mises en cache)
@st.cache_data(ttl=600, show_spinner="Calcul de la prévision…")
def cached_forecast(date: str, mode: str) -> dict:
    return client.get_json("/forecast", {"date": date, "mode": mode})


@st.cache_data(ttl=300, show_spinner=False)
def cached_model() -> dict:
    return client.get_json("/model", timeout=10)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_range() -> dict:
    return client.get_json("/replay/range", timeout=30)


@st.cache_data(show_spinner="Chargement des performances…")
def cached_performance(dataset: str, backtest: str, mtimes: tuple[float, float]):
    df, bt = performance.load(dataset, backtest)
    F = performance.eval_frame(df, bt)
    return (F, performance.summary(F, performance.mase_denominator(df)), performance.mae_by_month(F),
            performance.mae_by_temperature(F), performance.mae_by_slot(F), performance.rte_by_year(df))


def show_error(e: client.ApiError) -> None:
    if e.status == 404:
        st.warning(f"Pas de prévision disponible : {e}")
    elif e.status in (None, 503):
        st.error(f"Service indisponible : {e}")
    else:
        st.error(f"Erreur de l'API ({e.status}) : {e}")


def date_fr(d: dt.date) -> str:
    return f"{JOURS[d.weekday()]} {d:%d/%m/%Y}"


def gw(x: float) -> str:
    return f"{x / 1000:.1f} GW"


def forecast_details(df: pd.DataFrame, name: str) -> None:
    with st.expander("Voir et télécharger les données"):
        table = df.copy()
        table.index = table.index.strftime("%Y-%m-%d %H:%M")
        st.dataframe(table)
        st.download_button("Télécharger en CSV", table.to_csv().encode("utf-8"), file_name=f"{name}.csv", mime="text/csv")


# Pages
def page_live() -> None:
    st.header("Prévision du lendemain")
    now = client.now_paris()
    target = (now.tz_localize(None).normalize() + pd.Timedelta(days=1)).date()
    st.caption(f"Jour prévu : **{date_fr(target)}**. La prévision est émise le jour D à {ISSUE_HOUR} h pour D+1, "
               "à partir de la consommation RTE en temps réel et des prévisions météo Open-Meteo.")
    if now.hour < ISSUE_HOUR:
        st.warning(f"Il est avant {ISSUE_HOUR} h : la consommation de la matinée n'est pas encore connue, "
                   "la prévision est moins précise. Elle sera meilleure après 10 h.")
    if st.button("🔄 Actualiser"):
        cached_forecast.clear()
    try:
        payload = cached_forecast(str(target), "live")
    except client.ApiError as e:
        show_error(e)
        return

    df = client.forecast_frame(payload)
    peak = df["forecast_mw"].idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pointe prévue", gw(df["forecast_mw"].max()), f"à {peak:%H:%M}", delta_color="off")
    c2.metric("Creux prévu", gw(df["forecast_mw"].min()), f"à {df['forecast_mw'].idxmin():%H:%M}", delta_color="off")
    c3.metric("Moyenne prévue", gw(df["forecast_mw"].mean()))
    c4.metric("Largeur moyenne de l'intervalle", gw((df["upper_mw"] - df["lower_mw"]).mean()))
    st.plotly_chart(charts.forecast_figure(df, f"Consommation prévue le {target:%d/%m/%Y}"))
    st.caption(f"Modèle {payload['model_version']} · émission : {pd.Timestamp(payload['issue_time']):%d/%m/%Y %H:%M}")
    forecast_details(df, f"prevision_{target}")


def page_replay() -> None:
    st.header("Rejouer un jour passé")
    st.caption("Reconstitue la prévision telle qu'elle aurait été faite la veille à 10 h, et la compare au réel "
               "et à la prévision RTE J-1. Météo prévue : historique Open-Meteo.")
    try:
        rng = cached_range()
    except client.ApiError as e:
        if e.status == 404 and str(e) == "Not Found":          # route inconnue : image de l'API ancienne
            st.error("L'API ne connaît pas la route `/replay/range` : son image est ancienne. "
                     "Reconstruis-la avec `docker compose up --build -d`.")
        elif e.status == 503:
            st.warning(f"Le rejeu n'est pas disponible : {e} Il faut monter le dossier `data/` dans le conteneur "
                       "de l'API et définir `CONSO_DATASET` et `CONSO_METEO_FC`.")
        else:
            show_error(e)
        return
    lo, hi = dt.date.fromisoformat(rng["min_date"]), dt.date.fromisoformat(rng["max_date"])
    default = min(max(dt.date(2025, 1, 15), lo), hi)
    day = st.date_input("Jour à rejouer", value=default, min_value=lo, max_value=hi)
    try:
        payload = cached_forecast(str(day), "replay")
    except client.ApiError as e:
        show_error(e)
        return

    df = client.forecast_frame(payload)
    err_model = (df["forecast_mw"] - df["actual_mw"]).abs().mean()
    err_rte = (df["rte_j1_mw"] - df["actual_mw"]).abs().mean()
    inside = ((df["actual_mw"] >= df["lower_mw"]) & (df["actual_mw"] <= df["upper_mw"])).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Erreur moyenne du modèle", f"{err_model:,.0f} MW".replace(",", " "))
    c2.metric("Erreur moyenne de RTE J-1", f"{err_rte:,.0f} MW".replace(",", " "), f"{err_model - err_rte:+,.0f} MW pour le modèle".replace(",", " "),
              delta_color="inverse")
    c3.metric("Pointe réelle / prévue", f"{gw(df['actual_mw'].max())} / {gw(df['forecast_mw'].max())}")
    c4.metric("Réel dans l'intervalle à 90 %", f"{inside:.0%}")
    st.plotly_chart(charts.forecast_figure(df, f"{date_fr(day)} : prévision, réel et RTE J-1"))
    forecast_details(df, f"rejeu_{day}")


def page_performance() -> None:
    st.header("Performances sur la période de test")
    ds, bt = Path(DATASET), Path(BACKTEST)
    if not (ds.exists() and bt.exists()):
        st.info("Fichiers introuvables. Cette page utilise le jeu de données et les prédictions du backtest "
                "(notebook 04) : `data/processed/dataset_30min.parquet` et "
                "`data/processed/production_backtest_predictions.parquet`. Voir `CONSO_DATASET` et `CONSO_BACKTEST`.")
        return
    F, summ, by_month, by_temp, by_slot, rte_year = cached_performance(str(ds), str(bt), (ds.stat().st_mtime, bt.stat().st_mtime))

    loc = F.index.tz_convert(TZ)
    st.caption(f"Test du {loc.min():%d/%m/%Y} au {loc.max():%d/%m/%Y} : {len(F):,} demi-heures communes à tous les modèles. "
               "Modèle réentraîné chaque mois ; biais de la météo prévue réestimé chaque mois.".replace(",", " "))
    ref, best = summ.loc[performance.LABELS["rte_j1"]], summ.loc[performance.LABELS["prod_fc_debiased"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE du modèle (météo prévue débiaisée)", f"{best['MAE (MW)']:,.0f} MW".replace(",", " "),
              f"{best['MAE (MW)'] / ref['MAE (MW)'] - 1:+.0%} vs RTE J-1", delta_color="inverse")
    c2.metric("Écart-type de l'erreur", f"{best['Écart-type (MW)']:,.0f} MW".replace(",", " "),
              f"{best['Écart-type (MW)'] / ref['Écart-type (MW)'] - 1:+.0%} vs RTE J-1", delta_color="inverse")
    c3.metric("MAPE", f"{best['MAPE (%)']:.2f} %", f"RTE J-1 : {ref['MAPE (%)']:.2f} %", delta_color="off")

    fmt = {"MAE (MW)": "{:,.0f}", "MAPE (%)": "{:.2f}", "Biais (MW)": "{:+,.0f}", "Écart-type (MW)": "{:,.0f}", "MASE": "{:.3f}"}
    st.dataframe(summ.style.format({k: v for k, v in fmt.items() if k in summ.columns}))

    st.plotly_chart(charts.lines_figure(by_month, "Erreur moyenne (MAE) par mois", "MW", performance.COLORS))
    left, right = st.columns(2)
    left.plotly_chart(charts.bars_figure(by_temp, "MAE selon la température nationale (°C)", "MW", performance.COLORS))
    right.plotly_chart(charts.bars_figure(by_slot, "MAE selon le créneau cible", "MW", performance.COLORS))
    st.plotly_chart(charts.lines_figure(rte_year[["MAPE RTE J-1 (%)"]], "Prévision RTE J-1 : MAPE par année", "%", {"MAPE RTE J-1 (%)": charts.RED}))

    st.markdown(
        "**Comment lire ces résultats**\n"
        "- RTE J-1 a un **biais de −1 GW**, dont la cause est inconnue : une part de l'avantage en MAE vient de là. "
        "L'écart-type de l'erreur ne dépend pas de ce biais.\n"
        "- **L'avantage se concentre hors de l'hiver** : sous 0 °C, le modèle est à parité avec RTE.\n"
        "- Les créneaux de 0 h à 9 h 30 supposent la consommation de la matinée de D connue (émission à 10 h).\n"
        "- La météo prévue vient d'un historique de prévisions à courte échéance : le coût d'une vraie prévision J-1 est probablement un peu supérieur."
    )


def page_model() -> None:
    st.header("Modèle en production")
    try:
        health, model = client.get_json("/health", timeout=10), cached_model()
    except client.ApiError as e:
        show_error(e)
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Version", str(model.get("version", "?")))
    c2.metric("Entraîné jusqu'au", str(model.get("train_end", "?"))[:10])
    c3.metric("Variables", model.get("n_features", "?"))
    st.caption(f"Cible : {model.get('target', '?')} · Météo : {model.get('weather', '?')} · "
               f"Rejeu disponible : {'oui' if health.get('replay_available') else 'non'}")
    left, right = st.columns(2)
    if model.get("bias_hourly_c"):
        left.plotly_chart(charts.hourly_bias_figure({int(h): v for h, v in model["bias_hourly_c"].items()}))
    if model.get("interval_widening_mw"):
        w = pd.DataFrame({"Élargissement (MW)": model["interval_widening_mw"]})
        right.plotly_chart(charts.bars_figure(w, "Élargissement conforme de l'intervalle, par régime de température", "MW"))
    with st.expander("Métadonnées complètes"):
        st.json(model)


# Navigation
def sidebar() -> str:
    st.sidebar.title("⚡ Prévision J+1")
    page = st.sidebar.radio("Navigation", PAGES)
    try:
        h = client.get_json("/health", timeout=5)
        (st.sidebar.success if h.get("model_loaded") else st.sidebar.warning)(
            "API connectée · modèle chargé" if h.get("model_loaded") else "API connectée · modèle non chargé")
    except client.ApiError as e:
        st.sidebar.error(str(e))
    st.sidebar.caption(f"API : {client.api_url()}")
    return page


def main() -> None:
    page = sidebar()
    {"Demain (live)": page_live, "Rejouer un jour": page_replay, "Performance": page_performance, "Modèle": page_model}[page]()


main()
