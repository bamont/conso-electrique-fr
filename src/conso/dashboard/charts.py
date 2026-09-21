"""Graphiques Plotly (fonctions pures : un DataFrame en entrée, une figure en sortie)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BLUE, RED, GREY = "#1f77b4", "#d62728", "#7f7f7f"

def _layout(title: str, height: int, ylabel: str = "") -> dict:
    """Titre en haut à gauche, légende juste au-dessus du graphique, marge haute généreuse."""
    return {
        "title": {"text": title, "x": 0, "xanchor": "left", "y": 0.96, "yanchor": "top"},
        "yaxis_title": ylabel, "height": height,
        "legend": {"orientation": "h", "x": 0, "yanchor": "bottom", "y": 1.02},
        "margin": {"l": 10, "r": 10, "t": 110, "b": 10},
    }


def forecast_figure(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Prévision et intervalle ; ajoute le réel et la prévision RTE J-1 quand ils existent."""
    x = df.index.tz_localize(None) # heure locale
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["upper_mw"] / 1000, line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=df["lower_mw"] / 1000, fill="tonexty", fillcolor="rgba(31,119,180,0.20)",
                             line={"width": 0}, name="Intervalle à 90 %"))
    fig.add_trace(go.Scatter(x=x, y=df["forecast_mw"] / 1000, name="Prévision", line={"color": BLUE, "width": 3}))
    if "rte_j1_mw" in df and df["rte_j1_mw"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=df["rte_j1_mw"] / 1000, name="RTE J-1", line={"color": RED, "width": 1.5}))
    if "actual_mw" in df and df["actual_mw"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=df["actual_mw"] / 1000, name="Réel", line={"color": "black", "width": 2}))
    fig.update_layout(**_layout(title, 460, "GW"))
    fig.update_yaxes(hoverformat=".1f")
    return fig


def lines_figure(table: pd.DataFrame, title: str, ylabel: str, colors: dict | None = None) -> go.Figure:
    """Une courbe par colonne (index = axe des x)."""
    fig = go.Figure()
    for col in table.columns:
        c = (colors or {}).get(col)
        fig.add_trace(go.Scatter(x=table.index.astype(str), y=table[col], name=str(col), mode="lines+markers",
                                 line={"color": c} if c else None, marker={"size": 5}))
    fig.update_layout(**_layout(title, 420, ylabel), hovermode="x unified")
    return fig


def bars_figure(table: pd.DataFrame, title: str, ylabel: str, colors: dict | None = None) -> go.Figure:
    """Barres groupées : une série par colonne."""
    fig = go.Figure()
    for col in table.columns:
        c = (colors or {}).get(col)
        fig.add_trace(go.Bar(x=table.index.astype(str), y=table[col], name=str(col), marker_color=c))
    fig.update_layout(**_layout(title, 420, ylabel), barmode="group")
    return fig


def hourly_bias_figure(bias: dict[int, float]) -> go.Figure:
    s = pd.Series(bias).sort_index()
    fig = go.Figure(go.Bar(x=[f"{int(h)} h" for h in s.index], y=s.values, marker_color=BLUE))
    fig.update_layout(**_layout("Biais de la température prévue, par heure locale (prévu − observé)", 380, "°C"))
    return fig
