"""Journal des prévisions réelles : enregistre chaque jour la prévision faite en direct
(avant de connaître le résultat), puis la complète avec le réel et la prévision RTE J-1
une fois qu'ils sont publiés.

C'est la seule validation qui ne peut pas être ajustée après coup : contrairement à un backtest,
la prévision est écrite dans le journal avant que le réel n'existe.

    python -m conso.journal record     --model models/prod --out journal
    python -m conso.journal reconcile  --out journal
    python -m conso.journal summary    --out journal

Stockage : un fichier parquet par mois cible (``journal/forecasts_2026-03.parquet``), chaque ligne
étant une demi-heure prévue. Rejouer ``record`` deux fois pour le même jour remplace l'entrée
précédente (par exemple si la tâche planifiée est relancée manuellement).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TZ
from .metrics import metrics as compute_metrics
from .model import Bundle, load_bundle
from .pipeline import forecast_day
from .providers import LiveProvider

JOURNAL_COLUMNS = ["date", "time", "forecast_mw", "lower_mw", "upper_mw", "actual_mw", "rte_j1_mw", "model_version", "recorded_at"]


def _month_path(out_dir: Path, month: pd.Period) -> Path:
    return out_dir / f"forecasts_{month}.parquet"


def _load_month(out_dir: Path, month: pd.Period) -> pd.DataFrame:
    p = _month_path(out_dir, month)
    if not p.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df


def _save_month(out_dir: Path, month: pd.Period, df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values("time").reset_index(drop=True).to_parquet(_month_path(out_dir, month), index=False)


def load_journal(out_dir: str | Path, start=None, end=None) -> pd.DataFrame:
    """Concatène les fichiers mensuels du journal (tous, ou ceux couvrant [start, end])."""
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("forecasts_*.parquet"))
    if not files:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if start is not None:
        df = df[df["time"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["time"] < pd.Timestamp(end)]
    return df.sort_values("time").reset_index(drop=True)


def record_day(bundle: Bundle, provider: LiveProvider, target_date, out_dir: str | Path) -> pd.DataFrame:
    """Prévoit ``target_date`` en direct et enregistre le résultat dans le journal.

    Remplace toute entrée déjà présente pour ce même jour (clé : ``time``). N'écrase jamais
    ``actual_mw`` ni ``rte_j1_mw`` d'un jour déjà partiellement complété par :func:`reconcile`.
    """
    inp = provider.get_inputs(target_date, bundle.bias)
    fc = forecast_day(bundle, inp.d, inp.wt_raw, target_date).dropna(subset=["forecast"])
    if fc.empty:
        raise ValueError(f"Aucune prévision produite pour {target_date} (météo manquante ?).")
    now = pd.Timestamp.now(tz="UTC")
    rows = pd.DataFrame({
        "date": str(pd.Timestamp(target_date).date()), "time": fc.index.tz_convert("UTC"),
        "forecast_mw": fc["forecast"].to_numpy(), "lower_mw": fc["lower"].to_numpy(), "upper_mw": fc["upper"].to_numpy(),
        "actual_mw": np.nan, "rte_j1_mw": np.nan,
        "model_version": str(bundle.meta.get("version", "inconnue")), "recorded_at": now,
    })
    out_dir = Path(out_dir)
    month = fc.index[0].tz_convert(TZ).tz_localize(None).to_period("M")
    current = _load_month(out_dir, month)
    prev = current[current["date"] == rows["date"].iloc[0]]
    if not prev.empty:                                    # conserve un éventuel réel déjà rapproché
        rows = rows.merge(prev[["time", "actual_mw", "rte_j1_mw"]], on="time", how="left", suffixes=("", "_prev"))
        rows["actual_mw"] = rows["actual_mw_prev"].combine_first(rows["actual_mw"])
        rows["rte_j1_mw"] = rows["rte_j1_mw_prev"].combine_first(rows["rte_j1_mw"])
        rows = rows[JOURNAL_COLUMNS]
    merged = pd.concat([current[current["date"] != rows["date"].iloc[0]], rows], ignore_index=True)
    _save_month(out_dir, month, merged)
    return rows


def reconcile(provider: LiveProvider, out_dir: str | Path, lookback_days: int = 12, now: pd.Timestamp | None = None) -> dict:
    """Complète ``actual_mw`` et ``rte_j1_mw`` des entrées récentes, à partir des données RTE en direct.

    ``lookback_days`` doit rester sous la couverture de l'export temps réel de RTE (~80 jours).
    ``now`` : horloge à utiliser (défaut : l'heure réelle) ; paramétrable pour les tests.
    """
    out_dir = Path(out_dir)
    frame = provider.fetch_rte_frame()
    if frame.empty:
        return {"months_updated": [], "rows_filled_actual": 0, "rows_filled_rte": 0}
    now = now or pd.Timestamp.now(tz="UTC")
    since = now - pd.Timedelta(days=lookback_days)
    months = pd.period_range(since.tz_convert(TZ).tz_localize(None).to_period("M"),
                              now.tz_convert(TZ).tz_localize(None).to_period("M"), freq="M")

    touched, n_actual, n_rte = [], 0, 0
    for month in months:
        df = _load_month(out_dir, month)
        if df.empty:
            continue
        missing_a, missing_r = df["actual_mw"].isna().to_numpy(), df["rte_j1_mw"].isna().to_numpy()
        if not (missing_a.any() or missing_r.any()):
            continue
        matched = frame.reindex(df["time"])
        fill_a = missing_a & matched["consommation"].notna().to_numpy()
        fill_r = missing_r & matched["prevision_j1"].notna().to_numpy()
        if not (fill_a.any() or fill_r.any()):
            continue
        df.loc[fill_a, "actual_mw"] = matched["consommation"].to_numpy()[fill_a]
        df.loc[fill_r, "rte_j1_mw"] = matched["prevision_j1"].to_numpy()[fill_r]
        _save_month(out_dir, month, df)
        touched.append(str(month))
        n_actual += int(fill_a.sum())
        n_rte += int(fill_r.sum())
    return {"months_updated": touched, "rows_filled_actual": n_actual, "rows_filled_rte": n_rte}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Métriques du modèle et de RTE J-1 sur les entrées du journal dont le réel est connu."""
    done = df.dropna(subset=["actual_mw"])
    if done.empty:
        return pd.DataFrame(columns=["MAE", "RMSE", "MAPE %", "biais", "MASE", "n"])
    y = done["actual_mw"]
    scale = float((y - y).abs().mean()) or 1.0            # pas de référence naïve dans le journal seul
    rows = {"Modèle (journal, en direct)": compute_metrics(y, done["forecast_mw"], scale)}
    if done["rte_j1_mw"].notna().any():
        rows["RTE J-1"] = compute_metrics(y, done["rte_j1_mw"], scale)
    out = pd.DataFrame(rows).T.drop(columns=["MASE"])     # MASE sans intérêt ici : pas de dénominateur externe
    inside = (y >= done["lower_mw"]) & (y <= done["upper_mw"])
    out.loc["Modèle (journal, en direct)", "couverture intervalle %"] = inside.mean() * 100
    return out


# ----------------------------------------------------------------------------- ligne de commande
def _cli_record(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.model)
    target = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)
    rows = record_day(bundle, LiveProvider(), target, args.out)
    print(f"✔ {len(rows)} points enregistrés pour {target.date()} (modèle {bundle.meta.get('version')})")


def _cli_reconcile(args: argparse.Namespace) -> None:
    r = reconcile(LiveProvider(), args.out, lookback_days=args.lookback_days)
    print(f"✔ réel : {r['rows_filled_actual']} points complétés | RTE J-1 : {r['rows_filled_rte']} points "
          f"| mois touchés : {r['months_updated'] or 'aucun'}")


def _cli_summary(args: argparse.Namespace) -> None:
    df = load_journal(args.out)
    if df.empty:
        print("Journal vide.")
        return
    s = summary(df)
    n_days = df["date"].nunique()
    print(f"{len(df)} points sur {n_days} jours ({df['date'].min()} → {df['date'].max()}), "
          f"dont {df['actual_mw'].notna().sum()} avec le réel connu.\n")
    print(s.round({"MAE": 0, "RMSE": 0, "MAPE %": 2, "biais": 0, "n": 0, "couverture intervalle %": 1}).to_string())


def main() -> None:  # pragma: no cover - point d'entrée en ligne de commande
    p = argparse.ArgumentParser(description="Journal des prévisions réelles.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("record", help="prévoit le lendemain (ou --date) en direct et l'enregistre")
    r.add_argument("--model", default=os.environ.get("CONSO_MODEL", "models/prod"))
    r.add_argument("--out", default=os.environ.get("CONSO_JOURNAL", "journal"))
    r.add_argument("--date", default=None, help="AAAA-MM-JJ ; par défaut, demain")
    r.set_defaults(func=_cli_record)

    c = sub.add_parser("reconcile", help="complète le réel et la prévision RTE J-1 des entrées récentes")
    c.add_argument("--out", default=os.environ.get("CONSO_JOURNAL", "journal"))
    c.add_argument("--lookback-days", type=int, default=12)
    c.set_defaults(func=_cli_reconcile)

    s = sub.add_parser("summary", help="affiche les métriques du journal (modèle vs RTE J-1)")
    s.add_argument("--out", default=os.environ.get("CONSO_JOURNAL", "journal"))
    s.set_defaults(func=_cli_summary)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
