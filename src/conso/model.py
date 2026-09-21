"""Modèle LightGBM : entraînement, intervalle conforme, sauvegarde et chargement.

La cible n'est pas la consommation brute mais son écart au niveau récent ``level7``
(moyenne des 7 jours complets T-8 ... T-2) : la baisse tendancielle du niveau est ainsi absorbée.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import DEFAULT_PARAMS, TEMP_GROUP_EDGES, TEMP_GROUP_NAMES
from .features import FEATURE_COLUMNS


def train_weights(index: pd.DatetimeIndex, ref_end: pd.Timestamp, half_life: float | None):
    """Poids par récence (demi-vie en années) ; ``None`` = pas de pondération."""
    if half_life is None:
        return None
    age_years = np.asarray((ref_end - index).total_seconds()) / (365.25 * 86400)
    return 0.5 ** (age_years / half_life)


def fit_lgb(
    X: pd.DataFrame, y: pd.Series, mask: np.ndarray, rounds: int, params: dict | None = None,
    half_life: float | None = None, ref_end: pd.Timestamp | None = None, feats: list[str] | None = None,
) -> lgb.Booster:
    """Entraîne un modèle sur l'écart au niveau récent, pour les lignes de ``mask``."""
    feats = feats or FEATURE_COLUMNS
    tgt = y - X["level7"]
    tr = mask & tgt.notna().to_numpy()
    if tr.sum() == 0:
        raise ValueError("Aucune ligne d'entraînement valide.")
    w = train_weights(X.index[tr], ref_end, half_life) if ref_end is not None else None
    ds = lgb.Dataset(X.loc[tr, feats], label=tgt[tr], weight=w)
    return lgb.train({**DEFAULT_PARAMS, **(params or {})}, ds, num_boost_round=rounds)


def predict_level(model: lgb.Booster, X: pd.DataFrame, feats: list[str] | None = None) -> pd.Series:
    """Prévision en MW : niveau récent + écart prédit. NaN sans météo cible."""
    feats = feats or FEATURE_COLUMNS
    p = pd.Series(model.predict(X[feats]), index=X.index) + X["level7"]
    return p.where(X["temp_nat"].notna())


def temp_groups(te: pd.Series) -> pd.Series:
    """Groupe de température (froid / doux / chaud / très chaud) d'après la température lissée."""
    names = np.array(TEMP_GROUP_NAMES, dtype=object)
    idx = np.digitize(te.to_numpy(dtype=float), TEMP_GROUP_EDGES)
    g = pd.Series(names[np.minimum(idx, len(names) - 1)], index=te.index)
    return g.where(te.notna())


def conformal_qhat(scores: pd.Series, groups: pd.Series, alpha: float = 0.10, min_group: int = 200) -> dict[str, float]:
    """Élargissement conforme (CQR) par groupe : quantile (1-alpha) corrigé des scores de non-conformité."""

    def q(s: np.ndarray) -> float:
        k = min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))
        return float(np.quantile(s, k, method="higher"))

    ok = scores.notna() & groups.notna()
    overall = q(scores[ok].to_numpy())
    out = {}
    for g in TEMP_GROUP_NAMES:
        s = scores[ok & (groups == g)].to_numpy()
        out[g] = q(s) if len(s) >= min_group else overall # groupe trop petit : valeur globale
    return out


@dataclass
class Bundle:
    """Tout ce qu'il faut pour prévoir : modèles, biais météo horaire, calibration, métadonnées."""

    point: lgb.Booster
    q_lo: lgb.Booster
    q_hi: lgb.Booster
    bias: pd.Series
    qhat: dict[str, float]
    meta: dict = field(default_factory=dict)

    @property
    def feats(self) -> list[str]:
        return self.meta.get("features", FEATURE_COLUMNS)


def predict_with_interval(bundle: Bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Prévision ponctuelle et intervalle calibré (bornes garanties : lower <= forecast <= upper)."""
    f = bundle.feats
    point = predict_level(bundle.point, X, f)
    lo, hi = predict_level(bundle.q_lo, X, f), predict_level(bundle.q_hi, X, f)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    adj = temp_groups(X["te"]).map(bundle.qhat).astype(float)
    lower = np.minimum(lo - adj, point)
    upper = np.maximum(hi + adj, point)
    return pd.DataFrame({"forecast": point, "lower": lower, "upper": upper}, index=X.index)


def save_bundle(bundle: Bundle, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle.point.save_model(str(out / "point.txt"))
    bundle.q_lo.save_model(str(out / "q05.txt"))
    bundle.q_hi.save_model(str(out / "q95.txt"))
    meta = {**bundle.meta, "bias_hourly": {int(h): float(v) for h, v in bundle.bias.items()}, "qhat": bundle.qhat}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return out


def load_bundle(model_dir: str | Path) -> Bundle:
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    bias = pd.Series({int(h): v for h, v in meta.pop("bias_hourly").items()}).sort_index()
    qhat = meta.pop("qhat")
    return Bundle(
        point=lgb.Booster(model_file=str(d / "point.txt")),
        q_lo=lgb.Booster(model_file=str(d / "q05.txt")),
        q_hi=lgb.Booster(model_file=str(d / "q95.txt")),
        bias=bias, qhat=qhat, meta=meta,
    )
