"""Modèles alternatifs à LightGBM et mélange à poids appris.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer

from .config import DEFAULT_BLEND_WEIGHTS, DEFAULT_MLP_EPOCHS, DEFAULT_ROUNDS, DEFAULT_SEEDS
from .features import FEATURE_COLUMNS
from .model import SeedEnsemble, fit_lgb_seeds

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CATEGORIELLES = ["heure", "jour_semaine", "mois"]


def _cible(X: pd.DataFrame, y: pd.Series, mask: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    tgt = y - X["level7"]
    tr = mask & tgt.notna().to_numpy()
    if tr.sum() == 0:
        raise ValueError("Aucune ligne d'entraînement valide.")
    return tr, tgt


class LgbmSeedPoint:
    """Enveloppe un ``lgb.Booster`` ou un ``SeedEnsemble`` pour l'interface commune du mélange."""

    kind = "lgbm_seed"

    def __init__(self, model: lgb.Booster | SeedEnsemble | None = None):
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series, mask: np.ndarray, ref_end: pd.Timestamp | None = None,
            feats: list[str] | None = None, rounds: int = DEFAULT_ROUNDS, seeds: int = DEFAULT_SEEDS,
            params: dict | None = None) -> LgbmSeedPoint:
        self.model = fit_lgb_seeds(X, y, mask, rounds, params, None, ref_end, feats, n_seeds=seeds)
        return self

    def predict(self, Xf: pd.DataFrame) -> np.ndarray:
        return self.model.predict(Xf)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        models = self.model.models if isinstance(self.model, SeedEnsemble) else [self.model]
        for i, m in enumerate(models):
            m.save_model(str(out / f"seed{i}.txt"))
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> LgbmSeedPoint:
        fichiers = sorted(Path(out_dir).glob("seed*.txt"), key=lambda p: int(p.stem[4:]))
        models = [lgb.Booster(model_file=str(f)) for f in fichiers]
        return cls(models[0] if len(models) == 1 else SeedEnsemble(models))


class Lisse:
    """Une régression Ridge par créneau de 30 min, sur des splines de température et le reste en linéaire.
    """

    kind = "lisse"
    SPLINES = ["temp_nat", "te", "te_lent"]

    def __init__(self, feats: list[str] | None = None, alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0), n_knots: int = 7):
        self.feats = list(feats or FEATURE_COLUMNS)
        self.lineaires = [c for c in self.feats if c not in [*self.SPLINES, *CATEGORIELLES, "jour_annee"]]
        self.alphas, self.n_knots = alphas, n_knots

    def _design(self, Xf: pd.DataFrame) -> np.ndarray:
        lin = Xf[self.lineaires].to_numpy(dtype=float)
        lin = np.where(np.isnan(lin), self.med_lin, lin)
        spl = [self.spl[c].transform(Xf[[c]].fillna(self.med_spl[c]).to_numpy()) for c in self.SPLINES]
        dow = np.eye(7)[Xf["jour_semaine"].to_numpy(dtype=int)]
        ang = 2 * np.pi * Xf["jour_annee"].to_numpy(dtype=float) / 365.25
        saison = np.column_stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)])
        return np.hstack([*spl, lin, dow, saison])

    def fit(self, X: pd.DataFrame, y: pd.Series, mask: np.ndarray, ref_end: pd.Timestamp | None = None) -> Lisse:
        tr, tgt = _cible(X, y, mask)
        Xf = X.loc[tr, self.feats]
        self.med_lin = np.nan_to_num(np.nanmedian(Xf[self.lineaires].to_numpy(dtype=float), axis=0), nan=0.0)
        self.med_spl = {c: float(np.nanmedian(Xf[c])) for c in self.SPLINES}
        self.spl = {c: SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile", extrapolation="linear")
                    .fit(Xf[[c]].fillna(self.med_spl[c]).to_numpy()) for c in self.SPLINES}
        Z = self._design(Xf)
        self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z = (Z - self.mu) / self.sd
        slot = np.rint(Xf["heure"].to_numpy(dtype=float) * 2).astype(int)
        t = tgt[tr].to_numpy(dtype=float)
        self.models = {int(s): RidgeCV(alphas=self.alphas).fit(Z[slot == s], t[slot == s]) for s in np.unique(slot)}
        return self

    def predict(self, Xf: pd.DataFrame) -> np.ndarray:
        Z = (self._design(Xf) - self.mu) / self.sd
        slot = np.rint(Xf["heure"].to_numpy(dtype=float) * 2).astype(int)
        out = np.full(len(Xf), np.nan)
        for s, m in self.models.items():
            k = slot == s
            if k.any():
                out[k] = m.predict(Z[k])
        return out

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"feats": self.feats, "lineaires": self.lineaires, "alphas": self.alphas, "n_knots": self.n_knots,
                     "med_lin": self.med_lin, "med_spl": self.med_spl, "spl": self.spl,
                     "mu": self.mu, "sd": self.sd, "models": self.models}, out / "lisse.joblib")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> Lisse:
        d = joblib.load(Path(out_dir) / "lisse.joblib")
        obj = cls(feats=d["feats"], alphas=d["alphas"], n_knots=d["n_knots"])
        obj.lineaires = d["lineaires"]
        obj.med_lin, obj.med_spl, obj.spl = d["med_lin"], d["med_spl"], d["spl"]
        obj.mu, obj.sd, obj.models = d["mu"], d["sd"], d["models"]
        return obj


class Hybride:
    """``Lisse`` puis LightGBM sur son résidu."""

    kind = "hybride"

    def __init__(self, feats: list[str] | None = None, seeds: int = DEFAULT_SEEDS, rounds: int = DEFAULT_ROUNDS):
        self.feats, self.seeds, self.rounds = list(feats or FEATURE_COLUMNS), seeds, rounds

    def fit(self, X: pd.DataFrame, y: pd.Series, mask: np.ndarray, ref_end: pd.Timestamp | None = None) -> Hybride:
        self.lisse = Lisse(self.feats).fit(X, y, mask, ref_end)
        base = pd.Series(self.lisse.predict(X[self.feats]), index=X.index)
        self.gbm = LgbmSeedPoint().fit(X, y - base, mask & base.notna().to_numpy(), ref_end, self.feats, self.rounds, self.seeds)
        return self

    def predict(self, Xf: pd.DataFrame) -> np.ndarray:
        return self.lisse.predict(Xf) + self.gbm.predict(Xf)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.lisse.save(out / "lisse")
        self.gbm.save(out / "gbm")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> Hybride:
        out = Path(out_dir)
        obj = cls()
        obj.lisse = Lisse.load(out / "lisse")
        obj.gbm = LgbmSeedPoint.load(out / "gbm")
        obj.feats = obj.lisse.feats
        return obj


class CatBoostSeed:
    """Moyenne de modèles CatBoost (arbres symétriques) entraînés avec des graines différentes."""

    kind = "catboost"

    def __init__(self, feats: list[str] | None = None, seeds: int = DEFAULT_SEEDS, iters: int = DEFAULT_ROUNDS, depth: int = 8):
        if not HAS_CATBOOST:
            raise ImportError("catboost n'est pas installé (pip install catboost).")
        self.feats, self.seeds, self.iters, self.depth = list(feats or FEATURE_COLUMNS), seeds, iters, depth

    def fit(self, X: pd.DataFrame, y: pd.Series, mask: np.ndarray, ref_end: pd.Timestamp | None = None) -> CatBoostSeed:
        tr, tgt = _cible(X, y, mask)
        self.models = [CatBoostRegressor(iterations=self.iters, depth=self.depth, learning_rate=0.1, eval_metric="MAE",
                                         random_seed=s, verbose=0, thread_count=-1, allow_writing_files=False)
                       .fit(X.loc[tr, self.feats], tgt[tr]) for s in range(self.seeds)]
        return self

    def predict(self, Xf: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(Xf) for m in self.models], axis=0)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save_model(str(out / f"seed{i}.cbm"))
        (out / "config.json").write_text(json.dumps({"feats": self.feats, "depth": self.depth}), encoding="utf-8")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> CatBoostSeed:
        if not HAS_CATBOOST:
            raise ImportError("catboost n'est pas installé (pip install catboost).")
        out = Path(out_dir)
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        obj = cls(feats=cfg["feats"], depth=cfg["depth"], seeds=1)
        fichiers = sorted(out.glob("seed*.cbm"), key=lambda p: int(p.stem[4:]))
        obj.models = []
        for f in fichiers:
            m = CatBoostRegressor()
            m.load_model(str(f))
            obj.models.append(m)
        obj.seeds = len(obj.models)
        return obj


NUM_COLS_EXCLUS = CATEGORIELLES


def _prep_mlp(Xf: pd.DataFrame, num_cols: list[str], med: np.ndarray, miss_idx: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    num = Xf[num_cols].to_numpy(dtype=float)
    miss = np.isnan(num)
    num = np.where(miss, med, num)
    z = np.hstack([(num - mu) / sd, miss[:, miss_idx]]).astype(np.float32)
    cat = np.column_stack([np.rint(Xf["heure"].to_numpy(dtype=float) * 2), Xf["jour_semaine"].to_numpy(dtype=float), Xf["mois"].to_numpy(dtype=float) - 1])
    cat = np.minimum(np.clip(np.nan_to_num(cat), 0, None), [47, 6, 11]).astype(np.int64)
    return z, cat


if HAS_TORCH:

    class _Net(nn.Module):
        def __init__(self, n_in: int, hidden: int, drop: float):
            super().__init__()
            self.e_slot = nn.Embedding(48, 8)
            self.e_dow = nn.Embedding(7, 3)
            self.e_mois = nn.Embedding(12, 3)
            self.mlp = nn.Sequential(
                nn.Linear(n_in + 14, hidden), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1),
            )

        def forward(self, x, c):
            return self.mlp(torch.cat([x, self.e_slot(c[:, 0]), self.e_dow(c[:, 1]), self.e_mois(c[:, 2])], dim=1)).squeeze(1)


class MLPPoint:
    """Réseau à trois couches cachées avec embeddings du créneau, du jour de semaine et du mois.
    """

    kind = "mlp"

    def __init__(self, feats: list[str] | None = None, seeds: int = DEFAULT_SEEDS, epochs: int = DEFAULT_MLP_EPOCHS,
                 hidden: int = 256, drop: float = 0.1, lr: float = 2e-3, wd: float = 1e-4, bs: int = 1024):
        if not HAS_TORCH:
            raise ImportError("torch n'est pas installé (pip install torch).")
        self.feats = list(feats or FEATURE_COLUMNS)
        self.num_cols = [c for c in self.feats if c not in NUM_COLS_EXCLUS]
        self.seeds, self.epochs, self.hidden, self.drop, self.lr, self.wd, self.bs = seeds, epochs, hidden, drop, lr, wd, bs

    def fit(self, X: pd.DataFrame, y: pd.Series, mask: np.ndarray, ref_end: pd.Timestamp | None = None) -> MLPPoint:
        tr, tgt_s = _cible(X, y, mask)
        tgt = tgt_s.to_numpy(dtype=float)
        num = X.loc[tr, self.num_cols].to_numpy(dtype=float)
        miss = np.isnan(num)
        self.med = np.nan_to_num(np.nanmedian(num, axis=0), nan=0.0)
        self.miss_idx = np.where(miss.any(axis=0))[0]
        num = np.where(miss, self.med, num)
        self.mu, self.sd = num.mean(axis=0), num.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        z, c = _prep_mlp(X.loc[tr, self.feats], self.num_cols, self.med, self.miss_idx, self.mu, self.sd)
        self.ts = float(np.std(tgt[tr])) or 1.0
        t = torch.from_numpy((tgt[tr] / self.ts).astype(np.float32))
        z, c = torch.from_numpy(z), torch.from_numpy(c)
        n = len(t)
        steps = max(1, int(np.ceil(n / self.bs)))
        self.nets = []
        for s in range(self.seeds):
            torch.manual_seed(s)
            net = _Net(z.shape[1], self.hidden, self.drop)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=max(1, self.epochs * steps))
            loss_fn = nn.SmoothL1Loss(beta=0.3)
            net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, self.bs):
                    idx = perm[i:i + self.bs]
                    opt.zero_grad()
                    loss_fn(net(z[idx], c[idx]), t[idx]).backward()
                    opt.step()
                    sched.step()
            self.nets.append(net.eval())
        return self

    def predict(self, Xf: pd.DataFrame) -> np.ndarray:
        z, c = _prep_mlp(Xf, self.num_cols, self.med, self.miss_idx, self.mu, self.sd)
        z, c = torch.from_numpy(z), torch.from_numpy(c)
        with torch.no_grad():
            preds = [np.concatenate([n(z[i:i + 50000], c[i:i + 50000]).numpy() for i in range(0, len(z), 50000)]) for n in self.nets]
        return np.mean(preds, axis=0) * self.ts

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), out / f"seed{i}.pt")
        joblib.dump({"feats": self.feats, "num_cols": self.num_cols, "med": self.med, "miss_idx": self.miss_idx,
                     "mu": self.mu, "sd": self.sd, "ts": self.ts, "hidden": self.hidden, "drop": self.drop}, out / "scaler.joblib")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> MLPPoint:
        if not HAS_TORCH:
            raise ImportError("torch n'est pas installé (pip install torch).")
        out = Path(out_dir)
        d = joblib.load(out / "scaler.joblib")
        obj = cls(feats=d["feats"], seeds=0, epochs=0, hidden=d["hidden"], drop=d["drop"])
        obj.num_cols = d["num_cols"]
        obj.med, obj.miss_idx, obj.mu, obj.sd, obj.ts = d["med"], d["miss_idx"], d["mu"], d["sd"], d["ts"]
        fichiers = sorted(out.glob("seed*.pt"), key=lambda p: int(p.stem[4:]))
        n_in = len(obj.num_cols) + len(obj.miss_idx)
        obj.nets = []
        for f in fichiers:
            net = _Net(n_in, obj.hidden, obj.drop)
            net.load_state_dict(torch.load(f, weights_only=True))
            obj.nets.append(net.eval())
        obj.seeds = len(obj.nets)
        return obj


_LOADERS = {"lgbm_seed": LgbmSeedPoint.load, "lisse": Lisse.load, "hybride": Hybride.load,
            "catboost": CatBoostSeed.load, "mlp": MLPPoint.load}
_MEMBRES = {"lgbm": ("lgbm_seed", LgbmSeedPoint), "hybride": ("hybride", Hybride),
            "catboost": ("catboost", CatBoostSeed), "mlp": ("mlp", MLPPoint), "lisse": ("lisse", Lisse)}


def load_point_model(kind: str, path: str | Path):
    if kind not in _LOADERS:
        raise ValueError(f"kind de modèle inconnu : {kind!r} (attendu : {sorted(_LOADERS)})")
    return _LOADERS[kind](path)


class Blend:
    """Moyenne pondérée de modèles hétérogènes."""

    kind = "blend"

    def __init__(self, models: dict[str, object], weights: dict[str, float]):
        self.models = dict(models)
        self.weights = {k: float(weights.get(k, 0.0)) for k in self.models}

    def predict(self, Xf: pd.DataFrame) -> np.ndarray:
        total = np.zeros(len(Xf))
        for name, m in self.models.items():
            w = self.weights[name]
            if w != 0.0:
                total = total + w * m.predict(Xf)
        return total

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, m in self.models.items():
            m.save(out / name)
            manifest[name] = {"kind": m.kind, "weight": self.weights[name]}
        (out / "blend.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> Blend:
        out = Path(out_dir)
        manifest = json.loads((out / "blend.json").read_text(encoding="utf-8"))
        models = {name: load_point_model(info["kind"], out / name) for name, info in manifest.items()}
        weights = {name: info["weight"] for name, info in manifest.items()}
        return cls(models, weights)


def fit_blend(
    X: pd.DataFrame, y: pd.Series, mask: np.ndarray, ref_end: pd.Timestamp | None = None,
    feats: list[str] | None = None, weights: dict[str, float] | None = None, seeds: int = DEFAULT_SEEDS,
    rounds: int = DEFAULT_ROUNDS, mlp_epochs: int = DEFAULT_MLP_EPOCHS,
) -> Blend:
    """Entraîne chaque membre du mélange dont le poids est non nul (ou explicitement demandé via
    ``weights``) et renvoie le ``Blend``. Par défaut, poids et membres viennent de
    ``DEFAULT_BLEND_WEIGHTS`` (le ``mel_nnls`` du notebook 06)."""
    feats = list(feats or FEATURE_COLUMNS)
    weights = dict(weights if weights is not None else DEFAULT_BLEND_WEIGHTS)
    fitted: dict[str, object] = {}
    for name, w in weights.items():
        if name not in _MEMBRES:
            raise ValueError(f"membre de mélange inconnu : {name!r} (attendu : {sorted(_MEMBRES)})")
        if w == 0.0:
            continue  # poids nul : mesuré dans le notebook, mais inutile de l'entraîner en production
        _, cls_ = _MEMBRES[name]
        if name == "lgbm":
            fitted[name] = LgbmSeedPoint().fit(X, y, mask, ref_end, feats, rounds, seeds)
        elif name == "hybride":
            fitted[name] = Hybride(feats, seeds, rounds).fit(X, y, mask, ref_end)
        elif name == "catboost":
            fitted[name] = CatBoostSeed(feats, seeds, rounds).fit(X, y, mask, ref_end)
        elif name == "mlp":
            fitted[name] = MLPPoint(feats, seeds, mlp_epochs).fit(X, y, mask, ref_end)
        elif name == "lisse":
            fitted[name] = Lisse(feats).fit(X, y, mask, ref_end)
    if not fitted:
        raise ValueError("Aucun membre à poids non nul : rien à entraîner.")
    return Blend(fitted, weights)
