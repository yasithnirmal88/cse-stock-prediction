"""Time-series-safe direction-probability models for CSE research."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Literal
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

LOGGER = logging.getLogger(__name__)
ModelType = Literal["xgboost", "lightgbm"]


class DirectionProbabilityModel:
    """Binary direction classifier; optional boosters load only when requested.

    Scikit-learn GradientBoostingClassifier is a logged development fallback
    only when neither requested booster is installed; it has no early stopping.
    """
    def __init__(self, model_type: ModelType = "xgboost", n_estimators: int = 300,
                 max_depth: int = 4, learning_rate: float = 0.03, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, early_stopping_rounds: int = 30,
                 random_state: int = 42, probability_threshold: float = 0.55) -> None:
        if model_type not in ("xgboost", "lightgbm"):
            raise ValueError("model_type must be 'xgboost' or 'lightgbm'")
        if n_estimators < 1 or max_depth < 1 or learning_rate <= 0 or early_stopping_rounds < 0:
            raise ValueError("invalid estimator, depth, learning rate, or stopping rounds")
        if not 0 < subsample <= 1 or not 0 < colsample_bytree <= 1:
            raise ValueError("subsample and colsample_bytree must be in (0, 1]")
        if not 0 <= probability_threshold <= 1:
            raise ValueError("probability_threshold must be in [0, 1]")
        self.model_type, self.n_estimators, self.max_depth = model_type, int(n_estimators), int(max_depth)
        self.learning_rate, self.subsample = float(learning_rate), float(subsample)
        self.colsample_bytree, self.early_stopping_rounds = float(colsample_bytree), int(early_stopping_rounds)
        self.random_state, self.probability_threshold = int(random_state), float(probability_threshold)
        self.model: Any | None = None
        self.feature_names_: list[str] = []
        self.feature_importance_ = pd.DataFrame(columns=["feature", "importance"])
        self.backend_: str | None = None

    @staticmethod
    def prepare_data(df: pd.DataFrame, feature_cols: list[str],
                     target_col: str = "target_direction") -> tuple[pd.DataFrame, pd.Series]:
        """Return numeric complete-case X,y; preserve chronological row order/index."""
        if not feature_cols:
            raise ValueError("feature_cols must contain at least one feature")
        missing = set(feature_cols + [target_col]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        data = df.loc[:, feature_cols + [target_col]].copy()
        for col in feature_cols + [target_col]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + [target_col])
        if not data[target_col].isin([0, 1]).all():
            raise ValueError(f"Target {target_col!r} must contain only binary 0/1 values")
        return data.loc[:, feature_cols], data[target_col].astype("int8")

    def _make_estimator(self, pos_weight: float) -> tuple[Any, str]:
        """Create desired backend, preferring XGBoost and using installed alternatives."""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree, random_state=self.random_state,
                    scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=1, tree_method="hist"), "xgboost"
            except ImportError:
                LOGGER.warning("XGBoost unavailable; attempting LightGBM")
        if self.model_type == "lightgbm" or self.model_type == "xgboost":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree, random_state=self.random_state,
                    scale_pos_weight=pos_weight, verbosity=-1, n_jobs=1), "lightgbm"
            except ImportError:
                LOGGER.warning("LightGBM unavailable; attempting XGBoost")
        if self.model_type == "lightgbm":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree, random_state=self.random_state,
                    scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=1, tree_method="hist"), "xgboost"
            except ImportError:
                LOGGER.warning("Neither boosting library installed; using sklearn development fallback")
        return GradientBoostingClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            random_state=self.random_state), "sklearn_gradient_boosting"


    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "DirectionProbabilityModel":
        """Fit only training rows; validation rows must be later in time."""
        X = pd.DataFrame(X_train).copy()
        y = pd.Series(y_train, index=X.index).astype(int)
        if X.empty or len(X) != len(y) or not y.isin([0, 1]).all() or y.nunique() < 2:
            raise ValueError("Training must be non-empty, aligned, binary, and contain both classes")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")
        self.feature_names_ = list(X.columns)
        counts = y.value_counts()
        weight = float(counts.get(0, 0) / counts.get(1, 1))
        self.model, self.backend_ = self._make_estimator(weight)
        LOGGER.info("Training %s: %d rows, %d predictors, positive weight %.3f", self.backend_, len(X), X.shape[1], weight)
        kwargs: dict[str, Any] = {}
        if X_val is not None:
            xv = pd.DataFrame(X_val).loc[:, self.feature_names_]
            yv = pd.Series(y_val, index=xv.index).astype(int)
            if xv.empty or not yv.isin([0, 1]).all() or yv.nunique() < 2:
                raise ValueError("Validation data must be non-empty and contain both classes")
            if self.early_stopping_rounds and self.backend_ == "xgboost":
                kwargs = {"eval_set": [(xv, yv)], "verbose": False, "early_stopping_rounds": self.early_stopping_rounds}
            elif self.early_stopping_rounds and self.backend_ == "lightgbm":
                from lightgbm import early_stopping, log_evaluation
                kwargs = {"eval_set": [(xv, yv)], "callbacks": [early_stopping(self.early_stopping_rounds, verbose=False), log_evaluation(0)]}
            elif self.backend_ == "sklearn_gradient_boosting":
                LOGGER.warning("sklearn fallback has no native early stopping")
        self.model.fit(X, y, **kwargs)
        importance = getattr(self.model, "feature_importances_", np.zeros(len(self.feature_names_)))
        self.feature_importance_ = pd.DataFrame({"feature": self.feature_names_, "importance": importance})
        self.feature_importance_.sort_values("importance", ascending=False, inplace=True, ignore_index=True)
        LOGGER.info("Training complete (%s)", self.backend_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities for class 1 in the fitted feature order."""
        if self.model is None:
            raise RuntimeError("Model is not trained")
        frame = pd.DataFrame(X)
        if self.feature_names_:
            frame = frame.loc[:, self.feature_names_]
        classes = list(self.model.classes_)
        return np.asarray(self.model.predict_proba(frame)[:, classes.index(1)], dtype=float) if 1 in classes else np.zeros(len(frame))

    def predict_signal(self, X: pd.DataFrame, threshold: float | None = None) -> pd.Series:
        """Return Buy=1/Cash=0 signals and preserve input index."""
        cutoff = self.probability_threshold if threshold is None else float(threshold)
        if not 0 <= cutoff <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return pd.Series((self.predict_proba(X) >= cutoff).astype("int8"), index=X.index, name="signal")


    def walk_forward_train_predict(self, df: pd.DataFrame, feature_cols: list[str],
                                   train_window: int, test_window: int, step_size: int) -> pd.DataFrame:
        """Chronological rolling windows; train_window=0 means expanding history."""
        if train_window < 0 or test_window < 1 or step_size < 1:
            raise ValueError("train_window >= 0; test_window and step_size must be positive")


        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Walk-forward input must have a DatetimeIndex")
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise ValueError("Walk-forward input index must be sorted and unique")
        X, y = self.prepare_data(df, feature_cols)
        blocks: list[pd.DataFrame] = []
        start = train_window if train_window else 1
        while start < len(X):
            left = max(0, start - train_window) if train_window else 0
            stop = min(start + test_window, len(X))
            X_fit, y_fit = X.iloc[left:start], y.iloc[left:start]
            X_test, y_test = X.iloc[start:stop], y.iloc[start:stop]
            if y_fit.nunique() < 2:
                LOGGER.warning("Skipping OOS block %s: training has one class", X_test.index[0])
            else:
                # Never pass test rows or labels to fit, validation, or hyperparameter selection.
                self.train(X_fit, y_fit)
                probability = self.predict_proba(X_test)
                blocks.append(pd.DataFrame({"y_true": y_test, "probability": probability,
                    "signal": (probability >= self.probability_threshold).astype("int8")}, index=X_test.index))
            start += step_size
        if not blocks:
            return pd.DataFrame(columns=["y_true", "probability", "signal"],
                                index=pd.DatetimeIndex([], name=df.index.name))
        return pd.concat(blocks).sort_index()

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top fitted features sorted by importance."""
        if top_n < 1:
            raise ValueError("top_n must be positive")
        return self.feature_importance_.head(top_n).copy()

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
        """Return basic binary metrics, defining undefined precision/recall as zero."""
        truth, pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
        if not len(truth) or len(truth) != len(pred):
            raise ValueError("y_true and y_pred must have the same non-zero length")
        return {"accuracy": float(accuracy_score(truth, pred)),
                "precision": float(precision_score(truth, pred, zero_division=0)),
                "recall": float(recall_score(truth, pred, zero_division=0)),
                "f1": float(f1_score(truth, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(truth, pred))}

    def save(self, path: str | Path) -> Path:
        """Save the fitted model and metadata using joblib."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DirectionProbabilityModel":
        """Load model saved by :meth:`save`."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("File does not contain a DirectionProbabilityModel")
        return model

