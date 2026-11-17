"""Candidate models. Each produces home win prob, expected margin (home-away), expected total.

Logistic regression is a classifier, so its margin/total heads are ridge regressions on the
same standardized features.
"""
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7


def _heads(name, hp):
    if name == "logreg":
        return (make_pipeline(StandardScaler(), LogisticRegression(C=hp.get("lr_C", 0.05), max_iter=2000)),
                make_pipeline(StandardScaler(), Ridge(alpha=hp.get("ridge_alpha", 50.0))),
                make_pipeline(StandardScaler(), Ridge(alpha=hp.get("ridge_alpha", 50.0))))
    if name == "rf":
        kw = dict(n_estimators=150, min_samples_leaf=hp.get("rf_leaf", 40), max_features=0.33,
                  max_samples=0.5, n_jobs=-1, random_state=SEED)
        return RandomForestClassifier(**kw), RandomForestRegressor(**kw), RandomForestRegressor(**kw)
    if name == "lgbm":
        kw = dict(n_estimators=hp.get("gb_trees", 400), learning_rate=hp.get("gb_lr", 0.02),
                  num_leaves=hp.get("gb_leaves", 15), min_child_samples=hp.get("gb_min_child", 80),
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  random_state=SEED, verbose=-1)
        return LGBMClassifier(**kw), LGBMRegressor(**kw), LGBMRegressor(**kw)
    raise ValueError(name)


class Model:
    def __init__(self, name, hp=None):
        self.name, self.hp = name, hp or {}

    def fit(self, X, y_win, y_margin, y_total):
        if self.name == "elo":
            return self
        self.clf, self.reg_m, self.reg_t = _heads(self.name, self.hp)
        self.clf.fit(X, y_win)
        self.reg_m.fit(X, y_margin)
        self.reg_t.fit(X, y_total)
        return self

    def predict(self, X):
        if self.name == "elo":      # pure rating baseline, no fitting on outcomes beyond Elo updates
            return X["elo_p"].to_numpy(), X["elo_margin"].to_numpy(), X["exp_total"].to_numpy()
        return self.clf.predict_proba(X)[:, 1], self.reg_m.predict(X), self.reg_t.predict(X)

    def importance(self, cols):
        if self.name == "lgbm":
            return dict(zip(cols, self.clf.booster_.feature_importance("gain")))
        if self.name == "rf":
            return dict(zip(cols, self.clf.feature_importances_))
        if self.name == "logreg":
            return dict(zip(cols, np.abs(self.clf[-1].coef_[0])))
        return {}
