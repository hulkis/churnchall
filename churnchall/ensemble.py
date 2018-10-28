import catboost as cgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from wax_toolbox import Timer

from churnchall.boosters import CatBoostCookie, LgbCookie, XgbCookie, compute_auc_lift
from churnchall.datahandler import DataHandleCookie
from churnchall.constants import RESULT_DIR

"""
https://github.com/vecxoz/vecstack

--> What is blending? How is it related to stacking?
    Basically it is the same thing. Both approaches use predictions as features.
    Often this terms are used interchangeably.
    The difference is how we generate features (predictions) for the next level:

    stacking: perform cross-validation procedure and predict each part of train set (OOF)
    blending: predict fixed holdout set
"""


def auc_lift_score_func(y, y_pred):
    return compute_auc_lift(y_pred=y_pred, y_true=y, target=0)


auc_lift_scorer = make_scorer(auc_lift_score_func, greater_is_better=True)

DEFAULT_STACKER = Ridge()
DEFAULT_BASE_MODELS = (
    {
        'model': LgbCookie(random_state=1),
        'exec_params': {
            'num_boost_round': 10000,
            'early_stopping_rounds': 200,
        }
    },
    {
        'model': LgbCookie(random_state=2),
        'exec_params': {
            'num_boost_round': 10000,
            'early_stopping_rounds': 200,
        }
    },
    {
        'model': LgbCookie(random_state=3),
        'params_override': {
            'boosting_type': 'dart'
        },
        'exec_params': {
            'num_boost_round': 10000,
            'early_stopping_rounds': 200,
        }
    },
)


class Ensemble():
    def __init__(self,
                 n_splits=5,
                 stacker=DEFAULT_STACKER,
                 base_models=DEFAULT_BASE_MODELS):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        folds = list(
            KFold(n_splits=self.n_splits, shuffle=True,
                  random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            model = clf.pop('model')
            params_override = clf.pop('params_override', {})
            exec_params = clf.pop('exec_params', {})

            model.params_best_fit = {**model.params_best_fit, **params_override}
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_holdout = X.iloc[test_idx]
                y_holdout = y.iloc[test_idx]

                with Timer("Fit_Predict Model {} fold {}".format(clf, j)):
                    y_pred = model.fit_predict(X_train, y_train, X_holdout,
                                               y_holdout, **exec_params)

                S_train[test_idx, i] = y_pred.values.ravel()
                S_test_i[:, j] = model.booster.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(
            self.stacker,
            S_train,
            y.values.ravel(),
            cv=5,
            scoring=auc_lift_scorer)
        print("Stacker score: %.5f (%.5f)" % (results.mean(), results.std()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

    def validate(self, drop_lowimp_features=False):
        datahandle = DataHandleCookie(
            debug=False, drop_lowimp_features=drop_lowimp_features)
        dtrain, dtest = datahandle.get_train_valid_set()
        X_train, y_train = dtrain
        X_test, y_test = dtest

        y_pred = self.fit_predict(X_train, y_train, X_test)

        score = auc_lift_score_func(y_test, y_pred)
        print('Obtained AUC Lift of {}'.format(score))

    def generate_submit(self, drop_lowimp_features=False):
        datahandle = DataHandleCookie(
            debug=False, drop_lowimp_features=drop_lowimp_features)

        dtrain = datahandle.get_train_set()
        dtest = datahandle.get_test_set()

        X_train, y_train = dtrain
        X_test = dtest

        y_pred = self.fit_predict(X_train, y_train, X_test)

        df = pd.DataFrame(y_pred)
        now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")
        df.to_csv(
            RESULT_DIR / "submit_{}.csv".format(now),
            index=False,
            header=False)
