import catboost as cgb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from wax_toolbox import Timer

from churnchall.boosters import CatBoostCookie, LgbCookie, XgbCookie, compute_auc_lift
from churnchall.datahandler import DataHandleCookie


def auc_lift_score_func(y, y_pred):
    return compute_auc_lift(y_pred=y_pred, y_true=y, target=0)


auc_lift_scorer = make_scorer(auc_lift_score_func, greater_is_better=True)

DEFAULT_STACKER = Ridge()
DEFAULT_BASE_MDOELS = (
    {
        'model': LgbCookie(random_state=1),
        'num_boost_round': 10
    },
    {
        'model': LgbCookie(random_state=2),
        'num_boost_round': 10
    },
    {
        'model': LgbCookie(random_state=3),
        'num_boost_round': 10
    },
)


class Ensemble():
    def __init__(self,
                 n_splits=5,
                 stacker=DEFAULT_STACKER,
                 base_models=DEFAULT_BASE_MDOELS):
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
            kwargs = clf
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_holdout = X.iloc[test_idx]
                # y_holdout = y.iloc[test_idx]

                with Timer("Fit_Predict Model {} fold {}".format(clf, j)):
                    y_pred = model.fit_predict(X_train, y_train, X_holdout,
                                               **kwargs)

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
        exit()

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
