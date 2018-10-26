import copy
import warnings

import catboost as cgb
import hyperopt
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from wax_toolbox import Timer

from churnchall.constants import MODEL_DIR, RESULT_DIR
from churnchall.datahandler import DataHandleCookie, to_gradboost_dataset
from churnchall.tuning import HyperParamsTuningMixin

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


def compute_auc_lift(y_pred, y_true, target):
    df_lift = pd.DataFrame({'pred': y_pred, 'true': y_true})

    # Sort by prediction
    if target == 1:
        df_lift = df_lift.sort_values("pred", ascending=False)
    elif target == 0:
        df_lift = df_lift.sort_values("pred", ascending=True)
    else:
        raise ValueError

    # compute lift score for each sample of population
    nb_targets = float(df_lift[df_lift['true'] == target].shape[0])
    df_lift["auclift"] = (df_lift["true"] == target).cumsum() / nb_targets
    auc_lift = df_lift["auclift"].mean()
    return auc_lift


def lgb_auc_lift(y_pred, y_true, target=0):
    y_true = y_true.label
    auc_lift = compute_auc_lift(y_pred, y_true, target)
    return "AUC Lift", auc_lift, True


def xgb_auc_lift(y_pred, y_true, target=0):
    y_true = y_true.get_label()
    auc_lift = compute_auc_lift(y_pred, y_true, target)
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return "AUC_Lift", auc_lift


def get_df_importance(booster):
    if hasattr(booster, "feature_name"):  # lightgbm
        idx = booster.feature_name()
        arr = booster.feature_importance()
        df = pd.DataFrame(index=idx, data=arr, columns=["importance"])
    elif hasattr(booster, "get_score"):  # xgboost
        serie = pd.Series(booster.get_score())
        df = pd.DataFrame(columns=["importance"], data=serie)
    elif hasattr(booster, "get_feature_importance"):  # catboost
        idx = booster.feature_names_
        arr = booster.get_feature_importance()
        df = pd.DataFrame(index=idx, data=arr, columns=["importance"])
    else:
        raise ValueError(type(booster))

    # Traduce in percentage:
    df["importance"] = df["importance"] / df["importance"].sum() * 100
    df = df.sort_values("importance", ascending=False)
    return df


class BaseModelCookie(DataHandleCookie, HyperParamsTuningMixin):

    # Attributes to be defined:
    @property
    def algo():
        raise NotImplementedError

    @property
    def common_params():
        raise NotImplementedError

    @property
    def params_best_fit():
        raise NotImplementedError

    def save_model(self, booster):
        now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")
        f = MODEL_DIR / "{}_model_{}.txt".format(self.algo, now)
        booster.save_model(f.as_posix())
        return f

    @staticmethod
    def _generate_plot(eval_hist):
        try:
            from plotlyink import register_iplot_accessor
            register_iplot_accessor()
            dfhist = pd.DataFrame(eval_hist)
            fig = dfhist.iplot.scatter(as_figure=True)
            import plotly
            now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")
            filepath = RESULT_DIR / 'lgb_eval_hist_{}.html'.format(now)
            plotly.offline.plot(fig, filename=filepath.as_posix())
        except ImportError:
            pass

    # Methods to be implemented
    def train():
        raise NotImplementedError

    def validate():
        raise NotImplementedError

    def cv():
        raise NotImplementedError


class LgbCookie(BaseModelCookie):

    algo = 'lightgbm'

    # Common params for LightGBM
    common_params = {
        "verbose": -1,
        "nthreads": 16,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        "scale_pos_weight": 0.97,  # used only in binary application, weight of labels with positive class
        "objective": "xentropy",  # better optimize on cross-entropy loss for auc
        "metric": {"auc"},  # alias for roc_auc_score
    }

    # Best fit params
    params_best_fit = {
        "boosting_type": "gbdt",  # algorithm to use
        "learning_rate": 0.04,
        "num_leaves": 10,  # we should let it be smaller than 2^(max_depth)
        # "min_data_in_leaf": 20,  # Minimum number of data need in a child
        "max_depth": -1,  # -1 means no limit
        "bagging_fraction": 0.9487944316907742,  # Subsample ratio of the training instance.
        "feature_fraction": 0.9763410806631222,  # Subsample ratio of columns when constructing each tree.
        "bagging_freq": 14,  # frequence of subsample, <=0 means no enable
        # "max_bin": 200,
        'min_data_in_leaf': 14,  # minimal number of data in one leaf
        # 'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        # 'subsample_for_bin': 200000,  # Number of samples for constructing bin
        # 'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        # 'reg_alpha': 0,  # L1 regularization term on weights
        # 'reg_lambda': 0,  # L2 regularization term on weights
        **common_params,
    }

    # tuning attributes in relation to HyperParamsTuningMixin
    int_params = ("num_leaves", "max_depth", "min_data_in_leaf",
                  "bagging_freq")
    float_params = ("learning_rate", "feature_fraction", "bagging_fraction")
    hypertuning_space = {
        "boosting": hyperopt.hp.choice("boosting", ["gbdt", "dart"]),  # "rf",
        "num_leaves": hyperopt.hp.quniform("num_leaves", 10, 60, 2),
        "min_data_in_leaf": hyperopt.hp.quniform("min_data_in_leaf", 5, 20, 2),
        # "learning_rate": hyperopt.hp.uniform("learning_rate", 0.001, 0.1),
        "feature_fraction": hyperopt.hp.uniform("feature_fraction", 0.85,
                                                0.99),
        "bagging_fraction": hyperopt.hp.uniform("bagging_fraction", 0.85,
                                                0.99),
        "bagging_freq": hyperopt.hp.quniform("bagging_freq", 6, 18, 2),
    }

    def validate(self, save_model=True, **kwargs):
        dtrain, dtest = self.get_train_valid_set(as_lgb_dataset=True)
        valid_sets = [dtrain, dtest]
        valid_names = ['train', 'test']
        booster = lgb.train(
            params=self.params_best_fit,
            train_set=dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=lgb_auc_lift,
            # adaptative learning rate :
            # learning_rates=lambda iter: 0.5 * (0.999 ** iter),
            **kwargs,
        )

        if save_model:
            self.save_model(booster)

        return booster

    def cv(self,
           params_model=None,
           nfold=5,
           num_boost_round=10000,
           early_stopping_rounds=100,
           generate_plot=False,
           **kwargs):

        dtrain = self.get_train_set(as_lgb_dataset=True)

        # If no params_model is given, take self.params_best_fit
        if params_model is None:
            params_model = self.params_best_fit

        eval_hist = lgb.cv(
            params=params_model,
            train_set=dtrain,
            nfold=nfold,
            verbose_eval=True,  # display the progress
            feval=lgb_auc_lift,
            # display the standard deviation in progress, results are not affected
            show_stdv=True,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )

        if generate_plot:
            self._generate_plot(eval_hist)

        return eval_hist

    def predict(self, from_model_saved, label=None):
        booster = lgb.Booster(model_file=from_model_saved)
        df = self.get_test_set()

        with Timer("Predicting"):
            pred = booster.predict(df)

        if not label:
            now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")
            label = 'pred_{}_{}'.format(self.algo, now)

        return pd.DataFrame({label: pred})

    def fit_predict(self, X_train, y_train, X_pred, **kwargs):
        dtrain = to_gradboost_dataset(X_train, y_train, as_lgb_dataset=True)
        dtest = to_gradboost_dataset(X_pred, as_lgb_dataset=False)

        valid_sets = [dtrain]
        valid_names = ['train']

        params = copy.deepcopy(self.params_best_fit)
        params['random_state'] = self.random_state

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=lgb_auc_lift,
            **kwargs,
        )

        pred = booster.predict(dtest)

        self.booster = booster

        now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")
        label = 'pred_{}_{}'.format(self.algo, now)
        return pd.DataFrame({label: pred})

    def generate_submit(self, num_boost_round=None, from_model_saved=False):

        if not from_model_saved:
            assert num_boost_round is not None

            dtrain = self.get_train_set(as_lgb_dataset=True)

            booster = lgb.train(
                params=self.params_best_fit,
                train_set=dtrain,
                num_boost_round=num_boost_round,
            )

            from_model_saved = self.save_model(booster)

        df = self.predict(from_model_saved)
        now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")
        df.to_csv(
            RESULT_DIR / "submit_{}.csv".format(now),
            index=False,
            header=False)


class XgbCookie(BaseModelCookie):
    algo = 'xgboost'

    # Common params for Xgboost
    common_params = {
        "silent": True,
        "nthreads": 16,
        "objective": "binary:logistic",
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        "scale_pos_weight": 0.97,  # used only in binary application, weight of labels with positive class
        "eval_metric": "auc",
        "tree_method": "hist",
    }
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    params_best_fit = {
        "booster": "gbtree",
        "max_depth": 12,
        "learning_rate": 0.04,
        # "gamma": 0.015,
        # "subsample": max(min(subsample, 1), 0),
        # "colsample_bytree": max(min(colsample_bytree, 1), 0),
        # "min_child_weight": min_child_weight,
        # "max_delta_step": int(max_delta_step),
        **common_params,
    }

    int_params = ()
    float_params = ()
    hypertuning_space = {}

    def validate(self, save_model=True, **kwargs):
        dtrain, dtest = self.get_train_valid_set(as_xgb_dmatrix=True)
        watchlist = [(dtrain, "train"), (dtest, "eval")]

        booster = xgb.train(
            params=self.params_best_fit,
            dtrain=dtrain,
            evals=watchlist,
            feval=xgb_auc_lift,
            **kwargs,
        )

        if save_model:
            self.save_model(booster)
        return booster

    def cv(self,
           params_model=None,
           nfold=5,
           num_boost_round=10000,
           early_stopping_rounds=100,
           generate_plot=False,
           **kwargs):

        # If no params_model is given, take self.params_best_fit
        if params_model is None:
            params_model = self.params_best_fit

        dtrain = self.get_train_set(as_xgb_dmatrix=True)
        eval_hist = xgb.cv(
            params=params_model,
            dtrain=dtrain,
            nfold=nfold,
            feval=xgb_auc_lift,
            verbose_eval=True,
            show_stdv=True,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs)

        if generate_plot:
            self._generate_plot(eval_hist)

        return eval_hist


class CatBoostCookie(BaseModelCookie):
    algo = 'catboost'

    common_params = {
        "thread_count": 15,
        "objective": "Logloss",
        "eval_metric": "AUC",
        "scale_pos_weight": 0.97,  # used only in binary application, weight of labels with positive class
    }

    params_best_fit = {
        # "max_depth": 12,
        "learning_rate": 0.04,
        **common_params,
    }

    int_params = ()
    float_params = ()
    hypertuning_space = {}

    def validate(self, save_model=True, **kwargs):
        dtrain, dtest = self.get_train_valid_set(as_cgb_pool=True)
        watchlist = [dtrain, dtest]

        booster = cgb.train(
            dtrain=dtrain,
            params=self.params_best_fit,
            eval_set=watchlist,
            **kwargs,
        )

        if save_model:
            self.save_model(booster)
        return booster

    def cv(self,
           params_model=None,
           nfold=5,
           num_boost_round=10000,
           early_stopping_rounds=100,
           **kwargs):

        # If no params_model is given, take self.params_best_fit
        if params_model is None:
            params_model = self.params_best_fit

        dtrain = self.get_train_set(as_cgb_pool=True)

        eval_hist = cgb.cv(
            params=params_model,
            dtrain=dtrain,
            nfold=nfold,
            verbose_eval=True,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs)

        return eval_hist

    def generate_submit(self, num_boost_round=None, from_model_saved=False):
        assert num_boost_round is not None

        if not from_model_saved:
            dtrain = self.get_train_set(as_cgb_pool=True)

            booster = cgb.train(
                dtrain=dtrain,
                params=self.params_best_fit,
                num_boost_round=num_boost_round)

            self.save_model(booster)

        else:
            booster = cgb.CatBoost(model_file=from_model_saved)

        dftest = self.get_test_set(as_cgb_pool=True)

        with Timer("Predicting"):
            probas = booster.predict(dftest, prediction_type="Probability")
            dfpred = pd.DataFrame(probas)[[1]]  # Get proba classe one
            dfpred = dfpred.rename(columns={1: 'target'})

        now = pd.Timestamp.now(tz='CET').strftime("%d-%Hh-%Mm")

        fpath = RESULT_DIR / "catboost_submit_{}.csv".format(now)

        with Timer('Storing in {}'.format(fpath)):
            dfpred.to_csv(fpath, index=False)
