import re

import catboost as cgb
import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing

from wax_toolbox.profiling import Timer
from churnchall.constants import (CAT_COLS, CLEANED_DATA_DIR, COLS_DROPPED_RAW,
                                                     DATA_DIR, LOW_IMPORTANCE_FEATURES, NLP_COLS,
                                                     RAW_DATA_DIR, SEED, TIMESTAMP_COLS)

# Feature ing.:


def datetime_features_single(df, col):
    # It is needed to reconvert to datetime, as parquet fail and get sometimes
    # int64 which are epoch, but pandas totally handle it like a boss
    serie = pd.to_datetime(df[col]).dropna()
    df.loc[serie.index, col + "_year"] = serie.dt.year
    df.loc[serie.index, col + "_month"] = serie.dt.month
    df.loc[serie.index, col + "_day"] = serie.dt.day
    df.loc[serie.index, col + "_dayofyear"] = serie.dt.dayofyear
    df.loc[serie.index, col + "_week"] = serie.dt.week
    df.loc[serie.index, col + "_weekday"] = serie.dt.weekday
    df.loc[serie.index, col + "_hour"] = serie.dt.hour

    # Fill na with -1 for integer columns & convert it to signed
    regex = col + "_(year)|(month)|(day)|(dayofyear)|(week)|(weekday)|(hour)"
    lst_cols = df.filter(regex=regex).columns.tolist()
    for col in lst_cols:
        df[col] = df[col].fillna(-1)
        df[col] = pd.to_numeric(df[col], downcast="signed")

    return df


def get_describe_value_counts_cat(dataset='test'):
    dtest = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))
    df = pd.DataFrame()
    for col in CAT_COLS:
        df[col] = dtest[col].value_counts().describe()
    return df


class CleanedDataCookie:
    def __init__(self, debug=False, label_encode=True):
        self.debug = debug
        self.label_encode = label_encode

    @staticmethod
    def _build_features_datetime(df):
        timestamp_cols = list(set(df.columns).intersection(set(TIMESTAMP_COLS)))
        for col in timestamp_cols:
            df = datetime_features_single(df, col)

        # Relative features with ref date as CRE_DATE_GZL
        # (The main date at which the intervention bulletin is created):
        for col in [dt for dt in TIMESTAMP_COLS if dt != 'CRE_DATE_GZL']:
            td_col_name = "bulletin_creation_TD_" + col + "_days"
            df[td_col_name] = (df['CRE_DATE_GZL'] - df[col]).dt.days

        # Some additionnals datetime features
        df['nbdays_duration_of_intervention'] = (
            df['SCHEDULED_END_DATE'] - df['SCHEDULED_START_DATE']).dt.days
        df['nbdays_duration_of_contract'] = (
            df['DATE_FIN'] - df['DATE_DEBUT']).dt.days
        df['nbdays_delta_intervention_contract_start'] = (
            df['CRE_DATE_GZL'] - df['DATE_DEBUT']).dt.days

        # Ratios
        df['ratio_duration_contract_duration_first_interv'] = (
            df['nbdays_duration_of_contract'] /
            df['nbdays_delta_intervention_contract_start'])

        df['ratio_duration_contract_td_install_days'] = (
            df['nbdays_duration_of_contract'] /
            df['bulletin_creation_TD_INSTALL_DATE_days'])

        df['ratio_intervention_contract_start_td_install_days'] = (
            df['nbdays_delta_intervention_contract_start'] /
            df['bulletin_creation_TD_INSTALL_DATE_days'])

        return df

    @staticmethod
    def _build_features_str(df):
        # Some Str cleaning:

        # --> FORMULE:
        # treat 'SECURITE*' & 'SECURITE* 2V' & 'ESSENTIAL CLIENT' as the same
        r = re.compile('SECURITE.*')
        df['FORMULE'] = df['FORMULE'].str.replace(r, 'SECURITE')

        # --> INCIDENT_TYPE_NAME
        # Multi Label Binarize & one hot encoder INCIDENT_TYPE_NAME:
        # i.e. from :            to:
        # Dépannage                 1   0
        # Entretien                 0   1
        # Dépannage+Entretien       1   1

        df['INCIDENT_TYPE_NAME'] = df['INCIDENT_TYPE_NAME'].str.split('+')
        mlb = preprocessing.MultiLabelBinarizer()
        df['INCIDENT_TYPE_NAME'] = list(
            mlb.fit_transform(df['INCIDENT_TYPE_NAME']))
        dftmp = pd.DataFrame(
            index=df['INCIDENT_TYPE_NAME'].index,
            data=df['INCIDENT_TYPE_NAME'].values.tolist()).add_prefix(
                'INCIDENT_TYPE_NAME_label')
        df = pd.concat(
            [df.drop(columns=['INCIDENT_TYPE_NAME']), dftmp], axis=1)

        # --> Deal with Overfitting due to too many different categories:
        with Timer('Replacing rare RUE'):
            to_replace = (df['RUE'].value_counts() <= 10).index
            df['RUE'] = df['RUE'].replace(to_replace, 'KitKatIsGood')

        with Timer('Replacing rare VILLE'):
            to_replace = (df['VILLE'].value_counts() <= 10).index
            df['VILLE'] = df['VILLE'].replace(to_replace, 'WonderLand')

        # TODO Natural Language Processing
        nlp_cols = list(set(df.columns).intersection(set(NLP_COLS)))
        for col in nlp_cols:
            newcol = '{}_len'.format(col)
            df[newcol] = df[col].apply(len)
            df[newcol].replace(1, -1)  # considered as NaN

        df = df.drop(columns=nlp_cols)

        return df

    def _build_features(self, df):
        """Build features."""

        with Timer("Building timestamp features"):
            df = self._build_features_datetime(df)
            # Drop TIMESTAMP_COLS:
            to_drop = list(set(df.columns).intersection(set(TIMESTAMP_COLS)))
            df = df.drop(columns=to_drop)

        with Timer("Building str features"):
            df = self._build_features_str(df)

        cat_cols = list(set(df.columns).intersection(set(CAT_COLS)))
        df.loc[:, cat_cols] = df.loc[:, cat_cols].astype(str)

        if self.hash_encode:
            with Timer("Encoding with HashingEncoder"):
                for col in ['RESOURCE_ID', 'RUE', 'VILLE']:
                    hash_cols = list(set(label_cols).intersection(set([col])))
                    hash_encoder = ce.HashingEncoder(
                        cols=hash_cols, n_components=8, verbose=1)
                    dftmp = hash_encoder.fit_transform(df)
                    newcols = dftmp.columns.difference(df.columns)
                    dftmp = dftmp[newcols]
                    dftmp.columns = 'hash_{}_'.format(col) + dftmp.columns
                    df = pd.concat([df, dftmp], axis=1)

        if self.label_encode:
            # Forgotten columns at the end, simple Label Encoding:
            with Timer("Encoding remaning ones in as LabelEncoder"):
                other_cols = df.columns.difference(
                    df._get_numeric_data().columns).tolist()
                le = preprocessing.LabelEncoder()
                for col in other_cols:
                    df.loc[:, col] = le.fit_transform(df[col])

        return df

    # Methods to generate cleaned datas:
    def generate_single_set(self, dataset, drop_cols=COLS_DROPPED_RAW):
        """Generate one cleaned set amon ['train', 'test']"""
        with Timer("Reading {} set".format(dataset)):
            df = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))

        if self.debug:
            df = df.sample(n=30000, random_state=SEED)

        df = self._build_features(df)

        if drop_cols is not None:
            to_drop = list(set(df.columns).intersection(set(drop_cols)))
            df = df.drop(columns=to_drop)

        savepath = CLEANED_DATA_DIR / "{}{}_cleaned.parquet.gzip".format(
            'debug_' if self.debug else '', dataset)

        with Timer("Saving into {}".format(savepath)):
            df.to_parquet(savepath, compression="gzip")

    def generate_sets(self, drop_cols=COLS_DROPPED_RAW):
        """Generate cleaned sets."""
        with Timer("Gen clean trainset", True):
            self.generate_single_set(dataset="train", drop_cols=drop_cols)

        with Timer("Gen clean testset", True):
            self.generate_single_set(dataset="test", drop_cols=drop_cols)


class DataHandleCookie:

    target = 'target'  # column name of the target

    @property
    def train_parquetpath(self):
        if self.debug:
            return CLEANED_DATA_DIR / "debug_train.parquet.gzip"
        else:
            return CLEANED_DATA_DIR / "train.parquet.gzip"

    @property
    def test_parquetpath(self):
        if self.debug:
            return CLEANED_DATA_DIR / "debug_test.parquet.gzip"
        else:
            return CLEANED_DATA_DIR / "test.parquet.gzip"

    def __init__(self, debug=True, drop_lowimp_features=False):
        self.debug = debug
        self.drop_lowimp_features = drop_lowimp_features

    # Private:
    def _get_cleaned_single_set(self, dataset="train"):
        with Timer("Reading train set"):
            df = pd.read_parquet(self.train_parquetpath)

        # Consistency, always sort columns as follow:
        # | Categorical | Numerical | Target ? |

        cat_cols = set(df.columns).intersection(set(CAT_COLS))
        target_col = set([self.target])
        num_cols = set(df.columns) - cat_cols - target_col

        sorted_cols = sorted(list(cat_cols)) + sorted(list(num_cols))
        sorted_cols += list(target_col) if self.target in df.columns else []

        return df[:, sorted_cols]

    def _process_cleaned_single_set(self, df, as_xgb_dmatrix=False,
                                    as_lgb_dataset=False, as_cgb_pool=False):

        if self.drop_lowimp_features:
            print('Dropping low importance features !')
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

        Xtrain, ytrain = df.drop(columns=[self.target_col]), df[[self.target_col]]

        # Getting indices of categorical columns:
        cat_cols = set(Xtrain.columns).intersection(set(CAT_COLS))
        idx_cat_features = list(range(0, len(cat_cols)))  # as they are already sorted !

        if as_xgb_dmatrix:
            with Timer('Creating DMatrix for Train set Xgboost'):
                return xgb.DMatrix(Xtrain, ytrain)
        elif as_lgb_dataset:
            with Timer('Creating Dataset for Train set LightGBM'):
                return lgb.Dataset(Xtrain, ytrain.values.ravel())
        elif as_cgb_pool:
            with Timer('Creating Pool for Train set CatBoost'):
                pool = cgb.Pool(Xtrain, ytrain, idx_cat_features)
            return pool
        else:
            return Xtrain, ytrain

    # Public:
    def get_test_set(self, as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="test")
        return self._process_cleaned_single_set(
            df, as_xgb_dmatrix=False, as_lgb_dataset=False, as_cgb_pool=as_cgb_pool)

    def get_train_set(self,
                      as_xgb_dmatrix=False,
                      as_lgb_dataset=False,
                      as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="train")
        return self._process_cleaned_single_set(
            df, as_xgb_dmatrix, as_lgb_dataset, as_cgb_pool)

    def get_train_valid_set(self,
                            split_perc=0.2,
                            as_xgb_dmatrix=False,
                            as_lgb_dataset=False,
                            as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="train")

        dftrain = df.sample(frac=1 - split_perc, random_state=SEED)
        dftest = df.index.difference(dftrain.index)
        del df

        dtrain = self._process_cleaned_single_set(
            dftrain, as_xgb_dmatrix, as_lgb_dataset, as_cgb_pool)
        dtest = self._process_cleaned_single_set(
            dtest, as_xgb_dmatrix, as_lgb_dataset, as_cgb_pool)

        return dtrain, dtest
