import numpy as np

import catboost as cgb
import category_encoders as ce
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from churnchall.constants import (CAT_COLS, CLEANED_DATA_DIR, COLS_DROPPED_RAW,
                                  DATA_DIR, INT_COLS, LOW_IMPORTANCE_FEATURES,
                                  NLP_COLS, RAW_DATA_DIR, SEED, STR_COLS,
                                  TIMESTAMP_COLS)
from sklearn import model_selection, preprocessing
from wax_toolbox.profiling import Timer


def raw_convert_csv_to_parquet():
    with Timer('Converting train.csv in train.parquet.gzip'):
        df = pd.read_csv(RAW_DATA_DIR / 'train.csv')
        for col in INT_COLS + ['cible']:
            # As there is no NaN in INT_COLS and no negative values:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')

        # Replace NaN by '' for STR_COLS
        df[STR_COLS] = df[STR_COLS].fillna('')
        df.to_parquet(
            RAW_DATA_DIR / 'train.parquet.gzip',
            compression="gzip",
            engine="pyarrow")

    with Timer('Converting test.csv in test.parquet.gzip'):
        df = pd.read_csv(RAW_DATA_DIR / 'test.csv')
        for col in INT_COLS:
            # As there is no NaN in INT_COLS and no negative values:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')

        # Replace NaN by '' for STR_COLS
        df[STR_COLS] = df[STR_COLS].fillna('')

        df.to_parquet(
            RAW_DATA_DIR / 'test.parquet.gzip',
            compression="gzip",
            engine="pyarrow")


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
        #TODO
        return df

    @staticmethod
    def _build_features_str(df):
        # TODO
        return df

    def _build_features(self, df):
        """Build features."""

        with Timer("Building str features"):
            df = self._build_features_str(df)

        cat_cols = list(set(df.columns).intersection(set(CAT_COLS)))
        df.loc[:, cat_cols] = df.loc[:, cat_cols].astype(str)

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
        """Generate one cleaned set among ['train', 'test']"""
        with Timer("Reading {} set".format(dataset)):
            df = pd.read_parquet(
                RAW_DATA_DIR / "{}.parquet.gzip".format(dataset))

            # Put back NaN values
            df[STR_COLS] = df[STR_COLS].replace('', np.nan)

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


def to_gradboost_dataset(X,
                         y=None,
                         idx_cat_features=None,
                         as_xgb_dmatrix=False,
                         as_lgb_dataset=False,
                         as_cgb_pool=False):
    if y is None:
        if as_xgb_dmatrix:
            with Timer('Creating DMatrix for Xgboost'):
                return xgb.DMatrix(X)
        elif as_lgb_dataset:
            with Timer('Creating Dataset for LightGBM'):
                return lgb.Dataset(X)
        elif as_cgb_pool:
            with Timer('Creating Pool for CatBoost'):
                pool = cgb.Pool(X, None, idx_cat_features)
                return pool
        else:
            return X

    else:
        if as_xgb_dmatrix:
            with Timer('Creating DMatrix for Xgboost'):
                return xgb.DMatrix(X, y)
        elif as_lgb_dataset:
            with Timer('Creating Dataset for LightGBM'):
                return lgb.Dataset(X, y.values.ravel())
        elif as_cgb_pool:
            with Timer('Creating Pool for CatBoost'):
                pool = cgb.Pool(X, y, idx_cat_features)
            return pool
        else:
            return X, y


class DataHandleCookie:

    target_col = 'cible'  # column name of the target

    @property
    def train_parquetpath(self):
        if self.debug:
            return CLEANED_DATA_DIR / "debug_train.parquet.gzip"
        else:
            return CLEANED_DATA_DIR / "train_cleaned.parquet.gzip"

    @property
    def test_parquetpath(self):
        if self.debug:
            return CLEANED_DATA_DIR / "debug_test.parquet.gzip"
        else:
            return CLEANED_DATA_DIR / "test_cleaned.parquet.gzip"

    def __init__(self, debug=False, drop_lowimp_features=False, random_state=1):
        self.debug = debug
        self.drop_lowimp_features = drop_lowimp_features
        self.random_state = random_state

    # Private:
    def _get_cleaned_single_set(self, dataset="train"):
        with Timer("Reading {} set".format(dataset)):
            if dataset == 'train':
                path = self.train_parquetpath
            else:
                path = self.test_parquetpath
            df = pd.read_parquet(path)

        # Consistency, always sort columns as follow:
        # | Categorical | Numerical | Target ? |

        cat_cols = set(df.columns).intersection(set(CAT_COLS))
        target_col = set([self.target_col])
        num_cols = set(df.columns) - cat_cols - target_col

        sorted_cols = sorted(list(cat_cols)) + sorted(list(num_cols))
        sorted_cols += list(
            target_col) if self.target_col in df.columns else []

        return df.loc[:, sorted_cols]

    def _process_cleaned_single_set(self, df, **kwargs):
        if self.drop_lowimp_features:
            print('Dropping low importance features !')
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

        # Getting indices of categorical columns:
        cat_cols = set(df.columns).intersection(set(CAT_COLS))
        idx_cat_features = list(range(
            0, len(cat_cols)))  # as they are already sorted !

        # If test set and no need to split:
        if self.target_col not in df.columns:
            return to_gradboost_dataset(df, None, idx_cat_features, **kwargs)

        else:
            Xtrain = df.drop(columns=[self.target_col])
            ytrain = df[[self.target_col]]
            return to_gradboost_dataset(Xtrain, ytrain, idx_cat_features, **kwargs)

    # Public:
    def get_test_set(self, as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="test")
        return self._process_cleaned_single_set(df, as_cgb_pool=as_cgb_pool)

    def get_train_set(self, **kwargs):
        df = self._get_cleaned_single_set(dataset="train")
        return self._process_cleaned_single_set(df, **kwargs)

    def get_train_valid_set(self, split_perc=0.2, **kwargs):
        df = self._get_cleaned_single_set(dataset="train")

        dftrain = df.sample(frac=1 - split_perc, random_state=self.random_state)
        idxtest = df.index.difference(dftrain.index)
        dftest = df.loc[idxtest, :]
        del df

        dtrain = self._process_cleaned_single_set(dftrain, **kwargs)
        dtest = self._process_cleaned_single_set(dftest, **kwargs)

        return dtrain, dtest
