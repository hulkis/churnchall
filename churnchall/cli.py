"""Console script for homeserv_inter."""
import fire

from churnchall.datahandler import CleanedDataCookie, raw_convert_csv_to_parquet
from churnchall.boosters import CatBoostCookie, LgbCookie, XgbCookie


def main():
    return fire.Fire({
        "convert-raw": raw_convert_csv_to_parquet,
        "features": CleanedDataCookie,
        "lgb": LgbCookie,
        "xgb": XgbCookie,
        "cgb": CatBoostCookie,
    })
