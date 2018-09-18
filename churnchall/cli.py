"""Console script for homeserv_inter."""
import fire

from churnchall.raw import convert_csv_to_parquet
from churnchall.datahandler import CleanedDataCookie
from churnchall.boosters import CatBoostCookie, LgbCookie, XgbCookie


def main():
    return fire.Fire({
        "convert-raw": convert_csv_to_parquet,
        "features": CleanedDataCookie,
        "lgb": LgbCookie,
        "xgb": XgbCookie,
        "cgb": CatBoostCookie,
    })
