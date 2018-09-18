from pathlib import Path

SRC_DIR = Path(__file__).parent
PKG_DIR = SRC_DIR.parent

DATA_DIR = PKG_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"

GENERATED_RESULTS = PKG_DIR / "results"
MODEL_DIR = GENERATED_RESULTS / "model_bank"
TUNING_DIR = GENERATED_RESULTS / "tuning_hist"
RESULT_DIR = GENERATED_RESULTS / "submits"

for d in [DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, MODEL_DIR, TUNING_DIR, RESULT_DIR]:
    if not d.exists():
        d.mkdir(parents=True)

SEED = 42

ALL_COLS = []
FLOAT_COLS = []
INT_COLS = []
STR_COLS = []
TIMESTAMP_COLS = []

NLP_COLS = []

COLS_DROPPED_RAW = []

# Categorical features:
CAT_COLS_NUM = [  # Numeric columns that should be considered categorical
]

CAT_COLS = list(STR_COLS)
for col in NLP_COLS:
    CAT_COLS.remove(col)
CAT_COLS += CAT_COLS_NUM


LOW_IMPORTANCE_FEATURES = []
