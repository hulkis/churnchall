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

ALL_COLS = ['Echéance.Année',
            'Echéance.Mois',
            'Produit',
            'Durée',
            'Type.d.offre',
            'Type.de.prix',
            'Canal.de.vente',
            'Zone',
            'Marché.de.la.SC',
            'Segment.société.contractante',
            'Entité.société.contractante',
            'Profil.PRM',
            'Couleur.Tarif.Elec',
            'Volume_annuel',
            'Prix_Gaz_M3',
            'Prix_Elec_M3',
            'nb_dem_12',
            'nb_recla_12',
            'nb_dem_reco_12',
            'nb_recla_reco_12',
            'type_client',
            'libellé_NAF',
            'activité_NCE',
            'ancienneté_client',
            'Orientation Economique',
            'Population',
            'Evolution Pop %',
            'Nb propriétaire',
            'Nb Logement',
            'Dep Moyenne Salaires Horaires',
            'Urbanité Ruralité',
            'Nb Atifs',
            'Dynamique Démographique BV',
            'Environnement Démographique',
            'Seg Dyn Entre']

FLOAT_COLS = ['Volume_annuel',
              'Prix_Gaz_M3',
              'Prix_Elec_M3',
              'ancienneté_client',
              'Population',
              'Evolution Pop %',
              'Nb propriétaire',
              'Nb Logement',
              'Dep Moyenne Salaires Horaires',
              'Nb Atifs']

INT_COLS = ['Echéance.Année',
            'Echéance.Mois',
            'Durée',
            'nb_dem_12',
            'nb_recla_12',
            'nb_dem_reco_12',
            'nb_recla_reco_12']

STR_COLS = ['Produit',
            'Type.d.offre',
            'Type.de.prix',
            'Canal.de.vente',
            'Zone',
            'Marché.de.la.SC',
            'Segment.société.contractante',
            'Entité.société.contractante',
            'Profil.PRM',
            'Couleur.Tarif.Elec',
            'type_client',
            'libellé_NAF',
            'activité_NCE',
            'Orientation Economique',
            'Urbanité Ruralité',
            'Dynamique Démographique BV',
            'Environnement Démographique',
            'Seg Dyn Entre']

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
