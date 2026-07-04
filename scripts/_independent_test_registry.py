"""
Independent Test Set Registry

Maps training datasets to independent test data sources.
Three tiers of sources:
  1. Exact match    — same dataset from different canonical source (UCI vs OpenML)
  2. Cross-domain   — different dataset with similar task
  3. Synthetic fallback — generated data mimicking training distribution

Each entry maps a dataset filename key to a list of independent test sources.
"""
import os

# Base path for datasets
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads', 'datasets')

# ---------------------------------------------------------------------------
# Tier 1: Exact schema match — alternative canonical sources
# ---------------------------------------------------------------------------
EXACT_MATCH_SOURCES = {
    # Car Evaluation: OpenML id=21 → UCI original
    'real_car_eval': {
        'name': 'Car Evaluation (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip',
        'target': 'class',
        'note': 'UCI canonical source — exact same feature schema as OpenML version',
    },
    # Mushroom Toxicity: OpenML id=24 → UCI secondary
    'real_mushroom': {
        'name': 'Mushroom Toxicity (UCI secondary)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/73/mushroom.zip',
        'target': 'class',
        'note': 'UCI secondary mushroom dataset — slight encoding differences',
    },
    # Adult/Census Income: OpenML → UCI
    'real_adult': {
        'name': 'Adult Census Income (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
        'target': 'class',
        'note': 'UCI canonical source — exact same schema',
    },
    # Spambase: OpenML → UCI
    'real_spambase': {
        'name': 'Spambase (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/94/spambase.zip',
        'target': 'class',
        'note': 'UCI canonical source',
    },
    # Wine Quality Red: UCI direct
    'real_wine_quality_red': {
        'name': 'Wine Quality Red (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip',
        'target': 'quality',
        'note': 'UCI canonical source — full dataset including both red and white',
    },
    # Bank Marketing: UCI direct
    'real_bank_marketing': {
        'name': 'Bank Marketing (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
        'target': 'y',
        'note': 'UCI canonical source',
    },
    # Boston Housing: UCI original (deprecated but still available)
    'real_boston_housing': {
        'name': 'Boston Housing (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/1/housing.zip',
        'target': 'MEDV',
        'note': 'UCI canonical source',
    },
    # Auto MPG: UCI direct
    'real_auto_mpg': {
        'name': 'Auto MPG (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/9/auto+mpg.zip',
        'target': 'mpg',
        'note': 'UCI canonical source',
    },
    # Glass Identification: UCI direct
    'real_glass': {
        'name': 'Glass Identification (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/42/glass+identification.zip',
        'target': 'Type',
        'note': 'UCI canonical source',
    },
    # Ecoli: UCI direct
    'real_ecoli': {
        'name': 'Ecoli (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/39/ecoli.zip',
        'target': 'class',
        'note': 'UCI canonical source',
    },
    # Seeds: UCI direct
    'real_seeds': {
        'name': 'Wheat Seeds (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/236/seeds.zip',
        'target': 'class',
        'note': 'UCI canonical source',
    },
    # Heart Disease: UCI direct
    'real_heart_disease': {
        'name': 'Heart Disease (UCI Cleveland)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip',
        'target': 'num',
        'note': 'UCI Cleveland heart disease dataset',
    },
    # Air Quality: UCI direct
    'real_air_quality': {
        'name': 'Air Quality (UCI original)',
        'source': 'url',
        'url': 'https://archive.ics.uci.edu/static/public/360/air+quality.zip',
        'target': 'target',
        'note': 'UCI canonical source — full dataset',
    },
}

# ---------------------------------------------------------------------------
# Tier 2: Cross-domain — different dataset, similar task
# ---------------------------------------------------------------------------
CROSS_DOMAIN_SOURCES = {
    # Wine Variety (sklearn 178 rows) → Wine Quality Red (1599 rows)
    'real_wine_sklearn': {
        'name': 'Wine Quality Red (cross-domain)',
        'source': 'local',
        'local_key': 'real_wine_quality_red',
        'note': 'Different wine dataset — tests generalization to different wine features',
    },
    # Breast Cancer Wisconsin → Breast Cancer Diagnosis (biology)
    'real_breast_wisconsin': {
        'name': 'Breast Cancer Diagnosis (cross-domain)',
        'source': 'local',
        'local_key': 'breast_cancer_diagnosis_biology',
        'note': 'Different breast cancer dataset — UCI vs Wisconsin features',
    },
    # WDBC → Wisconsin Diagnostic → Biology set
    'real_wdbc': {
        'name': 'Breast Cancer Wisconsin Orig (cross-domain)',
        'source': 'local',
        'local_key': 'real_breast_wisconsin',
        'note': 'WDBC vs Wisconsin Original — feature overlap test',
    },
    # Wine Quality Red ↔ Wine Quality Binary
    'real_wine_quality_binary': {
        'name': 'Wine Quality Red (cross-domain)',
        'source': 'local',
        'local_key': 'real_wine_quality_red',
        'note': 'Regression vs binary classification of same domain',
    },
    # Pima Diabetes → Diabetes Regression
    'real_pima_diabetes': {
        'name': 'Diabetes Regression (cross-domain)',
        'source': 'local',
        'local_key': 'real_diabetes_regression',
        'note': 'Classification vs regression — same domain',
    },
}

# ---------------------------------------------------------------------------
# Tier 3: NLP datasets — GitHub Chinese NLP corpus mirrors
# ---------------------------------------------------------------------------
NLP_SOURCES = {
    'chnsenticorp_hotel': {
        'name': 'Weibo Sentiment 100k (cross-domain NLP)',
        'source': 'url',
        'urls': [
            'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/weibo_senti_100k/weibo_senti_100k.csv',
        ],
        'target': 'label',
        'note': 'Weibo social media reviews — different domain from hotel reviews',
    },
    'douban_reviews': {
        'name': 'Douban Movie Reviews v2 (GitHub mirror)',
        'source': 'url',
        'urls': [
            'https://raw.githubusercontent.com/hecongqing/ChineseNlpCorpus/master/datasets/douban_movie_reviews/douban_reviews.csv',
        ],
        'target': 'sentiment',
        'note': 'Public GitHub mirror of Douban reviews',
    },
    'shopping_reviews': {
        'name': 'Online Shopping 10 Cats (GitHub mirror)',
        'source': 'url',
        'urls': [
            'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/online_shopping_10_cats/online_shopping_10_cats.csv',
        ],
        'target': 'label',
        'note': 'Full 10-category shopping reviews from public mirror',
    },
}

# ---------------------------------------------------------------------------
# Helper: determine if a dataset has an independent test source
# ---------------------------------------------------------------------------

def get_independent_source(filename: str) -> dict | None:
    """
    Look up the independent test source for a dataset by its filename.

    Returns:
        dict with keys: tier, name, source, url/urls, target, note
        or None if no known source exists (use synthetic fallback)
    """
    basename = filename.replace('.csv', '') if filename.endswith('.csv') else filename

    # Tier 1: Exact match
    if basename in EXACT_MATCH_SOURCES:
        return {'tier': 1, **EXACT_MATCH_SOURCES[basename]}

    # Tier 2: Cross-domain
    if basename in CROSS_DOMAIN_SOURCES:
        return {'tier': 2, **CROSS_DOMAIN_SOURCES[basename]}

    # Tier 3: NLP
    if basename in NLP_SOURCES:
        return {'tier': 3, **NLP_SOURCES[basename]}

    # Check partial matches for NLP
    for key, source in NLP_SOURCES.items():
        if key in basename or basename in key:
            return {'tier': 3, **source}

    # No known source
    return None
