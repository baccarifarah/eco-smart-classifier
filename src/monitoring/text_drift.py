import pandas as pd
import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import jensenshannon

def main():

    print(" Text drift monitoring started")

    # =====================
    # LOAD DATA
    # =====================

    reference = pd.read_csv(
        "data/processed/df_clean.csv"
    )["Rapport_Collecte"].fillna("")

    current_path = "logs/current_nlp.csv"

    if os.path.exists(current_path):

        current = pd.read_csv(
            current_path
        )["Rapport_Collecte"].fillna("")

    else:

        current = pd.Series([""])

    # =====================
    # VECTORIZE
    # =====================

    vectorizer = TfidfVectorizer(
        max_features=50
    )

    ref_vec = vectorizer.fit_transform(
        reference
    ).toarray().mean(axis=0)

    cur_vec = vectorizer.transform(
        current
    ).toarray().mean(axis=0)

    # =====================
    # DRIFT
    # =====================

    if ref_vec.sum() == 0 or cur_vec.sum() == 0:

        distance = 0.0

    else:

        ref_vec = ref_vec / ref_vec.sum()

        cur_vec = cur_vec / cur_vec.sum()

        distance = jensenshannon(
            ref_vec,
            cur_vec
        )

    # =====================
    # SAVE JSON
    # =====================

    os.makedirs(
        "reports/evidently",
        exist_ok=True
    )

    with open(
        "reports/evidently/text_drift.json",
        "w"
    ) as f:

        json.dump({
            "text_drift_score": float(distance)
        }, f)

    print(" Text Drift:", distance)

    print(" Text monitoring finished")

if __name__ == "__main__":
    main()