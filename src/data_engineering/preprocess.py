import os

import pandas as pd

print("🚀 PREPROCESSING STARTED")

# =====================
# LOAD RAW DATA
# =====================

df = pd.read_csv("data/eco-system.csv")

# =====================
# CLEANING
# =====================

df = df.drop_duplicates()

df = df.fillna(0)

# =====================
# CREATE OUTPUT FOLDER
# =====================

os.makedirs("data/processed", exist_ok=True)

# =====================
# SAVE CLEAN DATA
# =====================

df.to_csv("data/processed/df_clean.csv", index=False)

print("✅ CLEAN DATA SAVED")
