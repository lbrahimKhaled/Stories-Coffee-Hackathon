import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.stats import entropy

# ============================
# CONFIG
# ============================

INPUT_FILE = "datasets/branch_profile_group_for_nmf.csv"
OUTPUT_DIR = "datasets/final_nmf_output"
K = 6   # <-- change manually (try 5–7)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Load Data
# ============================

df = pd.read_csv(INPUT_FILE)
branches = df["Branch"]
X = df.drop(columns=["Branch"]).values
feature_names = df.columns[1:]

# ============================
# Fit NMF
# ============================

model = NMF(
    n_components=K,
    init='nndsvda',
    random_state=42,
    max_iter=1000
)

W = model.fit_transform(X)
H = model.components_

# ============================
# 1) Pattern Definitions
# ============================

pattern_definitions = []

for i in range(K):
    weights = H[i]
    top_idx = np.argsort(weights)[::-1][:7]
    top_features = [feature_names[j] for j in top_idx]

    pattern_definitions.append({
        "Pattern": f"Pattern_{i}",
        "Top_7_Groups": ", ".join(top_features)
    })

pattern_df = pd.DataFrame(pattern_definitions)
pattern_df.to_csv(f"{OUTPUT_DIR}/pattern_definitions.csv", index=False)

# ============================
# 2) Branch Dominant Pattern
# ============================

dominant_pattern = np.argmax(W, axis=1)
dominant_weight = np.max(W, axis=1)

branch_summary = pd.DataFrame({
    "Branch": branches,
    "Dominant_Pattern": [f"Pattern_{i}" for i in dominant_pattern],
    "Dominant_Weight": dominant_weight
})

branch_summary.to_csv(f"{OUTPUT_DIR}/branch_dominant_patterns.csv", index=False)

# ============================
# 3) Branch Diversification
# ============================

entropy_values = []

for row in W:
    p = row / row.sum()
    entropy_values.append(entropy(p))

branch_summary["Entropy"] = entropy_values

branch_summary.to_csv(f"{OUTPUT_DIR}/branch_with_entropy.csv", index=False)

# ============================
# 4) Rank Branches Per Pattern
# ============================

ranking_list = []

for k in range(K):
    pattern_weights = W[:, k]
    sorted_idx = np.argsort(pattern_weights)[::-1]

    for rank, idx in enumerate(sorted_idx[:5]):
        ranking_list.append({
            "Pattern": f"Pattern_{k}",
            "Rank": rank + 1,
            "Branch": branches.iloc[idx],
            "Weight": pattern_weights[idx]
        })

ranking_df = pd.DataFrame(ranking_list)
ranking_df.to_csv(f"{OUTPUT_DIR}/top_branches_per_pattern.csv", index=False)

print("NMF analysis complete.")
print("Check:", OUTPUT_DIR)