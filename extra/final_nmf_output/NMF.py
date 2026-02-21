import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# =============================
# CONFIG
# =============================

INPUT_FILE = "datasets/branch_profile_group_for_nmf.csv"
OUTPUT_DIR = "datasets/nmf_results"
K_RANGE = range(2, 11)   # Test K from 2 to 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Load Data
# =============================

df = pd.read_csv(INPUT_FILE)
branches = df["Branch"].copy()

# Remove Branch column
X = df.drop(columns=["Branch"]).values

# Optional: ensure rows sum to 1 (already should)
X = normalize(X, norm='l1')

# Store metrics
metrics = []

# =============================
# Run NMF for different K
# =============================

for k in K_RANGE:
    print(f"Running NMF with K = {k}")

    model = NMF(
        n_components=k,
        init='nndsvda',
        random_state=42,
        max_iter=1000
    )

    W = model.fit_transform(X)
    H = model.components_

    # Reconstruction
    X_hat = np.dot(W, H)
    frob_error = np.linalg.norm(X - X_hat, 'fro')

    pseudo_r2 = 1 - (np.linalg.norm(X - X_hat, 'fro')**2 /
                     np.linalg.norm(X, 'fro')**2)

    metrics.append({
        "K": k,
        "Reconstruction_Error": frob_error,
        "Pseudo_R2": pseudo_r2
    })

    # Save Pattern Definitions (H matrix)
    H_df = pd.DataFrame(
        H,
        columns=df.columns[1:]
    )
    H_df.to_csv(f"{OUTPUT_DIR}/patterns_K{k}.csv", index=False)

    # Save Branch Pattern Weights (W matrix)
    W_df = pd.DataFrame(W)
    W_df.insert(0, "Branch", branches)
    W_df.to_csv(f"{OUTPUT_DIR}/branch_weights_K{k}.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{OUTPUT_DIR}/nmf_metrics.csv", index=False)

print("\nDone.")
print("Check datasets/nmf_results/")