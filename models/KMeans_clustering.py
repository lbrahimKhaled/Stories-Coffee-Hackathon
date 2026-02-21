import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


'''We used KMeans clustering to identify structural patterns in how branches behave when they open. Instead of predicting sales, this model groups branches based on similarities in their early performance 
 specifically how long they stayed at zero before opening, their average sales in the first three active months,
   their volatility during that period, and their short-term growth. The algorithm does not use labels or targets; it simply detects natural groupings in the data. 
From this, we discovered that branches fall into three distinct categories: mature branches that were already active and stable, mid-year openings that ramp up aggressively with high volatility, and late-year openings that grow more slowly and steadily.
 This shows that branch dynamics are not uniform and that opening timing strongly influences early performance behavior.(WE Will dive deeper later about teh location and studf like that )'''
file_path = "datasets/REP_S_00134_SMRY_cleaned.csv"
df_raw = pd.read_csv(file_path, header=None)

header_row = df_raw[
    df_raw.apply(lambda row: row.astype(str).str.contains("January", na=False).any(), axis=1)
].index[0]

df = pd.read_csv(file_path, header=header_row)
df.columns = df.columns.astype(str).str.strip()

months_2025 = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

df = df[[col for col in months_2025 if col in df.columns]]


for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

df = df.fillna(0)

print("Data shape:", df.shape)


opening_features = []

for i in range(len(df)):
    
    series = df.iloc[i].values
    
    # Count initial consecutive zeros
    initial_zeros = 0
    for value in series:
        if value == 0:
            initial_zeros += 1
        else:
            break
   
    if initial_zeros >= len(series):
        continue
    
    opening_month = initial_zeros
   
    if opening_month + 3 > len(series):
        continue
    
    first3 = series[opening_month:opening_month+3]
    
    # Skip if any zero inside first 3 months
    if np.any(first3 == 0):
        continue
    
    avg_first3 = np.mean(first3)
    vol_first3 = np.std(first3)
    growth_first3 = first3[-1] - first3[0]
    
    opening_features.append([
        initial_zeros,
        avg_first3,
        vol_first3,
        growth_first3
    ])

opening_df = pd.DataFrame(
    opening_features,
    columns=["initial_zeros","avg_first3","vol_first3","growth_first3"]
)

print("Branches analyzed:", len(opening_df))


opening_df = opening_df.replace([np.inf, -np.inf], np.nan)
opening_df = opening_df.dropna()

print("Branches after cleaning:", len(opening_df))

# =====================================================
# CLUSTERING
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(opening_df)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

opening_df["cluster"] = clusters

# =====================================================
# CLUSTER SUMMARY
# =====================================================
print("\nCluster Summary (Mean Values):")
print(opening_df.groupby("cluster").mean())
import matplotlib.pyplot as plt

# =====================================================
# PLOT AVERAGE OPENING TRAJECTORY PER CLUSTER
# =====================================================

# Rebuild trajectories per branch
cluster_trajectories = {0: [], 1: [], 2: []}

for i in range(len(df)):
    
    series = df.iloc[i].values
    
    # Count initial zeros
    initial_zeros = 0
    for value in series:
        if value == 0:
            initial_zeros += 1
        else:
            break
    
    if initial_zeros >= len(series):
        continue
    
    if initial_zeros + 6 > len(series):
        continue
    
    first6 = series[initial_zeros:initial_zeros+6]
    
    if np.any(first6 == 0):
        continue
    
    # Normalize trajectory by first month (to compare shapes)
    first6_norm = first6 / first6[0]
    
    # Assign cluster (match by feature row index)
    if len(cluster_trajectories[0]) + len(cluster_trajectories[1]) + len(cluster_trajectories[2]) < len(opening_df):
        cluster_id = opening_df.iloc[
            len(cluster_trajectories[0]) + len(cluster_trajectories[1]) + len(cluster_trajectories[2])
        ]["cluster"]
        
        cluster_trajectories[cluster_id].append(first6_norm)

# Plot
plt.figure(figsize=(8,5))

for cluster_id in cluster_trajectories:
    if len(cluster_trajectories[cluster_id]) > 0:
        avg_traj = np.mean(cluster_trajectories[cluster_id], axis=0)
        plt.plot(range(1,7), avg_traj, label=f"Cluster {cluster_id}")

plt.xlabel("Months Since Opening")
plt.ylabel("Relative Sales (Normalized)")
plt.title("Average Opening Trajectory per Cluster")
plt.legend()
plt.tight_layout()
plt.show()