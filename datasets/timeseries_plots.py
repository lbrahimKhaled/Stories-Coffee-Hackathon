import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


''''We wanted to check whether the last three months actually relate to the next month before using them for prediction. After cleaning the data and normalizing each branch to remove size effects, we computed rolling three-month averages and measured their correlation with the following month. 
We found strong positive correlations in some parts of the year, showing short-term persistence, but also strong negative correlations around mid-year, indicating seasonal shifts. 
This means recent months do contain useful information, but the relationship changes across seasons rather than staying consistent all year.'''
# now we know that there is a correlation between las three months and the one after , we will use a model to predict february sales . This was just the analysis and making sure that the model we chose and the features are a good option.



file_path = "datasets/REP_S_00134_SMRY_cleaned.csv"   
df_raw = pd.read_csv(file_path, header=None)

header_row = df_raw[
    df_raw.apply(lambda row: row.astype(str).str.contains("January", na=False).any(), axis=1)
].index[0]

df = pd.read_csv(file_path, header=header_row)
df.columns = df.columns.astype(str).str.strip()


months = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

df = df[[m for m in months if m in df.columns]]

for m in df.columns:
    df[m] = (
        df[m]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("0", np.nan)
        .astype(float)
    )

df = df.dropna(how="all")

df_norm = df.div(df.mean(axis=1), axis=0)


rolling3 = df_norm.T.rolling(window=3).mean().T

results = []

for i in range(3, len(months)):
    prev3 = rolling3.iloc[:, i-1]
    next_month = df_norm.iloc[:, i]

    mask = (~prev3.isna()) & (~next_month.isna())

    corr, pval = pearsonr(prev3[mask], next_month[mask])

    results.append({
        "Previous Months": f"{months[i-3]}-{months[i-1]}",
        "Predicting": months[i],
        "Correlation": corr,
        "p-value": pval
    })

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10,8))
sns.heatmap(df_norm.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Size-Normalized Data)")
plt.tight_layout()
plt.show()
df_growth = df_norm.pct_change(axis=1)

