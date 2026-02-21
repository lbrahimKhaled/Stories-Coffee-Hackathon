import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
# this linear regression model is linked with the findings of timeseries_plots.py
file_path = "datasets/REP_S_00134_SMRY_cleaned.csv"
df_raw = pd.read_csv(file_path, header=None)

header_row = df_raw[
    df_raw.apply(lambda row: row.astype(str).str.contains("January", na=False).any(), axis=1)
].index[0]

df = pd.read_csv(file_path, header=header_row)
df.columns = df.columns.astype(str).str.strip()


months_2025 = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

jan_2026 = "January.1"

df = df[[col for col in months_2025 + [jan_2026] if col in df.columns]]

# Clean numeric values
for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

df = df.dropna(how="all")

print("Data shape:", df.shape)


branch_means = df[months_2025].replace(0, np.nan).mean(axis=1)

df_norm = df.copy()
df_norm[months_2025] = df[months_2025].div(branch_means, axis=0)
df_norm[jan_2026] = df[jan_2026] / branch_means


train_rows = []
jan2026_rows = []

for i in range(len(df)):
    
    series_real = df.iloc[i][months_2025].values
    series_norm = df_norm.iloc[i][months_2025].values
    jan2026_real = df.iloc[i][jan_2026]
    branch_mean = branch_means.iloc[i]
    
    for t in range(3, len(series_real)):
        
        window_real = series_real[t-3:t+1]
        
        if np.any(window_real == 0):
            continue
        
        lag_1 = series_norm[t-1]
        lag_2 = series_norm[t-2]
        rolling_3 = np.mean(series_norm[t-3:t])
        target = series_norm[t]
        
        train_rows.append([lag_1, lag_2, rolling_3, target])
    
    # January 2026 prediction
    if not np.isnan(jan2026_real):
        
        last_3_real = series_real[-3:]
        
        if np.any(last_3_real == 0):
            continue
        
        lag_1 = series_norm[-1]
        lag_2 = series_norm[-2]
        rolling_3 = np.mean(series_norm[-3:])
        
        jan2026_rows.append([
            lag_1, lag_2, rolling_3,
            jan2026_real, branch_mean
        ])

train_df = pd.DataFrame(train_rows,
                        columns=["lag_1","lag_2","rolling_3","target"])

jan2026_df = pd.DataFrame(jan2026_rows,
                          columns=["lag_1","lag_2","rolling_3",
                                   "actual_sales","branch_mean"])

print("Training samples:", len(train_df))
print("January 2026 samples:", len(jan2026_df))


X_train = train_df[["lag_1","lag_2","rolling_3"]]
y_train = train_df["target"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

X_jan = jan2026_df[["lag_1","lag_2","rolling_3"]]
X_jan_scaled = scaler.transform(X_jan)

predicted_norm = model.predict(X_jan_scaled)

predicted_sales = predicted_norm * jan2026_df["branch_mean"].values
actual_sales = jan2026_df["actual_sales"].values


mae = mean_absolute_error(actual_sales, predicted_sales)
r2 = r2_score(actual_sales, predicted_sales)
mape = np.mean(np.abs((actual_sales - predicted_sales) / actual_sales)) * 100

print("\nJanuary 2026 Prediction Performance (Linear Regression):")
print("MAE:", round(mae,2))
print("R²:", round(r2,4))
print("MAPE (%):", round(mape,2))

coef = pd.Series(model.coef_, index=["lag_1","lag_2","rolling_3"])
print("\nCoefficients:")
print(coef.sort_values(ascending=False))