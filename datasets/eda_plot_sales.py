CSV_PATH = "datasets/REP_S_00134_SMRY_cleaned.csv"  
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def _to_number(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace('"', '')
    if s in ("", "-", "nan", "None"):
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0

def _make_unique(cols):
    counts = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c == "":
            out.append("")
            continue
        counts[c] = counts.get(c, 0) + 1
        out.append(c if counts[c] == 1 else f"{c}_{counts[c]}")
    return out

def _detect_branch_column(df: pd.DataFrame) -> str:
    """
    Pick the column that most looks like branch names:
    it should contain many cells with 'Stories' text.
    """
    best_col = None
    best_score = -1
    for c in df.columns:
        s = df[c].astype(str)
        score = s.str.contains(r"\bStories\b", case=False, na=False).sum()
        if score > best_score:
            best_score = score
            best_col = c
    if best_col is None or best_score == 0:
        raise ValueError("Could not detect the Branch column (no 'Stories' text found).")
    return best_col

def load_comparative_monthly_sales(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, header=None, dtype=str, engine="python")

    months = ["January","February","March","April","May","June","July","August",
              "September","October","November","December"]

    # Find the row that contains the month headers
    header_row = None
    for i in range(len(raw)):
        row_vals = raw.iloc[i].fillna("").astype(str).tolist()
        if all(m in row_vals for m in months):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find the month header row (January..December).")

    cols = _make_unique(raw.iloc[header_row].fillna("").astype(str).tolist())

    df = raw.iloc[header_row + 1:].copy()
    df.columns = cols

    df = df.loc[:, [c for c in df.columns if c != ""]]

    branch_col = _detect_branch_column(df)

    year_col = df.columns[0]

    # clean branch names
    df[branch_col] = df[branch_col].fillna("").astype(str).str.strip()
    df = df[df[branch_col] != ""]

    df = df[~df[branch_col].str.strip().str.lower().eq("total")]

    for c in df.columns:
        if c not in (year_col, branch_col):
            df[c] = df[c].map(_to_number)

    return df.rename(columns={year_col: "Year", branch_col: "Branch"}).reset_index(drop=True)

def dollars(x, pos):
    return f"${x:,.0f}"

df = load_comparative_monthly_sales(CSV_PATH)

nov_col, dec_col = "November", "December"

jan_cols = [c for c in df.columns if c == "January" or c.startswith("January_")]
if not jan_cols:
    raise ValueError("No January column found.")
jan_col = jan_cols[-1]  # last January aka jan 2026

df["Total_NovDecJan"] = df[nov_col] + df[dec_col] + df[jan_col]

plot_df = df[["Branch", "Total_NovDecJan"]].drop_duplicates("Branch")
plot_df = plot_df.sort_values("Total_NovDecJan", ascending=False)

plt.figure(figsize=(max(12, 0.35 * len(plot_df)), 6))
plt.bar(plot_df["Branch"], plot_df["Total_NovDecJan"])
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollars))
plt.ylabel("Total Sales ($) — Nov + Dec + Jan")
plt.title("Branch Total Sales — Nov, Dec, Jan (in $)")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

