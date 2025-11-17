import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
import os

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "MovieDB.xlsx")  # <-- Excel file
output_dir = os.path.join(base_dir, "analysis_results_Q3")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------
# 1. LOAD DATA (Excel)
# ----------------------------------------------------------
# Attempts to load the first sheet. If your data is in a different sheet name,
# update sheet_name param.
try:
    df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
except Exception as e:
    raise SystemExit(f"Failed to read {file_path}: {e}")

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Columns list:", df.columns.tolist())

# ----------------------------------------------------------
# 2. QUICK CLEAN / NORMALIZATION
# ----------------------------------------------------------
# Strip column names (in case of leading/trailing spaces)
df.columns = df.columns.str.strip()

# Normalize movie names
if "Movie Name" in df.columns:
    df["Movie Name"] = df["Movie Name"].astype(str).str.strip().str.title()
if "Scraped Name" in df.columns:
    df["Scraped Name"] = df["Scraped Name"].astype(str).str.strip().str.title()

# Convert Date (year) to numeric if possible
if "Date" in df.columns:
    df["Date"] = pd.to_numeric(df["Date"], errors="coerce").astype("Int64")

# Convert numeric-like columns
for col in [
    "Budget",
    "Rating",
    "DirectorsRating",
    "WritersRating",
    "TotalFollowers",
    "Revenue",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------------
# 3. PARSE LIST-LIKE COLUMNS (Director, Writer, Actor)
# ----------------------------------------------------------
list_columns = [c for c in ["Director", "Writer", "Actor"] if c in df.columns]


def safe_literal_eval(val):
    if isinstance(val, str):
        val = val.strip()
        # sometimes python repr lists; sometimes single-name strings
        if val.startswith("[") and val.endswith("]"):
            try:
                return ast.literal_eval(val)
            except:
                pass
        # if looks like "['A','B']" or "['A']" handled above; otherwise attempt split on commas
        # remove surrounding brackets if present
        val_clean = re.sub(r"^\[|\]$", "", val)
        # split by comma only if commas present
        if "," in val_clean:
            parts = [p.strip().strip("'\"") for p in val_clean.split(",") if p.strip()]
            return parts
        # single name -> return single element list
        if val_clean != "":
            return [val_clean.strip("'\"")]
    return []


for col in list_columns:
    df[col] = df[col].apply(safe_literal_eval)


# ----------------------------------------------------------
# 4. PARSE "OtherInfo" into MPAA + RUNTIME (if present)
# ----------------------------------------------------------
def extract_otherinfo_fields(info):
    if not isinstance(info, str):
        return pd.Series({"MPAA": pd.NA, "RuntimeMin": pd.NA})
    parts = [p.strip() for p in re.split(r"[\r\n]+", info) if p.strip()]
    mpaa = None
    runtime_min = pd.NA
    # typical layouts: [year, MPAA, runtime] or [MPAA, runtime] etc.
    # find MPAA token that is <= 5 chars and contains letters and/or hyphen (e.g. "PG-13", "Not Rated")
    for p in parts:
        if re.match(r"^[A-Za-z\- ]{1,10}$", p):
            mpaa = p
            continue
    # find runtime like "1h 31m" or "90 min"
    for p in parts:
        m = re.search(
            r"(?:(\d+)\s*h(?:ou?r)?s?)\s*(?:(\d+)\s*m(?:in)?)?", p, flags=re.I
        )
        if m:
            h = int(m.group(1))
            mm = int(m.group(2)) if m.group(2) else 0
            runtime_min = h * 60 + mm
            break
        m2 = re.search(r"(\d+)\s*min", p, flags=re.I)
        if m2:
            runtime_min = int(m2.group(1))
            break
    return pd.Series({"MPAA": mpaa, "RuntimeMin": runtime_min})


if "OtherInfo" in df.columns:
    other_parsed = df["OtherInfo"].apply(extract_otherinfo_fields)
    df = pd.concat([df, other_parsed], axis=1)

# ----------------------------------------------------------
# 5. DERIVED FEATURES
# ----------------------------------------------------------
# Critic composite (simple mean of directors/writers ratings when available)
if {"DirectorsRating", "WritersRating"}.issubset(df.columns):
    df["CriticComposite"] = df[["DirectorsRating", "WritersRating"]].mean(axis=1)

# Log-transform followers to reduce skew
if "TotalFollowers" in df.columns:
    df["LogFollowers"] = np.log1p(df["TotalFollowers"])

# If Revenue exists, make log revenue
if "Revenue" in df.columns:
    df["LogRevenue"] = np.log1p(df["Revenue"].fillna(0))

# Placeholder Genre column (if not present). If you do have genre info elsewhere, map it here.
if "Genre" not in df.columns:
    df["Genre"] = pd.NA

# ----------------------------------------------------------
# 6. BASIC EXPLORATION & CORRELATION
# ----------------------------------------------------------
corr_cols = []
for c in [
    "Rating",
    "CriticComposite",
    "DirectorsRating",
    "WritersRating",
    "LogFollowers",
    "LogRevenue",
]:
    if c in df.columns:
        corr_cols.append(c)

if corr_cols:
    corr_df = df[corr_cols].corr()
    print("\nCorrelation matrix:")
    print(corr_df)

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", square=True)
    plt.title("Correlation: Audience Rating vs Critics / Popularity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.show()
else:
    print("No numeric columns available for correlation analysis.")

# ----------------------------------------------------------
# 7. HIGHEST-RATED GENRES (if genre data exists)
# ----------------------------------------------------------
if df["Genre"].notna().sum() > 0:
    def genre_to_list(g):
        if isinstance(g, str):
            if "|" in g:
                return [x.strip() for x in g.split("|") if x.strip()]
            if "," in g:
                return [x.strip() for x in g.split(",") if x.strip()]
            return [g.strip()]
        return []

    df["GenreList"] = df["Genre"].apply(genre_to_list)
    genre_ratings = (
        df.explode("GenreList")
        .groupby("GenreList")["Rating"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nTop genres by average rating:")
    print(genre_ratings.head(20))
    genre_ratings.head(20).to_csv(os.path.join(output_dir, "top_genres_by_rating.csv"))
else:
    print("No genre data present — skipping genre ranking.")


if "Director" in df.columns:
    director_mean = (
        df.explode("Director")
        .dropna(subset=["Director"])
        .groupby("Director")["Rating"]
        .agg(["mean", "count"])
    )
    director_mean = director_mean[director_mean["count"] >= 2].sort_values(
        "mean", ascending=False
    )
    print("\nTop directors (min 2 movies):")
    print(director_mean.head(20))
    director_mean.to_csv(os.path.join(output_dir, "director_ratings.csv"))

if "Writer" in df.columns:
    writer_mean = (
        df.explode("Writer")
        .dropna(subset=["Writer"])
        .groupby("Writer")["Rating"]
        .agg(["mean", "count"])
    )
    writer_mean = writer_mean[writer_mean["count"] >= 2].sort_values(
        "mean", ascending=False
    )
    print("\nTop writers (min 2 movies):")
    print(writer_mean.head(20))
    writer_mean.to_csv(os.path.join(output_dir, "writer_ratings.csv"))

if "Date" in df.columns:
    yearly = df.dropna(subset=["Date"]).groupby("Date")["Rating"].mean().sort_index()
    if not yearly.empty:
        plt.figure(figsize=(10, 5))
        yearly.plot(marker="o")
        plt.title("Average Audience Rating by Release Year")
        plt.xlabel("Year")
        plt.ylabel("Avg Rating")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rating_by_year.png"))
        plt.show()
        yearly.to_csv(os.path.join(output_dir, "rating_by_year.csv"))
    else:
        print("No valid year data for trend plot.")
else:
    print("No Date column present — skipping year trend.")

if "Rating" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Rating"].dropna(), bins=20)
    plt.title("Distribution of Audience Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rating_distribution.png"))
    plt.show()

if {"CriticComposite", "Rating"}.issubset(df.columns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df["CriticComposite"], y=df["Rating"], alpha=0.6)
    plt.title("Audience Rating vs Critic Composite")
    plt.xlabel("Critic Composite")
    plt.ylabel("Audience Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "critic_vs_audience_scatter.png"))
    plt.show()

clean_path = os.path.join(output_dir, "movie_db_cleaned.csv")
df.to_csv(clean_path, index=False)
print(f"\nCleaned/enriched data saved to: {clean_path}")

print("\nQ3 exploration complete. Results, plots and CSVs are in:", output_dir)
