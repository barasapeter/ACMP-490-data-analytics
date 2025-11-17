"""
crime_analysis.py

- Input: crimes.csv
- Outputs:
  - top_areas.csv (top areas by crime count)
  - monthly_trend.csv, daily_trend.csv, yearly_counts.csv
  - plots: top_areas.png, monthly_trend.png, yearly_counts.png
  - crime_map.html (if lat/lon columns found)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Optional mapping libs
try:
    import folium
    from folium.plugins import HeatMap

    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# --------------------------
# User settings
# --------------------------
CSV_PATH = "crimes.csv"  # path to your CSV
DATE_COL_CANDIDATES = [
    "DATE OCC",
    "Date Occ",
    "DATE_OCC",
    "DATE_OCCURRENCE",
    "Date Rptd",
    "Date Rpted",
]
AREA_COL_CANDIDATES = ["AREA NAME", "Area Name", "AREA", "AREA_NAME"]
CRIME_TYPE_COL_CANDIDATES = [
    "Crm Cd Desc",
    "CRM_CD_DESC",
    "CrmCdDesc",
    "Crm Description",
    "Crm Desc",
]
LAT_CANDIDATES = ["LAT", "Latitude", "Y", "LATITUDE"]
LON_CANDIDATES = ["LON", "Longitude", "X", "LONGITUDE"]

CHUNKSIZE = 200000  # adjust based on your RAM: larger = faster
SAMPLE_FOR_MAP = (
    5000  # how many points to sample for map visualization (if lat/lon exist)
)
OUTPUT_DIR = "crime_analysis_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------
# Helper: find best column name from candidates
# --------------------------
def find_col(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    # try case-insensitive match
    lower = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# --------------------------
# 1) detect columns by reading header only (fast)
# --------------------------
df_head = pd.read_csv(CSV_PATH, nrows=5)
cols = df_head.columns.tolist()

date_col = find_col(cols, DATE_COL_CANDIDATES)
area_col = find_col(cols, AREA_COL_CANDIDATES)
crime_col = find_col(cols, CRIME_TYPE_COL_CANDIDATES)
lat_col = find_col(cols, LAT_CANDIDATES)
lon_col = find_col(cols, LON_CANDIDATES)

print("Detected columns:")
print(" date_col:", date_col)
print(" area_col:", area_col)
print(" crime_col:", crime_col)
print(" lat_col:", lat_col)
print(" lon_col:", lon_col)
print("folium installed:", FOLIUM_AVAILABLE)

# If date col not found, try to find any column that looks like a date
if date_col is None:
    for c in cols:
        if "date" in c.lower() or "dt" in c.lower():
            date_col = c
            break

# --------------------------
# 2) Chunked aggregation: area counts, monthly counts, daily counts, year counts
# --------------------------
area_counts = {}
monthly_counts = {}  # key: YYYY-MM
daily_counts = {}  # key: YYYY-MM-DD
year_counts = {}

parse_dates = [date_col] if date_col else None

reader = pd.read_csv(
    CSV_PATH, chunksize=CHUNKSIZE, parse_dates=parse_dates, low_memory=True
)

for i, chunk in enumerate(reader):
    print(f"Processing chunk {i+1} ...")
    # normalize column names in chunk to original
    if date_col and date_col in chunk.columns:
        # ensure datetime
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")

        # Monthly
        months = chunk[date_col].dt.to_period("M").dropna().astype(str)
        for m, cnt in months.value_counts().items():
            monthly_counts[m] = monthly_counts.get(m, 0) + int(cnt)

        # Daily
        days = chunk[date_col].dt.to_period("D").dropna().astype(str)
        for d, cnt in days.value_counts().items():
            daily_counts[d] = daily_counts.get(d, 0) + int(cnt)

        # Year
        yrs = chunk[date_col].dt.year.dropna()
        for y, cnt in yrs.value_counts().items():
            year_counts[int(y)] = year_counts.get(int(y), 0) + int(cnt)

    # Area counts
    if area_col and area_col in chunk.columns:
        for a, cnt in chunk[area_col].fillna("UNKNOWN").value_counts().items():
            area_counts[a] = area_counts.get(a, 0) + int(cnt)

# --------------------------
# 3) Save and plot top areas
# --------------------------
area_series = pd.Series(area_counts).sort_values(ascending=False)
area_series.to_csv(os.path.join(OUTPUT_DIR, "top_areas.csv"), header=["count"])

top_n = 15
plt.figure(figsize=(12, 6))
area_series.head(top_n).plot(kind="bar")
plt.title(f"Top {top_n} Areas by Crime Count")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_areas.png"))
plt.close()

print("Top areas saved to:", os.path.join(OUTPUT_DIR, "top_areas.csv"))

# --------------------------
# 4) Time series outputs & plots
# --------------------------
# Monthly trend
monthly_df = pd.Series(monthly_counts).sort_index()
monthly_df.index = pd.to_datetime(monthly_df.index.astype(str))
monthly_df = monthly_df.sort_index()
monthly_df.to_csv(os.path.join(OUTPUT_DIR, "monthly_trend.csv"), header=["count"])

plt.figure(figsize=(14, 6))
monthly_df.plot()
plt.title("Monthly Crime Trend")
plt.ylabel("Crime Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_trend.png"))
plt.close()

# Daily trend (optionally large; save to CSV)
daily_df = pd.Series(daily_counts).sort_index()
if not daily_df.empty:
    daily_df.index = pd.to_datetime(daily_df.index.astype(str))
    daily_df = daily_df.sort_index()
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "daily_trend.csv"), header=["count"])

# Yearly
year_df = pd.Series(year_counts).sort_index()
year_df.to_csv(os.path.join(OUTPUT_DIR, "yearly_counts.csv"), header=["count"])

plt.figure(figsize=(8, 5))
year_df.plot(kind="bar")
plt.title("Crime Count per Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "yearly_counts.png"))
plt.close()

print("Time-series outputs saved to", OUTPUT_DIR)

# --------------------------
# 5) Answering the key questions (quick summary printed)
# --------------------------
print("\n--- Quick Answers ---")
if not area_series.empty:
    top_area = area_series.idxmax()
    print(
        f"Which areas have the highest crime rates? Top area (by count): {top_area} ({area_series.max()} incidents)"
    )
else:
    print(
        "No AREA column found; can't compute area-based counts. See outputs for details."
    )

# Has crime increased or decreased over the last year?
if not year_df.empty:
    years_sorted = sorted(year_df.index)
    if len(years_sorted) >= 2:
        last, prev = years_sorted[-1], years_sorted[-2]
        last_count = int(year_df.loc[last])
        prev_count = int(year_df.loc[prev])
        trend = (
            "increased"
            if last_count > prev_count
            else "decreased" if last_count < prev_count else "stayed the same"
        )
        print(
            f"Year-over-year: {prev} -> {last}: {prev_count} -> {last_count} (crime has {trend})"
        )
    else:
        print("Not enough years to compare.")
else:
    print("No year information available.")

# --------------------------
# 6) Geospatial mapping (only if lat/lon columns exist AND folium installed)
# --------------------------
if lat_col and lon_col and FOLIUM_AVAILABLE:
    print("\nCreating geospatial map (sampled).")
    # load a sample of rows with coordinates (we'll read in chunks and sample)
    sampled_points = []
    reader = pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE)
    total_collected = 0
    for chunk in reader:
        if lat_col not in chunk.columns or lon_col not in chunk.columns:
            break
        coords = chunk[[lat_col, lon_col]].dropna()
        # quick numeric coercion
        coords[lat_col] = pd.to_numeric(coords[lat_col], errors="coerce")
        coords[lon_col] = pd.to_numeric(coords[lon_col], errors="coerce")
        coords = coords.dropna()
        if coords.empty:
            continue
        # append sample from this chunk
        needed = max(0, SAMPLE_FOR_MAP - total_collected)
        if needed <= 0:
            break
        sample = coords.sample(n=min(len(coords), needed), random_state=42)
        sampled_points.append(sample)
        total_collected += len(sample)
        if total_collected >= SAMPLE_FOR_MAP:
            break

    if sampled_points:
        sdf = pd.concat(sampled_points, ignore_index=True)
        center_lat = sdf[lat_col].mean()
        center_lon = sdf[lon_col].mean()
        crime_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        # Add heatmap
        heat_data = sdf[[lat_col, lon_col]].values.tolist()
        HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(crime_map)

        # Add small circle markers (limit to first 1000)
        for idx, row in sdf.head(1000).iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]], radius=2, fill=True
            ).add_to(crime_map)

        out_map = os.path.join(OUTPUT_DIR, "crime_map.html")
        crime_map.save(out_map)
        print("Map saved to", out_map)
    else:
        print("No valid coordinate rows found to create a map.")
else:
    if not FOLIUM_AVAILABLE:
        print(
            "\nfolium not available — install it (`pip install folium`) to create interactive maps."
        )
    else:
        print(
            "\nNo latitude/longitude columns detected. To map incidents you need coordinates."
        )
        print(
            "If your dataset only has addresses, see optional geocoding example below (slow & rate-limited)."
        )

# --------------------------
# 7) Optional: Geocode addresses (if you only have LOCATION strings)
# --------------------------
print("\nOptional: If you only have addresses and want lat/lon, you can geocode them.")
print(
    "Warning: Geocoding large datasets requires an API (Google, Mapbox, etc.) and will be rate limited."
)
print("Example (slow) using geopy + Nominatim (not recommended for large files):")
print(
    """
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="crime_geocoder_example", timeout=10)
def geocode_address(addr):
    try:
        loc = geolocator.geocode(addr)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except Exception as e:
        return (None, None)
# Apply to a small sample of unique addresses, then merge back.
"""
)

print("\nDone. Check the", OUTPUT_DIR, "folder for CSVs and plots.")
