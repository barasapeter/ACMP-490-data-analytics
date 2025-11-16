import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Q1 Dataset.csv")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")

cols_to_clean = ["Category", "Sub-Category", "Region", "State", "City"]
for col in cols_to_clean:
    df[col] = df[col].astype(str).str.strip().str.title()

print("\nMissing Values:\n", df.isna().sum())

df = df.drop_duplicates()

print("\nData cleaned. Rows after cleaning:", len(df))
df["Month"] = df["Order Date"].dt.to_period("M")

top_products = (
    df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
)

print("\nTop 10 Products by Sales:\n", top_products)

region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print("\nSales by Region:\n", region_sales)

monthly_sales = df.groupby("Month")["Sales"].sum()
print("\nMonthly Sales:\n", monthly_sales)

plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product Name")
plt.tight_layout()
plt.show()

output_dir = os.path.join(base_dir, "q1AnalysisResults")
os.makedirs(output_dir, exist_ok=True)

top_products.to_csv(os.path.join(output_dir, "topProducts.csv"))
region_sales.to_csv(os.path.join(output_dir, "regionSales.csv"))
monthly_sales.to_csv(os.path.join(output_dir, "monthlySales.csv"))

print("\nCSV outputs saved in:", output_dir)

print("\nAnalysis complete.")
