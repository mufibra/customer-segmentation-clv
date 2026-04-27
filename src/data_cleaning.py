"""Step 3: Data Cleaning for UCI Online Retail Dataset."""
import pandas as pd
import os


def clean_retail_data(filepath="data/raw/online_retail.csv"):
    """Clean raw retail data for RFM analysis."""
    df = pd.read_csv(filepath)
    print(f"Raw: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Unique customers (inc NaN): {df['CustomerID'].nunique()}")
    print(f"  Null CustomerIDs: {df['CustomerID'].isna().sum()} ({df['CustomerID'].isna().mean():.1%})")

    # 1. Drop rows without CustomerID — can't segment anonymous buyers
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    print(f"\nAfter dropping null CustomerID: {df.shape[0]:,} rows")

    # 2. Remove cancelled orders (InvoiceNo starts with 'C')
    cancelled = df["InvoiceNo"].astype(str).str.startswith("C").sum()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"After removing {cancelled:,} cancellations: {df.shape[0]:,} rows")

    # 3. Remove negative/zero quantities and prices (returns, errors)
    bad_qty = (df["Quantity"] <= 0).sum()
    bad_price = (df["UnitPrice"] <= 0).sum()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"After removing {bad_qty:,} bad qty + {bad_price:,} bad price: {df.shape[0]:,} rows")

    # 4. Compute line total
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # 5. Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    print(f"\n--- Clean Dataset ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"Unique invoices: {df['InvoiceNo'].nunique():,}")
    print(f"Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    print(f"Total revenue: £{df['TotalPrice'].sum():,.0f}")
    print(f"Countries: {df['Country'].nunique()}")
    print(f"\nTop 5 countries by revenue:")
    top = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head()
    for country, rev in top.items():
        print(f"  {country}: £{rev:,.0f}")

    return df


if __name__ == "__main__":
    df = clean_retail_data()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_transactions.csv", index=False)
    print(f"\nSaved to data/processed/clean_transactions.csv")
