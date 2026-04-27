"""Step 4-5: RFM Analysis — compute, score, and label segments."""
import pandas as pd
import numpy as np


def compute_rfm(df):
    """Raw RFM values per customer."""
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"Reference date: {reference_date.date()}")

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    print(f"\nRFM computed for {len(rfm):,} customers")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(1))
    return rfm


def score_rfm(rfm):
    """Assign R, F, M scores (1-5) using quantiles."""
    # Recency: LOWER = BETTER, so reverse labels
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency & Monetary: HIGHER = BETTER
    # rank(method='first') handles duplicate values at bin edges
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Combined score (3-15 range) and segment string ("555" = best)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    print(f"\nRFM Score distribution:")
    print(rfm["RFM_Score"].describe().round(1))
    return rfm


def assign_segment_name(row):
    """Map RFM scores to business-friendly names."""
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 4 and f >= 4 and m < 4:
        return "Loyal Customers"
    elif r >= 4 and f < 3:
        return "New Customers"
    elif r >= 3 and f >= 3:
        return "Promising"
    elif r <= 2 and f >= 4:
        return "Can't Lose Them"
    elif r <= 2 and f >= 2 and m >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2:
        return "Lost"
    else:
        return "Need Attention"


def label_segments(rfm):
    """Apply segment names and print summary."""
    rfm["Segment"] = rfm.apply(assign_segment_name, axis=1)

    total_rev = rfm["Monetary"].sum()
    total_cust = len(rfm)

    summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(1)
    )
    summary["Cust_Pct"] = (summary["Customers"] / total_cust * 100).round(1)
    summary["Rev_Pct"] = (summary["Total_Revenue"] / total_rev * 100).round(1)
    summary = summary.sort_values("Total_Revenue", ascending=False)

    print(f"\n{'='*80}")
    print(f"SEGMENT SUMMARY")
    print(f"{'='*80}")
    for seg_name, row in summary.iterrows():
        print(
            f"  {seg_name:20s} | {row['Customers']:>5.0f} customers ({row['Cust_Pct']:>5.1f}%) | "
            f"£{row['Total_Revenue']:>12,.0f} revenue ({row['Rev_Pct']:>5.1f}%) | "
            f"Avg R={row['Avg_Recency']:.0f}d F={row['Avg_Frequency']:.1f} M=£{row['Avg_Monetary']:,.0f}"
        )
    print(f"{'='*80}")

    return rfm


if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean_transactions.csv", parse_dates=["InvoiceDate"])
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    rfm = label_segments(rfm)
    rfm.to_csv("data/processed/rfm_scored.csv", index=False)
    print(f"\nSaved to data/processed/rfm_scored.csv")
