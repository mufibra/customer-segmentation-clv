"""Step 8-11: CLV Prediction using BG/NBD + Gamma-Gamma (lifetimes library)."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter


def prepare_lifetimes_data(df_clean):
    """
    Transform transaction data to lifetimes' RFM format.
    Note: lifetimes defines frequency as REPEAT purchases (excludes first).
    Customer who bought 3 times → frequency=2.
    """
    summary = summary_data_from_transaction_data(
        df_clean,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=df_clean["InvoiceDate"].max(),
    )

    print(f"Total customers: {len(summary):,}")
    print(f"Returning customers (frequency > 0): {(summary['frequency'] > 0).sum():,}")
    print(f"One-time buyers (frequency = 0): {(summary['frequency'] == 0).sum():,}")
    print(f"\nSummary stats:")
    print(summary.describe().round(2))
    return summary


def fit_bgnbd(summary):
    """Fit BG/NBD model — predicts purchase frequency + churn probability."""
    # Higher penalizer helps convergence on this dataset
    bgf = BetaGeoFitter(penalizer_coef=0.1)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    print("\nBG/NBD Model Parameters:")
    print(bgf.summary)

    # Predict purchases in next 90 days
    summary["pred_purchases_90d"] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            90, summary["frequency"], summary["recency"], summary["T"]
        )
    )

    # Probability still alive
    summary["prob_alive"] = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )

    print(f"\nPredicted purchases (90d) summary:")
    print(summary["pred_purchases_90d"].describe().round(3))

    print(f"\nProb alive summary:")
    print(summary["prob_alive"].describe().round(3))

    alive_pct = (summary["prob_alive"] > 0.5).mean() * 100
    print(f"\nCustomers with >50% prob alive: {alive_pct:.1f}%")

    return bgf


def fit_gamma_gamma(summary):
    """Fit Gamma-Gamma model — predicts average order value."""
    returning = summary[summary["frequency"] > 0].copy()
    print(f"\nFitting Gamma-Gamma on {len(returning):,} returning customers")

    # Check assumption: frequency-monetary should NOT be correlated
    corr = returning[["frequency", "monetary_value"]].corr().iloc[0, 1]
    print(f"Frequency-Monetary correlation: {corr:.3f}", end="")
    if abs(corr) > 0.3:
        print(" ⚠️ WARNING: > 0.3, assumption weakly violated")
    else:
        print(" ✓ OK (< 0.3)")

    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(returning["frequency"], returning["monetary_value"])

    print(f"\nGamma-Gamma Model Parameters:")
    print(ggf.summary)

    returning["pred_avg_order_value"] = ggf.conditional_expected_average_profit(
        returning["frequency"], returning["monetary_value"]
    )

    print(f"\nActual avg order value:    £{returning['monetary_value'].mean():.2f}")
    print(f"Predicted avg order value: £{returning['pred_avg_order_value'].mean():.2f}")

    return ggf, returning


def calculate_clv(bgf, ggf, returning, months=12, discount_rate=0.01):
    """Combine BG/NBD + Gamma-Gamma into CLV predictions."""
    returning = returning.copy()

    returning["CLV"] = ggf.customer_lifetime_value(
        bgf,
        returning["frequency"],
        returning["recency"],
        returning["T"],
        returning["monetary_value"],
        time=months,
        discount_rate=discount_rate,  # ~12.7% annually
    )

    print(f"\nCLV Summary (next {months} months):")
    print(returning["CLV"].describe().round(2))

    # Tier customers
    returning["CLV_Tier"] = pd.qcut(
        returning["CLV"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Platinum"],
    )

    # The money insight — Pareto
    total_clv = returning["CLV"].sum()
    n_top20 = int(len(returning) * 0.2)
    top20_clv = returning.nlargest(n_top20, "CLV")["CLV"].sum()
    pareto_pct = top20_clv / total_clv * 100

    print(f"\n{'='*60}")
    print(f"PARETO INSIGHT")
    print(f"Top 20% of returning customers ({n_top20:,}) account for")
    print(f"  {pareto_pct:.0f}% of predicted future revenue (£{top20_clv:,.0f})")
    print(f"{'='*60}")

    # Tier breakdown
    print(f"\nCLV Tier Breakdown:")
    for tier in ["Platinum", "High", "Medium", "Low"]:
        t = returning[returning["CLV_Tier"] == tier]
        print(
            f"  {tier:10s}: {len(t):>5,} customers | "
            f"Avg CLV: £{t['CLV'].mean():>8,.0f} | "
            f"Total: £{t['CLV'].sum():>12,.0f}"
        )

    return returning


def merge_clv_with_clusters(clv_df, rfm_clustered_path="data/processed/rfm_clustered.csv"):
    """Merge CLV predictions back with cluster labels."""
    rfm = pd.read_csv(rfm_clustered_path)

    # Reset index if CustomerID is the index
    if clv_df.index.name == "CustomerID":
        clv_df = clv_df.reset_index()

    # Merge
    merged = rfm.merge(
        clv_df[["CustomerID", "CLV", "CLV_Tier", "prob_alive", "pred_purchases_90d",
                "pred_avg_order_value", "frequency", "recency", "T", "monetary_value"]],
        on="CustomerID",
        how="left",
        suffixes=("", "_lt"),
    )

    # CLV by cluster
    print(f"\nCLV by Cluster (returning customers only):")
    clv_by_cluster = merged.dropna(subset=["CLV"]).groupby("Cluster_Name").agg(
        customers=("CustomerID", "count"),
        avg_clv=("CLV", "mean"),
        total_clv=("CLV", "sum"),
        avg_prob_alive=("prob_alive", "mean"),
    ).round(1).sort_values("total_clv", ascending=False)

    for name, row in clv_by_cluster.iterrows():
        print(
            f"  {name:20s}: {row['customers']:>5.0f} cust | "
            f"Avg CLV: £{row['avg_clv']:>8,.0f} | "
            f"Avg Prob Alive: {row['avg_prob_alive']:.0%}"
        )

    return merged


if __name__ == "__main__":
    print("Loading clean transactions...")
    df = pd.read_csv(
        "data/processed/clean_transactions.csv", parse_dates=["InvoiceDate"]
    )

    print("\n" + "=" * 60)
    print("STEP 8: Preparing lifetimes data")
    print("=" * 60)
    summary = prepare_lifetimes_data(df)

    print("\n" + "=" * 60)
    print("STEP 9: Fitting BG/NBD model")
    print("=" * 60)
    bgf = fit_bgnbd(summary)

    print("\n" + "=" * 60)
    print("STEP 10: Fitting Gamma-Gamma model")
    print("=" * 60)
    ggf, returning = fit_gamma_gamma(summary)

    print("\n" + "=" * 60)
    print("STEP 11: Calculating CLV")
    print("=" * 60)
    returning = calculate_clv(bgf, ggf, returning)

    # Add prob_alive and pred_purchases from summary back
    returning["prob_alive"] = summary.loc[returning.index, "prob_alive"]
    returning["pred_purchases_90d"] = summary.loc[returning.index, "pred_purchases_90d"]

    # Merge with clusters
    print("\n" + "=" * 60)
    print("MERGING CLV WITH CLUSTERS")
    print("=" * 60)
    merged = merge_clv_with_clusters(returning)
    merged.to_csv("data/processed/clv_predictions.csv", index=False)
    print(f"\nSaved to data/processed/clv_predictions.csv")

    # Also save the returning-only CLV for the dashboard
    returning_out = returning.reset_index() if returning.index.name == "CustomerID" else returning
    returning_out.to_csv("data/processed/clv_returning_only.csv", index=False)
    print(f"Saved returning-only CLV to data/processed/clv_returning_only.csv")
