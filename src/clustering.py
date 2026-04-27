"""Step 6-7: K-Means Clustering on RFM features."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def prepare_features(rfm):
    """Log-transform + scale RFM for K-Means."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    print("Skewness BEFORE transform:")
    skew_before = features.skew().round(2)
    for col, val in skew_before.items():
        print(f"  {col}: {val}")

    # Log transform — RFM values are heavily right-skewed
    features_log = np.log1p(features)

    print("\nSkewness AFTER log transform:")
    skew_after = features_log.skew().round(2)
    for col, val in skew_after.items():
        print(f"  {col}: {val}")

    # Scale — K-Means uses Euclidean distance, so scale matters
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_log)

    return scaled, scaler


def find_optimal_k(scaled, k_range=range(2, 11)):
    """Elbow + silhouette to pick k."""
    results = []
    print("\nSearching for optimal k...")
    print(f"{'k':>3} | {'Inertia':>10} | {'Silhouette':>10}")
    print("-" * 30)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"{k:>3} | {km.inertia_:>10,.0f} | {sil:>10.4f}")

    best = max(results, key=lambda x: x["silhouette"])
    print(f"\nBest k={best['k']} (silhouette={best['silhouette']:.4f})")
    return results, best["k"]


def fit_and_profile(rfm, scaled, n_clusters):
    """Fit final K-Means and profile clusters."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = km.fit_predict(scaled)

    sil = silhouette_score(scaled, rfm["Cluster"])
    print(f"\nFinal model: k={n_clusters}, silhouette={sil:.4f}")

    total_rev = rfm["Monetary"].sum()
    total_cust = len(rfm)

    print(f"\n{'='*90}")
    print(f"CLUSTER PROFILES")
    print(f"{'='*90}")

    # Sort clusters by avg monetary (descending) to make naming easier
    cluster_order = (
        rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index
    )

    for c in cluster_order:
        grp = rfm[rfm["Cluster"] == c]
        rev_pct = grp["Monetary"].sum() / total_rev * 100
        cust_pct = len(grp) / total_cust * 100
        print(
            f"  Cluster {c}: {len(grp):>5,} customers ({cust_pct:>5.1f}%) | "
            f"Revenue: £{grp['Monetary'].sum():>12,.0f} ({rev_pct:>5.1f}%) | "
            f"Avg R={grp['Recency'].mean():.0f}d, F={grp['Frequency'].mean():.1f}, M=£{grp['Monetary'].mean():,.0f}"
        )
    print(f"{'='*90}")

    return rfm, km


def name_clusters(rfm):
    """
    Assign business names based on cluster profiles.
    Must be called AFTER inspecting fit_and_profile output.
    This mapping is determined by examining the cluster stats.
    """
    # Sort clusters by avg monetary to create a consistent ordering
    cluster_stats = rfm.groupby("Cluster").agg(
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
        count=("CustomerID", "count"),
    )

    # Assign names based on characteristics
    cluster_map = {}
    for c, row in cluster_stats.iterrows():
        if row["avg_frequency"] > 8 and row["avg_monetary"] > 3000:
            cluster_map[c] = {
                "name": "VIP Champions",
                "action": "Loyalty rewards, referral programs, exclusive early access",
            }
        elif row["avg_frequency"] > 3 and row["avg_recency"] < 80:
            cluster_map[c] = {
                "name": "Loyal Regulars",
                "action": "Upsell campaigns, early access to new products, cross-sell",
            }
        elif row["avg_recency"] < 40 and row["avg_frequency"] < 3:
            cluster_map[c] = {
                "name": "New Potentials",
                "action": "Welcome series, onboarding discounts, education content",
            }
        elif row["avg_recency"] > 150 and row["avg_frequency"] > 2:
            cluster_map[c] = {
                "name": "At-Risk Churners",
                "action": 'Win-back campaigns, "we miss you" emails, special offers',
            }
        elif row["avg_recency"] > 150:
            cluster_map[c] = {
                "name": "Lost Causes",
                "action": "Final reactivation attempt or remove from active campaigns",
            }
        else:
            cluster_map[c] = {
                "name": "Need Attention",
                "action": "Targeted engagement, personalized recommendations",
            }

    rfm["Cluster_Name"] = rfm["Cluster"].map(lambda x: cluster_map[x]["name"])
    rfm["Cluster_Action"] = rfm["Cluster"].map(lambda x: cluster_map[x]["action"])

    print("\nCluster naming applied:")
    for c, info in sorted(cluster_map.items()):
        count = (rfm["Cluster"] == c).sum()
        print(f"  Cluster {c} → {info['name']} ({count:,} customers)")

    return rfm, cluster_map


if __name__ == "__main__":
    rfm = pd.read_csv("data/processed/rfm_scored.csv")
    scaled, scaler = prepare_features(rfm)
    results, best_k = find_optimal_k(scaled)

    # Let user pick k — silhouette favors k=2 but that's too coarse for marketing
    print(f"\nSilhouette suggests k={best_k}, but review the table above.")
    user_k = input(f"Enter k to use (default {best_k}): ").strip()
    chosen_k = int(user_k) if user_k else best_k

    rfm, km = fit_and_profile(rfm, scaled, chosen_k)
    rfm, cluster_map = name_clusters(rfm)
    rfm.to_csv("data/processed/rfm_clustered.csv", index=False)
    print(f"\nSaved to data/processed/rfm_clustered.csv")
