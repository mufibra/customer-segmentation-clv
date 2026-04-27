"""Customer Segmentation & CLV Intelligence Dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segmentation & CLV", layout="wide")
st.title("Customer Segmentation & CLV Intelligence")
st.caption(
    "RFM Analysis · K-Means Clustering · BG/NBD & Gamma-Gamma CLV Prediction  |  "
    "UCI Online Retail Dataset (Dec 2010 – Dec 2011)"
)


# ── Load data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    rfm = pd.read_csv("data/processed/rfm_clustered.csv")
    clv_full = pd.read_csv("data/processed/clv_predictions.csv")
    clv_ret = pd.read_csv("data/processed/clv_returning_only.csv")
    return rfm, clv_full, clv_ret


rfm, clv_full, clv_ret = load_data()

# Color map for consistent cluster colors
COLOR_MAP = {
    "VIP Champions": "#2563EB",
    "Loyal Regulars": "#16A34A",
    "New Potentials": "#F59E0B",
    "Lost / Dormant": "#DC2626",
}

# ── Tab 1: Segment Overview ──────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Segment Overview",
        "🔍 Cluster Deep Dive",
        "💰 CLV Analysis",
        "🔎 Customer Lookup",
        "📋 Recommendations",
    ]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(rfm):,}")
    c2.metric("Total Revenue", f"£{rfm['Monetary'].sum():,.0f}")
    c3.metric("Segments", rfm["Cluster_Name"].nunique())
    avg_clv = clv_ret["CLV"].mean() if "CLV" in clv_ret.columns else 0
    c4.metric("Avg CLV (12mo)", f"£{avg_clv:,.0f}")

    st.markdown("---")

    # Segment summary table
    seg = (
        rfm.groupby("Cluster_Name")
        .agg(
            Customers=("CustomerID", "count"),
            Revenue=("Monetary", "sum"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        )
        .reset_index()
        .round(1)
    )
    seg["Rev_Pct"] = (seg["Revenue"] / seg["Revenue"].sum() * 100).round(1)
    seg = seg.sort_values("Revenue", ascending=False)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Segments by Revenue")
        fig = px.treemap(
            seg,
            path=["Cluster_Name"],
            values="Revenue",
            color="Cluster_Name",
            color_discrete_map=COLOR_MAP,
            title=None,
        )
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Revenue Distribution")
        fig = px.bar(
            seg,
            x="Cluster_Name",
            y="Revenue",
            color="Cluster_Name",
            color_discrete_map=COLOR_MAP,
            text=seg["Rev_Pct"].apply(lambda x: f"{x:.0f}%"),
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Details")
    st.dataframe(
        seg[
            [
                "Cluster_Name",
                "Customers",
                "Rev_Pct",
                "Avg_Recency",
                "Avg_Frequency",
                "Avg_Monetary",
                "Revenue",
            ]
        ].rename(
            columns={
                "Cluster_Name": "Segment",
                "Rev_Pct": "Revenue %",
                "Avg_Recency": "Avg Recency (days)",
                "Avg_Frequency": "Avg Orders",
                "Avg_Monetary": "Avg Spend (£)",
                "Revenue": "Total Revenue (£)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ── Tab 2: Cluster Deep Dive ────────────────────────────
with tab2:
    st.subheader("3D Cluster Visualization")
    st.caption("Each point is a customer, colored by segment. Drag to rotate.")

    fig = px.scatter_3d(
        rfm,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cluster_Name",
        color_discrete_map=COLOR_MAP,
        opacity=0.5,
        hover_data=["CustomerID", "RFM_Score"],
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Recency (days)",
            yaxis_title="Frequency (orders)",
            zaxis_title="Monetary (£)",
        ),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Snake plot
    st.subheader("Snake Plot — Normalized Cluster Profiles")
    means = rfm.groupby("Cluster_Name")[["Recency", "Frequency", "Monetary"]].mean()
    means_norm = (means - means.mean()) / means.std()

    fig2 = go.Figure()
    for name in means_norm.index:
        fig2.add_trace(
            go.Scatter(
                x=means_norm.columns,
                y=means_norm.loc[name],
                mode="lines+markers",
                name=name,
                line=dict(color=COLOR_MAP.get(name), width=3),
                marker=dict(size=10),
            )
        )
    fig2.update_layout(
        yaxis_title="Normalized Value",
        xaxis_title="",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # RFM Score distribution by cluster
    st.subheader("RFM Score Distribution by Segment")
    fig3 = px.histogram(
        rfm,
        x="RFM_Score",
        color="Cluster_Name",
        color_discrete_map=COLOR_MAP,
        barmode="overlay",
        nbins=13,
        opacity=0.7,
    )
    fig3.update_layout(xaxis_title="RFM Score (3-15)", yaxis_title="Customer Count")
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 3: CLV Analysis ─────────────────────────────────
with tab3:
    st.subheader("CLV Distribution (Returning Customers)")

    fig = px.histogram(
        clv_ret,
        x="CLV",
        color="CLV_Tier",
        nbins=50,
        marginal="box",
        color_discrete_sequence=["#94A3B8", "#F59E0B", "#16A34A", "#2563EB"],
    )
    fig.update_layout(xaxis_title="Customer Lifetime Value (£)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # Pareto chart
    st.subheader("Pareto Analysis — Cumulative CLV")
    clv_sorted = clv_ret.sort_values("CLV", ascending=False).reset_index(drop=True)
    clv_sorted["cum_pct"] = clv_sorted["CLV"].cumsum() / clv_sorted["CLV"].sum() * 100
    clv_sorted["cust_pct"] = (clv_sorted.index + 1) / len(clv_sorted) * 100

    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=clv_sorted["cust_pct"],
            y=clv_sorted["cum_pct"],
            mode="lines",
            name="Cumulative CLV %",
            line=dict(color="#2563EB", width=3),
        )
    )
    fig4.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% of CLV")
    fig4.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="20% of customers")
    fig4.update_layout(
        xaxis_title="% of Customers (ranked by CLV)",
        yaxis_title="% of Cumulative CLV",
        height=400,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Top 20 by CLV
    st.subheader("Top 20 Customers by Predicted CLV (12 months)")
    top = clv_ret.nlargest(20, "CLV")[
        ["CustomerID", "frequency", "monetary_value", "pred_purchases_90d", "CLV", "CLV_Tier"]
    ].reset_index(drop=True)
    top.columns = ["Customer ID", "Repeat Purchases", "Avg Order (£)", "Pred Purchases (90d)", "CLV (£)", "Tier"]
    st.dataframe(
        top.style.format(
            {"Avg Order (£)": "£{:,.2f}", "CLV (£)": "£{:,.0f}", "Pred Purchases (90d)": "{:.1f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )


# ── Tab 4: Customer Lookup ───────────────────────────────
with tab4:
    st.subheader("Customer Lookup")
    all_ids = sorted(clv_full["CustomerID"].dropna().unique().astype(int))
    cid = st.selectbox("Select Customer ID", all_ids, index=0)

    row = clv_full[clv_full["CustomerID"] == cid]
    if len(row) > 0:
        r = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Segment", r.get("Cluster_Name", "N/A"))
        c2.metric("Total Spent", f"£{r['Monetary']:,.0f}")
        c3.metric("RFM Score", f"{int(r['R_Score'])}{int(r['F_Score'])}{int(r['M_Score'])}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recency", f"{int(r['Recency'])} days")
        c5.metric("Frequency", f"{int(r['Frequency'])} orders")
        c6.metric("RFM Segment Label", r.get("Segment", "N/A"))

        if pd.notna(r.get("CLV")):
            st.markdown("---")
            st.markdown("**CLV Predictions (Returning Customer)**")
            c7, c8, c9 = st.columns(3)
            c7.metric("Predicted CLV (12mo)", f"£{r['CLV']:,.0f}")
            c8.metric("CLV Tier", r.get("CLV_Tier", "N/A"))
            c9.metric(
                "Predicted Purchases (90d)",
                f"{r['pred_purchases_90d']:.1f}" if pd.notna(r.get("pred_purchases_90d")) else "N/A",
            )

            st.info(f"**Recommended Action:** {r.get('Cluster_Action', 'N/A')}")
        else:
            st.warning("One-time buyer — no CLV prediction available (requires repeat purchases).")
            st.info(f"**Recommended Action:** {r.get('Cluster_Action', 'N/A')}")
    else:
        st.warning("Customer not found.")


# ── Tab 5: Recommendations ───────────────────────────────
with tab5:
    st.subheader("Marketing Actions by Segment")

    actions = (
        clv_full.groupby(["Cluster_Name", "Cluster_Action"])
        .agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    for _, seg in actions.iterrows():
        with st.container():
            color = COLOR_MAP.get(seg["Cluster_Name"], "#666")
            st.markdown(
                f"### <span style='color:{color}'>{seg['Cluster_Name']}</span>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Customers", f"{seg['Customers']:,}")
            c2.metric("Historical Revenue", f"£{seg['Revenue']:,.0f}")
            st.info(f"**→ {seg['Cluster_Action']}**")
            st.markdown("---")

    st.subheader("Key Findings")
    champs = rfm[rfm["Cluster_Name"] == "VIP Champions"]
    total_rev = rfm["Monetary"].sum()
    st.markdown(
        f"""
    - **Pareto Principle confirmed:** {len(champs):,} VIP Champions ({len(champs)/len(rfm)*100:.0f}% of customers) generate £{champs['Monetary'].sum():,.0f} ({champs['Monetary'].sum()/total_rev*100:.0f}% of revenue)
    - **{(rfm['Cluster_Name'] == 'Lost / Dormant').sum():,} customers are Lost/Dormant** — haven't purchased in avg 182 days. Win-back campaigns could recover some of this segment.
    - **{(rfm['Cluster_Name'] == 'New Potentials').sum():,} New Potentials** bought recently but only once or twice — onboarding + education content could move them to Loyal Regulars.
    - **CLV Tier spread: 16x** — Platinum tier avg CLV is £{clv_ret.groupby('CLV_Tier')['CLV'].mean().get('Platinum', 0):,.0f} vs Low tier £{clv_ret.groupby('CLV_Tier')['CLV'].mean().get('Low', 0):,.0f}.
    """
    )
