# Customer Segmentation & CLV Prediction

A customer segmentation engine combining **RFM analysis**, **K-Means clustering**, and **probabilistic CLV prediction** (BG/NBD + Gamma-Gamma models) to identify which customers are most valuable, which are about to churn, and what marketing action each segment needs — with pound-sterling values attached.

## Key Findings

| Metric | Value |
|--------|-------|
| Total Customers Analyzed | 4,338 |
| Total Revenue (1 year) | £8,911,408 |
| Optimal Clusters | 4 (silhouette = 0.337) |
| Pareto Insight | **Top 20% of returning customers → 66% of predicted future revenue** |
| CLV Range | £28 — £1,756,709 |
| Platinum vs Low Tier | 16x difference in avg CLV |

### The Four Segments

| Segment | Customers | Revenue Share | Avg Recency | Avg Orders | Avg Spend |
|---------|-----------|---------------|-------------|------------|-----------|
| VIP Champions | 716 (16.5%) | 64.9% | 12 days | 13.7 | £8,074 |
| Loyal Regulars | 1,173 (27.0%) | 23.7% | 71 days | 4.1 | £1,803 |
| New Potentials | 837 (19.3%) | 5.2% | 18 days | 2.1 | £552 |
| Lost / Dormant | 1,612 (37.2%) | 6.2% | 182 days | 1.3 | £343 |

## Architecture

```
Raw Transactions (UCI Online Retail, 541K rows)
        │
        ▼
  Data Cleaning ──► 397,884 clean rows, 4,338 customers
        │
        ▼
  RFM Computation ──► Recency, Frequency, Monetary + quantile scoring (1-5)
        │
    ┌───┴────┐
    ▼        ▼
 K-Means   CLV Prediction
 (k=4)     (lifetimes library)
    │        │
    │        ├── BG/NBD → purchase frequency prediction
    │        └── Gamma-Gamma → monetary value prediction
    │                │
    └───────┬────────┘
            ▼
  Segment Profiling + CLV per customer
            │
            ▼
  Streamlit Dashboard (5 tabs)
```

## Tech Stack

- **Python 3.12+**
- **pandas / numpy** — data cleaning & RFM computation
- **scikit-learn** — K-Means, StandardScaler, silhouette analysis
- **lifetimes** — BG/NBD + Gamma-Gamma probabilistic CLV models
- **plotly** — interactive 3D scatter, treemaps, snake plots, Pareto chart
- **streamlit** — deployed interactive dashboard
- **scipy** — log transforms for skewness reduction

> **Note:** The `lifetimes` library is archived. Its official successor is [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing), which provides Bayesian CLV modeling. For production use, consider migrating to PyMC-Marketing.

## Dataset

**UCI Online Retail Dataset** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

541,909 transactions from a UK-based online retailer (Dec 2010 – Dec 2011). The company sells unique all-occasion gifts, primarily to wholesalers.

Source: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

## How to Run

```bash
# Clone
git clone https://github.com/mufibra23/customer-segmentation-clv.git
cd customer-segmentation-clv

# Setup
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Run pipeline (or use notebooks/analysis.ipynb)
python src/data_cleaning.py
python src/rfm_analysis.py
python src/clustering.py
python src/clv_prediction.py

# Launch dashboard
streamlit run app.py
```

## Dashboard

The Streamlit app has 5 tabs:

1. **Segment Overview** — KPIs, treemap, revenue distribution
2. **Cluster Deep Dive** — 3D scatter plot, snake plot, RFM score distribution
3. **CLV Analysis** — CLV histogram, Pareto curve, top 20 customers
4. **Customer Lookup** — enter any Customer ID to see their segment, RFM, and CLV
5. **Recommendations** — marketing actions per segment with key findings

## Project Structure

```
customer-segmentation-clv/
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── raw/online_retail.csv
│   └── processed/
│       ├── clean_transactions.csv
│       ├── rfm_scored.csv
│       ├── rfm_clustered.csv
│       ├── clv_predictions.csv
│       └── clv_returning_only.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   └── clv_prediction.py
└── assets/screenshots/
```

## Methodology Notes

- **RFM Scoring:** Quantile-based (1-5) with `rank(method='first')` to handle duplicate bin edges
- **Feature Prep:** `log1p` transform to reduce right-skewness before K-Means (Monetary skew: 19.3 → 0.4)
- **K Selection:** k=4 chosen over k=2 (highest silhouette) because 2 clusters are not actionable for marketing. k=4 silhouette (0.337) is still strong for customer data.
- **BG/NBD:** `penalizer_coef=0.1` needed for convergence on this dataset. The `prob_alive` metric shows 100% for all customers due to the short observation window — noted as a limitation.
- **Gamma-Gamma assumption:** Frequency-monetary correlation = 0.016 (well below 0.3 threshold) ✓
- **CLV Horizon:** 12 months with 1% monthly discount rate (~12.7% annually)

## License

This project uses the UCI Online Retail Dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code is MIT licensed.
