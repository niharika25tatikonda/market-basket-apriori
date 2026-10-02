"""
app.py  -  Market Basket Analysis using the Apriori algorithm (Streamlit UI)

Run with:   streamlit run app.py
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from collections import Counter

from apriori import load_baskets, apriori, make_rules

DEFAULT_CSV = "data/grocery_transactions.csv"

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------
# Cached helper functions (so the app stays fast when sliders change)
# ----------------------------------------------------------------------
@st.cache_data
def read_data(file):
    return pd.read_csv(file)


@st.cache_data
def get_baskets(df):
    return load_baskets(df)


@st.cache_data
def get_frequent_itemsets(_baskets, min_support, max_len):
    return apriori(_baskets, min_support, max_len)


# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
st.title("🛒 Market Basket Analysis using Apriori")
st.write(
    "Find out **which products customers buy together** in a single visit to a grocery store. "
    "Change the settings on the left and the results update automatically."
)

# ----------------------------------------------------------------------
# Sidebar: dataset + settings
# ----------------------------------------------------------------------
st.sidebar.header("1. Dataset")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
df = read_data(uploaded if uploaded is not None else DEFAULT_CSV)

required = {"Transaction_ID", "Items"}
if not required.issubset(df.columns):
    st.error(f"The CSV must contain these columns: {', '.join(sorted(required))}")
    st.stop()

st.sidebar.header("2. Apriori settings")
min_support_pct = st.sidebar.slider(
    "Minimum support (%)", 1.0, 20.0, 5.0, 0.5,
    help="An itemset must appear in at least this % of all baskets.",
)
min_conf_pct = st.sidebar.slider(
    "Minimum confidence (%)", 10, 100, 60, 5,
    help="How often the 'then' item is bought when the 'if' item is bought.",
)
min_lift = st.sidebar.slider(
    "Minimum lift", 1.0, 10.0, 2.0, 0.5,
    help="Lift above 1 means the items are bought together more than by chance.",
)
max_len = st.sidebar.select_slider(
    "Maximum items in an itemset", options=[2, 3, 4], value=3
)
st.sidebar.info(
    "Tip: if you see too few rules, lower the minimum support, confidence or lift. "
    "If you see too many, raise them."
)

# ----------------------------------------------------------------------
# Run the algorithm
# ----------------------------------------------------------------------
baskets = get_baskets(df)
frequent = get_frequent_itemsets(baskets, min_support_pct / 100, max_len)
rules = make_rules(frequent, min_conf_pct / 100, min_lift)

# ----------------------------------------------------------------------
# Explanation of the terms
# ----------------------------------------------------------------------
with st.expander("📘 How does it work? (click to read)"):
    st.markdown(
        """
**Basket** – everything one customer bought in one visit to the store.

**Apriori steps**
1. Count how often each single item appears and keep the *frequent* ones (support ≥ minimum).
2. Combine frequent items into pairs, keep the frequent pairs. Then triples, and so on.
3. Turn the frequent itemsets into rules like **"if Bread → then Butter"**.

**Measures**
| Term | Meaning | Formula |
|---|---|---|
| **Support** | How popular the itemset is | baskets with the itemset ÷ total baskets |
| **Confidence** | How often *B* is bought when *A* is bought | support(A and B) ÷ support(A) |
| **Lift** | Is the link stronger than chance? (>1 = yes) | confidence ÷ support(B) |
"""
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset overview", "🧺 Frequent itemsets", "🔗 Association rules", "💡 Recommendations"]
)

# ======================================================================
# TAB 1 : Dataset overview
# ======================================================================
with tab1:
    st.subheader("About the dataset")
    st.write(
        "Each row is **one visit to the grocery store** (one bill). "
        "The *Items* column lists everything the customer bought in that visit."
    )
    basket_sizes = [len(b) for b in baskets]
    c1, c2, c3 = st.columns(3)
    c1.metric("Visits (transactions)", f"{len(baskets):,}")
    c2.metric("Unique products", f"{len(set().union(*baskets)):,}")
    c3.metric("Average items per visit", f"{sum(basket_sizes) / len(baskets):.1f}")

    st.write(f"**All {len(df):,} transactions** (scroll inside the table to see more)")
    search = st.text_input("Search by product (optional)", placeholder="e.g. milk")
    shown = df[df["Items"].str.contains(search.strip(), case=False, regex=False)] if search.strip() else df
    if search.strip():
        st.caption(f"{len(shown):,} visits contain '{search.strip()}'")
    st.dataframe(shown, use_container_width=True, hide_index=True, height=400)

    left, right = st.columns(2)
    with left:
        top_n = st.slider("Number of top products to show", 5, 20, 10)
        counts = Counter(item for b in baskets for item in b)
        top_items = counts.most_common(top_n)
        names = [name for name, _ in top_items][::-1]
        values = [c for _, c in top_items][::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(names, values, color="#2a9d8f")
        ax.set_xlabel("Number of visits containing the product")
        ax.set_title(f"Top {top_n} most frequently bought products")
        plt.tight_layout()
        st.pyplot(fig)
    with right:
        size_counts = Counter(basket_sizes)
        sizes = sorted(size_counts)
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.bar(sizes, [size_counts[s] for s in sizes], color="#e9c46a")
        ax.set_xlabel("Number of items in a visit")
        ax.set_ylabel("Number of visits")
        ax.set_title("How big are the baskets?")
        plt.tight_layout()
        st.pyplot(fig)

# ======================================================================
# TAB 2 : Frequent itemsets
# ======================================================================
with tab2:
    st.subheader("Frequent itemsets")
    st.write(
        f"Itemsets that appear in at least **{min_support_pct:.2f}%** of the "
        f"{len(baskets):,} visits."
    )

    if frequent.empty:
        st.warning("No frequent itemsets found. Lower the minimum support.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total frequent itemsets", len(frequent))
        c2.metric("Single items", int((frequent["size"] == 1).sum()))
        c3.metric("Item combinations (2+)", int((frequent["size"] >= 2).sum()))

        show = frequent.copy()
        show["itemset"] = show["itemset"].apply(lambda t: ", ".join(t))
        show["support"] = show["support"] * 100
        show = show.rename(
            columns={"itemset": "Itemset", "size": "Items", "count": "Baskets", "support": "Support (%)"}
        )
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={"Support (%)": st.column_config.NumberColumn(format="%.2f")},
        )

        combos = show[show["Items"] >= 2].head(10)
        if not combos.empty:
            st.write("**Top item combinations by support**")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(combos["Itemset"][::-1], combos["Support (%)"][::-1], color="#e76f51")
            ax.set_xlabel("Support (%)")
            plt.tight_layout()
            st.pyplot(fig)

# ======================================================================
# TAB 3 : Association rules
# ======================================================================
with tab3:
    st.subheader("Association rules")
    st.write(
        f"Rules with confidence ≥ **{min_conf_pct}%** and lift ≥ **{min_lift:.2f}**, "
        "sorted by lift (strongest link first)."
    )

    if rules.empty:
        st.warning("No rules found. Try lowering minimum support, confidence or lift.")
    else:
        st.metric("Number of rules found", len(rules))

        st.write("**Top 5 rules in plain English**")
        for _, r in rules.head(5).iterrows():
            st.success(
                f"Customers who buy **{r['if customer buys']}** also buy "
                f"**{r['then also buys']}** in {r['confidence']*100:.1f}% of cases "
                f"(lift {r['lift']:.2f} → {r['lift']:.2f}× more likely than normal)."
            )

        st.write("**All rules**")
        show = rules.copy()
        show["support"] = show["support"] * 100
        show["confidence"] = show["confidence"] * 100
        show = show.rename(
            columns={"support": "Support (%)", "confidence": "Confidence (%)", "lift": "Lift"}
        )
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Support (%)": st.column_config.NumberColumn(format="%.2f"),
                "Confidence (%)": st.column_config.NumberColumn(format="%.1f"),
                "Lift": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button(
            "⬇️ Download rules as CSV",
            rules.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv",
        )

        st.write("**Support vs Confidence** (darker colour = higher lift)")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sc = ax.scatter(
            rules["support"] * 100, rules["confidence"] * 100,
            c=rules["lift"], cmap="viridis_r", s=60, edgecolor="black",
        )
        ax.set_xlabel("Support (%)")
        ax.set_ylabel("Confidence (%)")
        fig.colorbar(sc, label="Lift")
        plt.tight_layout()
        st.pyplot(fig)

# ======================================================================
# TAB 4 : Recommendations
# ======================================================================
with tab4:
    st.subheader("Product recommendations")
    st.write("Pick a product to see what customers usually buy along with it.")

    if rules.empty:
        st.warning("No rules available. Adjust the settings in the sidebar.")
    else:
        items_with_rules = sorted(rules["if customer buys"].unique())
        chosen = st.selectbox("Product", items_with_rules)
        matches = rules[rules["if customer buys"] == chosen].sort_values(
            "confidence", ascending=False
        )
        st.write(f"Customers who buy **{chosen}** also tend to buy:")
        for _, r in matches.iterrows():
            st.info(
                f"**{r['then also buys']}**  —  confidence {r['confidence']*100:.1f}%, "
                f"lift {r['lift']:.2f}"
            )
        st.caption(
            "Use this for shelf placement, combo offers, or 'customers also bought' suggestions."
        )
