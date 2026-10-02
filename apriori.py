"""
apriori.py  -  A simple, from-scratch implementation of the Apriori algorithm.

Main ideas (in plain words)
---------------------------
1. A "basket" (transaction) = all items one customer bought in one visit.
2. SUPPORT of an itemset  = fraction of baskets that contain the itemset.
3. Apriori rule: if an itemset is NOT frequent, then no bigger itemset that
   contains it can be frequent either.  This lets us skip a lot of work.
4. From the frequent itemsets we build association rules  A -> B  and
   measure them with Confidence and Lift.
"""

from itertools import combinations
import pandas as pd


# ----------------------------------------------------------------------
# Step 0: turn the raw CSV rows into baskets
# ----------------------------------------------------------------------
def load_baskets(df):
    """
    The CSV has ONE ROW PER VISIT to the store:
        Transaction_ID | Date | Items
        T0001          | ...  | "bread, butter, milk"
    One basket = the set of items in the 'Items' column of one row.
    Returns a list of baskets, each basket being a set of item names.
    """
    baskets = []
    for text in df["Items"]:
        items = {item.strip() for item in str(text).split(",") if item.strip()}
        baskets.append(items)
    return baskets


# ----------------------------------------------------------------------
# Step 1 + 2: find frequent itemsets (the Apriori algorithm)
# ----------------------------------------------------------------------
def apriori(baskets, min_support, max_len=3):
    """
    Returns a DataFrame with columns: itemset, size, count, support.

    baskets     : list of sets of items
    min_support : e.g. 0.01 means "appears in at least 1% of baskets"
    max_len     : largest itemset size to look for
    """
    n = len(baskets)
    min_count = min_support * n

    # For every item remember WHICH baskets contain it (a set of basket ids).
    # The baskets containing an itemset = intersection of these sets.
    basket_ids = {}
    for bid, basket in enumerate(baskets):
        for item in basket:
            basket_ids.setdefault(item, set()).add(bid)

    # ---- Level 1: single items ----
    current = {}  # {itemset (sorted tuple): set of basket ids}
    for item, ids in basket_ids.items():
        if len(ids) >= min_count:
            current[(item,)] = ids

    all_frequent = dict(current)
    size = 1

    # ---- Level 2, 3, ... : grow the itemsets one item at a time ----
    while current and size < max_len:
        size += 1
        next_level = {}
        itemsets = sorted(current.keys())

        # Join step: combine two frequent itemsets that share all but the
        # last item, e.g. (A,B) + (A,C) -> (A,B,C)
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                a, b = itemsets[i], itemsets[j]
                if a[:-1] != b[:-1]:
                    break  # itemsets are sorted, no more matches for 'a'
                candidate = a + (b[-1],)

                # Prune step (the Apriori property): every smaller subset
                # of the candidate must itself be frequent.
                all_subsets_frequent = all(
                    sub in current for sub in combinations(candidate, size - 1)
                )
                if not all_subsets_frequent:
                    continue

                # Count: baskets that contain ALL items of the candidate
                ids = current[a] & current[b]
                if len(ids) >= min_count:
                    next_level[candidate] = ids

        all_frequent.update(next_level)
        current = next_level

    rows = []
    for itemset, ids in all_frequent.items():
        rows.append({
            "itemset": itemset,
            "size": len(itemset),
            "count": len(ids),
            "support": len(ids) / n,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("support", ascending=False).reset_index(drop=True)
    return result


# ----------------------------------------------------------------------
# Step 3: build association rules from the frequent itemsets
# ----------------------------------------------------------------------
def make_rules(frequent, min_confidence=0.1, min_lift=1.0):
    """
    For each frequent itemset with 2+ items, try every way of splitting it
    into  antecedent -> consequent  and compute:

        support(A->B)    = support(A and B)
        confidence(A->B) = support(A and B) / support(A)
        lift(A->B)       = confidence / support(B)

    Lift > 1 : A and B are bought together MORE than by chance
    Lift = 1 : no relationship
    Lift < 1 : bought together LESS than by chance
    """
    if frequent.empty:
        return pd.DataFrame()

    support_of = dict(zip(frequent["itemset"], frequent["support"]))
    rows = []

    for itemset in frequent["itemset"]:
        if len(itemset) < 2:
            continue
        for k in range(1, len(itemset)):
            for antecedent in combinations(itemset, k):
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                antecedent = tuple(sorted(antecedent))

                sup_ab = support_of[itemset]
                sup_a = support_of[antecedent]
                sup_b = support_of[consequent]
                confidence = sup_ab / sup_a
                lift = confidence / sup_b

                if confidence >= min_confidence and lift >= min_lift:
                    rows.append({
                        "if customer buys": ", ".join(antecedent),
                        "then also buys": ", ".join(consequent),
                        "support": sup_ab,
                        "confidence": confidence,
                        "lift": lift,
                    })

    rules = pd.DataFrame(rows)
    if not rules.empty:
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules
