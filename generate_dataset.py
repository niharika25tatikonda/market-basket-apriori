"""
generate_dataset.py - creates a realistic grocery-store dataset.

Each ROW = ONE VISIT (one bill) to the store, with every item bought in that visit.
Customers usually shop with a "purpose" (breakfast, curry night, tea time ...),
so some items naturally appear together. A few random extra items are added too.

Run:  python generate_dataset.py
"""
import random
import csv
from datetime import date, timedelta

random.seed(42)
N_TRANSACTIONS = 1500

# Shopping "purposes": (probability of the purpose in a visit, {item: chance it is bought})
PURPOSES = {
    "breakfast":   (0.26, {"bread": 0.90, "butter": 0.75, "milk": 0.70, "eggs": 0.55, "jam": 0.30}),
    "cereal":      (0.10, {"cereal": 0.95, "milk": 0.90, "bananas": 0.30}),
    "tea time":    (0.14, {"tea": 0.90, "sugar": 0.75, "biscuits": 0.70, "milk": 0.40}),
    "coffee":      (0.08, {"coffee": 0.95, "sugar": 0.60, "milk": 0.70, "biscuits": 0.25}),
    "curry night": (0.20, {"rice": 0.85, "onions": 0.90, "tomatoes": 0.85, "potatoes": 0.60,
                           "cooking oil": 0.60, "salt": 0.35}),
    "chicken meal":(0.08, {"chicken": 0.95, "onions": 0.60, "rice": 0.60, "cooking oil": 0.40}),
    "pasta night": (0.09, {"pasta": 0.95, "pasta sauce": 0.85, "cheese": 0.60, "onions": 0.25}),
    "snacking":    (0.15, {"chips": 0.90, "soft drink": 0.80, "chocolate": 0.35, "juice": 0.15}),
    "burger":      (0.05, {"buns": 0.95, "chicken": 0.60, "cheese": 0.70, "ketchup": 0.70, "mayonnaise": 0.50}),
    "noodles":     (0.06, {"noodles": 0.95, "ketchup": 0.40, "eggs": 0.35}),
    "fruit":       (0.09, {"apples": 0.80, "bananas": 0.85, "yogurt": 0.55, "juice": 0.25}),
    "baby":        (0.05, {"diapers": 0.95, "baby food": 0.75, "milk": 0.65}),
    "cleaning":    (0.08, {"detergent": 0.90, "toilet paper": 0.65, "soap": 0.45}),
    "personal":    (0.07, {"shampoo": 0.85, "toothpaste": 0.80, "soap": 0.65}),
    "baking":      (0.04, {"flour": 0.95, "sugar": 0.80, "eggs": 0.80, "butter": 0.70}),
}

# Items that are bought on their own now and then (weight = how common)
RANDOM_ITEMS = {
    "milk": 10, "bread": 8, "eggs": 6, "bananas": 5, "onions": 4, "tomatoes": 4,
    "yogurt": 4, "soft drink": 3, "chips": 3, "biscuits": 3, "chocolate": 3,
    "rice": 3, "cooking oil": 2, "salt": 2, "potatoes": 3, "apples": 2,
    "juice": 2, "cheese": 2, "butter": 2, "chicken": 2, "fish": 3, "cake": 2,
    "soap": 1, "toothpaste": 1, "sugar": 2, "tea": 1, "coffee": 1,
}

def make_basket():
    basket = set()
    for _, (prob, items) in PURPOSES.items():
        if random.random() < prob:
            for item, p in items.items():
                if random.random() < p:
                    basket.add(item)
    # 0-3 random extra items
    for _ in range(random.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0]):
        basket.add(random.choices(list(RANDOM_ITEMS), weights=RANDOM_ITEMS.values())[0])
    return basket

rows = []
start = date(2024, 1, 1)
while len(rows) < N_TRANSACTIONS:
    b = make_basket()
    if len(b) >= 2:                        # a visit has at least 2 items
        items = sorted(b)
        random.shuffle(items)              # bills are not alphabetical
        rows.append((start + timedelta(days=random.randint(0, 364)), items))

rows.sort(key=lambda r: r[0])
with open("data/grocery_transactions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Transaction_ID", "Date", "Items"])
    for i, (d, items) in enumerate(rows, start=1):
        w.writerow([f"T{i:04d}", d.strftime("%d-%m-%Y"), ", ".join(items)])
print("Created data/grocery_transactions.csv with", len(rows), "transactions")
