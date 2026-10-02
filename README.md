# Market Basket Analysis using Apriori (Python + Streamlit)

Finds which grocery products are bought together **in one visit to the store**,
using the Apriori algorithm (written from scratch, no special library needed),
and shows everything in a simple web UI.

## Dataset (data/grocery_transactions.csv)
One row = one visit (one bill). 1500 visits, 41 products.

| Transaction_ID | Date       | Items                              |
|----------------|------------|------------------------------------|
| T0001          | 01-01-2024 | butter, bread, cereal, jam, biscuits |
| T0002          | 01-01-2024 | cake, eggs, bananas                |

The data is synthetic but realistic: customers shop with a purpose (breakfast,
curry night, tea time, baby care ...) so related items appear together,
plus a few random extra items in each bill.
`generate_dataset.py` creates it (`python generate_dataset.py`).

## Project structure
```
market_basket_apriori/
├── app.py                 # Streamlit user interface
├── apriori.py             # Apriori algorithm + association rules
├── generate_dataset.py    # script that created the dataset
├── requirements.txt
├── data/grocery_transactions.csv
└── README.md
```

## How to run in VS Code
1. Unzip the folder and open it in VS Code (File -> Open Folder).
2. Open the terminal (Ctrl + `) and run: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
   (if `streamlit` is not recognised: `python -m streamlit run app.py`)
4. Your browser opens at http://localhost:8501

## What the app shows
| Tab | Content |
|---|---|
| Dataset overview | visits, products, average basket size, all transactions (with product search), top products, basket-size chart |
| Frequent itemsets | items / combinations that pass the minimum support |
| Association rules | rules with support, confidence, lift + plain-English sentences + chart |
| Recommendations | choose a product, see what is bought with it |

Use the sidebar sliders to change minimum support, confidence and lift.
You can also upload your own CSV with columns `Transaction_ID, Items` (items separated by commas).

## Key terms
- **Basket**: all items bought in one visit.
- **Support**: % of visits containing the itemset.
- **Confidence**: of visits with A, % that also contain B.
- **Lift**: confidence / support(B); above 1 means A and B are bought together more than by chance.
