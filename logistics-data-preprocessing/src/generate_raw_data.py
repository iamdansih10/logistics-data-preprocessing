"""Creates a deliberately messy shipment dataset.

Reference: the structure mirrors the public DataCo Smart Supply Chain dataset
(order date, shipping mode, scheduled vs real shipping days, category, market),
but this file is SIMULATED so the project runs without any download. A clean
"ground truth" copy is saved too, only so we can measure how well the
cleaning worked - in real life you would not have that.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 3000

modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
sched_days = {"Standard Class": 5, "Second Class": 3, "First Class": 2, "Same Day": 0}
cities = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Meerut"]
cats = ["Clothing", "Sports", "Electronics"]

mode = rng.choice(modes, N, p=[.5, .2, .2, .1])
city = rng.choice(cities, N)
cat = rng.choice(cats, N, p=[.4, .35, .25])
qty = rng.integers(1, 6, N)
price = np.round(rng.gamma(3, 600, N) + 150, 2)
disc = np.round(rng.choice([0, .05, .1, .15, .2, .3], N), 2)
weight = np.round(rng.gamma(2, 1.2, N) + 0.2, 2)
sched = np.array([sched_days[m] for m in mode])
real = np.clip(np.round(sched + 0.35 * weight + rng.normal(0, 1, N)), 0, None).astype(int)
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
total = np.round(qty * price * (1 - disc), 2)

truth = pd.DataFrame({"order_id": np.arange(10001, 10001 + N), "order_date": dates,
    "shipping_mode": mode, "city": city, "category": cat, "quantity": qty,
    "unit_price": price, "discount": disc, "weight_kg": weight,
    "days_scheduled": sched, "days_real": real, "order_total": total})
truth["late"] = (truth.days_real > truth.days_scheduled).astype(int)
truth.to_csv("data/ground_truth.csv", index=False)

raw = truth.copy()
# --- messy dates: three different formats ---
fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"], N, p=[.5, .3, .2])
raw["order_date"] = [d.strftime(f) for d, f in zip(dates, fmt)]
# --- inconsistent text ---
mv = {"Standard Class": ["standard class", "STANDARD CLASS", "Standard  Class ", "Std Class"],
      "Second Class": ["second class", "Second class "], "First Class": ["first class", "FIRST CLASS"],
      "Same Day": ["same day", "Same-Day"]}
cv = {"Delhi": ["delhi", "New Delhi", " Delhi"], "Mumbai": ["mumbai", "Bombay"],
      "Bengaluru": ["Bangalore", "bangalore ", "bengaluru"], "Chennai": ["chennai", "Madras"],
      "Kolkata": ["kolkata", "Calcutta"], "Hyderabad": ["hyderabad "], "Pune": ["pune"], "Meerut": ["meerut", "MEERUT"]}
for col, mp in (("shipping_mode", mv), ("city", cv)):
    for i in rng.choice(N, int(N * 0.25), replace=False):
        raw.at[i, col] = rng.choice(mp[raw.at[i, col]])
# --- mixed units: ~6% of weights stored in grams ---
raw["weight_unit"] = "kg"
g = rng.choice(N, int(N * .06), replace=False)
raw.loc[g, "weight_kg"] = raw.loc[g, "weight_kg"] * 1000
raw.loc[g, "weight_unit"] = "g"
raw = raw.rename(columns={"weight_kg": "weight"})
# --- impossible / outlier values ---
bad = rng.choice(N, 30, replace=False)
raw.loc[bad[:20], "days_real"] = rng.integers(90, 366, 20)
raw.loc[bad[20:], "days_real"] = -rng.integers(1, 6, 10)
raw.loc[rng.choice(N, 15, replace=False), "unit_price"] *= 100
# --- missing values ---
for col, p in (("weight", .06), ("days_real", .03), ("city", .02), ("discount", .04)):
    raw.loc[rng.choice(N, int(N * p), replace=False), col] = np.nan
# --- exact duplicate rows ---
raw = pd.concat([raw, raw.sample(85, random_state=1)], ignore_index=True)
raw = raw.drop(columns=["late"]).sample(frac=1, random_state=3).reset_index(drop=True)
raw.to_csv("data/raw_shipments.csv", index=False)
print("raw:", raw.shape)
