"""Data cleaning and preprocessing pipeline for the raw shipment data.

Every step logs what it changed, so the report can quote real numbers.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

log = {}
raw = pd.read_csv("data/raw_shipments.csv")
truth = pd.read_csv("data/ground_truth.csv")
df = raw.copy()

# ---------- 0. audit ----------
log["rows_raw"] = len(df)
log["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()
log["duplicates_before"] = int(df.duplicated().sum())
log["shipping_mode_labels_before"] = int(df.shipping_mode.nunique())
log["city_labels_before"] = int(df.city.nunique())

# ---------- 1. duplicates ----------
df = df.drop_duplicates().reset_index(drop=True)
log["rows_after_dedupe"] = len(df)

# ---------- 2. text standardisation ----------
def norm(s):
    return " ".join(str(s).lower().replace("-", " ").split())

mode_map = {"std class": "Standard Class", "standard class": "Standard Class",
            "second class": "Second Class", "first class": "First Class", "same day": "Same Day"}
city_map = {"new delhi": "Delhi", "bombay": "Mumbai", "bangalore": "Bengaluru",
            "madras": "Chennai", "calcutta": "Kolkata"}
df["shipping_mode"] = df.shipping_mode.map(lambda s: mode_map[norm(s)])
df["city"] = df.city.map(lambda s: np.nan if pd.isna(s) else city_map.get(norm(s), norm(s).title()))
log["shipping_mode_labels_after"] = int(df.shipping_mode.nunique())
log["city_labels_after"] = int(df.city.nunique())

# ---------- 3. dates (three formats) ----------
def parse_date(x):
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"):
        try:
            return pd.to_datetime(x, format=f)
        except ValueError:
            continue
    return pd.NaT
df["order_date"] = df.order_date.map(parse_date)
log["unparsed_dates"] = int(df.order_date.isna().sum())

# ---------- 4. units: grams -> kg ----------
g_mask = df.weight_unit.str.lower().eq("g")
log["weights_in_grams_fixed"] = int(g_mask.sum())
df.loc[g_mask, "weight"] = df.loc[g_mask, "weight"] / 1000
df = df.drop(columns="weight_unit").rename(columns={"weight": "weight_kg"})

# ---------- 5. impossible values (business rules) ----------
bad_days = (df.days_real < 0) | (df.days_real > 30)
log["impossible_days_real"] = int(bad_days.sum())
df.loc[bad_days, "days_real"] = np.nan

# ---------- 6. outliers: IQR rule on unit_price, then cap ----------
q1, q3 = df.unit_price.quantile([.25, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
z = (df.unit_price - df.unit_price.mean()) / df.unit_price.std()
log["price_outliers_iqr"] = int(((df.unit_price < lo) | (df.unit_price > hi)).sum())
log["price_outliers_z3"] = int((z.abs() > 3).sum())
log["price_cap_upper"] = round(float(hi), 1)
df["unit_price"] = df.unit_price.clip(lo, hi)

# ---------- 7. missing values ----------
log["missing_filled"] = {
    "weight_kg": int(df.weight_kg.isna().sum()), "days_real": int(df.days_real.isna().sum()),
    "discount": int(df.discount.isna().sum()), "city": int(df.city.isna().sum())}
df["weight_kg"] = df.weight_kg.fillna(df.groupby("category").weight_kg.transform("median"))
df["days_real"] = df.days_real.fillna(df.groupby("shipping_mode").days_real.transform("median"))
df["discount"] = df.discount.fillna(df.discount.median())
df["city"] = df.city.fillna("Unknown")
df["order_total"] = (df.quantity * df.unit_price * (1 - df.discount)).round(2)

# ---------- 8. feature engineering ----------
df["order_month"] = df.order_date.dt.month
df["order_dow"] = df.order_date.dt.dayofweek
df["late"] = (df.days_real > df.days_scheduled).astype(int)
log["rows_final"] = len(df)
log["missing_after"] = int(df.isna().sum().sum())
df.to_csv("outputs/cleaned_shipments.csv", index=False)

# ---------- 9. normalisation + encoding (model-ready) ----------
ml = df[["unit_price", "weight_kg", "quantity", "discount", "days_scheduled",
         "shipping_mode", "category", "late", "days_real"]].copy()
for c in ["unit_price", "weight_kg"]:
    ml[c + "_mm"] = (ml[c] - ml[c].min()) / (ml[c].max() - ml[c].min())
ml["quantity_z"] = (ml.quantity - ml.quantity.mean()) / ml.quantity.std()
ml = pd.get_dummies(ml, columns=["shipping_mode", "category"], dtype=int)
ml.to_csv("outputs/model_ready.csv", index=False)

# ---------- 10. does cleaning matter? compare two pipelines ----------
feat = ["days_scheduled", "weight_kg", "quantity", "unit_price"]
tr = truth.set_index("order_id")
_, test_ids = train_test_split(tr.index, test_size=.25, random_state=4)
test = tr.loc[test_ids]

naive = raw.drop_duplicates().rename(columns={"weight": "weight_kg"})
for c in ["weight_kg", "days_real", "unit_price", "days_scheduled", "quantity"]:
    naive[c] = pd.to_numeric(naive[c], errors="coerce")
naive = naive.dropna(subset=feat + ["days_real"])
naive = naive[~naive.order_id.isin(test_ids)]
clean = df[~df.order_id.isin(test_ids)]

def score(train):
    m = LinearRegression().fit(train[feat], train.days_real)
    p = m.predict(test[feat])
    return [round(r2_score(test.days_real, p), 3), round(mean_absolute_error(test.days_real, p), 2)]

log["naive_rows_kept"] = len(naive); log["clean_rows_kept"] = len(clean)
log["naive_r2_mae"] = score(naive); log["clean_r2_mae"] = score(clean)

# ---------- charts ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
miss = raw.isna().mean().mul(100)[lambda s: s > 0]
ax[0].bar(miss.index, miss.values, color="#c05621"); ax[0].set(title="Missing values in raw data (%)")
ax[1].boxplot([raw.unit_price.dropna(), df.unit_price], tick_labels=["raw", "cleaned"])
ax[1].set(title="Unit price: before vs after capping")
ax[2].boxplot([pd.to_numeric(raw.days_real, errors="coerce").dropna(), df.days_real],
              tick_labels=["raw", "cleaned"]); ax[2].set(title="Days to deliver: before vs after")
plt.tight_layout(); plt.savefig("outputs/quality_before_after.png", dpi=130)

json.dump(log, open("outputs/cleaning_log.json", "w"), indent=2, default=str)
print(json.dumps(log, indent=2, default=str))
