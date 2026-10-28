"""EDA and visualisation for the simulated logistics dataset.

Produces six charts (outputs/fig*.png) and a stats file (outputs/stats.json).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", font_scale=1.0)
df = pd.read_csv("data/shipments.csv", parse_dates=["date"])
S = {}

# ---------- 1. descriptive statistics ----------
num = ["distance_km", "weight_kg", "units", "dwell_hours", "transit_hours", "total_cost", "cost_per_km"]
desc = df[num].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
desc.to_csv("outputs/descriptive_stats.csv")
S["n"] = len(df)
S["desc"] = desc.to_dict("index")
S["on_time_pct"] = round(df.on_time.mean() * 100, 1)
S["mode_share_pct"] = (df["mode"].value_counts(normalize=True) * 100).round(1).to_dict()

# ---------- 2. correlations ----------
corr = df[num].corr().round(2)
S["corr_cost"] = corr["total_cost"].drop("total_cost").to_dict()
S["corr_transit_dwell"] = float(corr.loc["transit_hours", "dwell_hours"])

# ---------- Fig 1: monthly trend ----------
m = df.groupby("month").agg(shipments=("shipment_id", "count"), avg_cost=("total_cost", "mean"),
                            on_time=("on_time", "mean")).reset_index()
S["monthly"] = m.round(2).to_dict("list")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(m.month, m.shipments, color="#90cdf4", label="Shipments")
ax.set(xlabel="Month (2025)", ylabel="Shipments", xticks=range(1, 13))
ax2 = ax.twinx(); ax2.plot(m.month, m.avg_cost, color="#c05621", marker="o", label="Avg cost (Rs)")
ax2.set_ylabel("Average cost per shipment (Rs)"); ax2.grid(False)
ax.set_title("Shipment volume and average cost by month")
fig.legend(loc="upper left", bbox_to_anchor=(.08, .88)); plt.tight_layout()
plt.savefig("outputs/fig1_monthly_trend.png", dpi=140); plt.close()

# ---------- Fig 2: distributions ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df.transit_hours, bins=40, kde=True, color="#2b6cb0", ax=ax[0])
ax[0].axvline(df.transit_hours.mean(), color="red", ls="--", label="mean")
ax[0].axvline(df.transit_hours.median(), color="green", ls="--", label="median"); ax[0].legend()
ax[0].set(title="Transit time distribution", xlabel="hours")
sns.boxplot(data=df, x="mode", y="total_cost", hue="mode", palette="Set2", legend=False, ax=ax[1])
ax[1].set(title="Shipment cost by transport mode", xlabel="", ylabel="Cost (Rs)")
plt.tight_layout(); plt.savefig("outputs/fig2_distributions.png", dpi=140); plt.close()
S["mode_cost"] = df.groupby("mode").total_cost.agg(["mean", "median"]).round(0).to_dict("index")
S["mode_cost_per_km"] = df.groupby("mode").cost_per_km.mean().round(2).to_dict()

# ---------- Fig 3: correlation heatmap ----------
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", ax=ax)
ax.set_title("Correlation matrix of key variables")
plt.tight_layout(); plt.savefig("outputs/fig3_correlation.png", dpi=140); plt.close()

# ---------- Fig 4: distance vs cost ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="total_cost",
                hue="mode", alpha=.5, s=18, ax=ax)
for md, c in zip(["Road", "Rail", "Air"], ["#3182ce", "#dd6b20", "#38a169"]):
    d = df[df["mode"] == md]
    k, b = np.polyfit(d.distance_km, d.total_cost, 1)
    xs = np.linspace(d.distance_km.min(), d.distance_km.max(), 50)
    ax.plot(xs, k * xs + b, color=c, lw=2)
    S.setdefault("slope_rs_per_km", {})[md] = round(float(k), 2)
ax.set(title="Distance vs cost, with trend line per mode", xlabel="Distance (km)", ylabel="Cost (Rs)")
plt.tight_layout(); plt.savefig("outputs/fig4_distance_cost.png", dpi=140); plt.close()

# ---------- Fig 5: bottlenecks (hub dwell + carrier on-time) ----------
hub = df.groupby("hub").dwell_hours.mean().round(2)
car = (df.groupby("carrier").on_time.mean() * 100).round(1)
S["hub_dwell"] = hub.to_dict(); S["carrier_on_time"] = car.to_dict()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
hub.plot.bar(ax=ax[0], color=["#a0aec0" if h != hub.idxmax() else "#e53e3e" for h in hub.index])
ax[0].set(title="Average hub dwell time", ylabel="hours", xlabel=""); ax[0].tick_params(axis="x", rotation=0)
car.plot.bar(ax=ax[1], color=["#a0aec0" if c != car.idxmin() else "#e53e3e" for c in car.index])
ax[1].set(title="On-time delivery by carrier", ylabel="% on time", xlabel=""); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("outputs/fig5_bottlenecks.png", dpi=140); plt.close()

# ---------- Fig 6: weekday x hub dwell heatmap ----------
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pv = df.pivot_table(index="weekday", columns="hub", values="dwell_hours", aggfunc="mean").reindex(order)
S["weekday_dwell"] = pv.mean(axis=1).round(2).to_dict()
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
ax.set(title="Average dwell time (hours) by weekday and hub", ylabel="", xlabel="")
plt.tight_layout(); plt.savefig("outputs/fig6_dwell_heatmap.png", dpi=140); plt.close()

# ---------- peak season effect ----------
df["season"] = np.where(df.month >= 10, "Oct-Dec peak", "Jan-Sep")
S["season"] = df.groupby("season").agg(on_time=("on_time", lambda s: round(s.mean() * 100, 1)),
        dwell=("dwell_hours", "mean"), cost=("total_cost", "mean")).round(1).to_dict("index")
a = df[df.season == "Oct-Dec peak"].transit_hours; b = df[df.season != "Oct-Dec peak"].transit_hours
t, p = stats.ttest_ind(a, b, equal_var=False)
S["peak_vs_rest_transit"] = {"peak_mean": round(a.mean(), 1), "rest_mean": round(b.mean(), 1),
                             "t": round(float(t), 1), "p_value": float(p)}

# ---------- cost drivers (standardised regression) ----------
X = pd.get_dummies(df[["distance_km", "weight_kg", "dwell_hours", "mode"]], columns=["mode"], drop_first=True, dtype=int)
Xs = StandardScaler().fit_transform(X)
lr = LinearRegression().fit(Xs, df.total_cost)
S["cost_driver_r2"] = round(lr.score(Xs, df.total_cost), 3)
S["cost_driver_std_coef"] = {c: round(float(v), 0) for c, v in zip(X.columns, lr.coef_)}

# cost breakdown by mode
brk = df.groupby("mode")[["fuel_cost", "handling_cost", "labour_cost", "toll_cost"]].mean()
S["cost_share_pct"] = (brk.div(brk.sum(axis=1), axis=0) * 100).round(1).to_dict("index")

json.dump(S, open("outputs/stats.json", "w"), indent=2, default=float)
print(json.dumps(S, indent=2, default=float))
