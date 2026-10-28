"""Simulates a year of shipment records for a hypothetical logistics network.

Key variables (as the task asks): delivery/transit times, shipment volumes
(weight, units) and transportation costs. Patterns are built in on purpose:
  * festive-season volume peak (Oct-Dec)
  * cost driven by distance, weight, mode and a rising fuel price
  * one carrier (Carrier C) and one hub (Hub 3) that cause most delays
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
N = 6000

date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
    rng.choice(365, N, p=None), unit="D")
# seasonal weighting: more shipments Oct-Dec
w = np.where(date.month >= 10, 1.9, np.where(date.month.isin([3, 4]), 0.9, 1.0))
keep = rng.random(N) < (w / w.max())
date = date[keep]; N = len(date)

mode = rng.choice(["Road", "Rail", "Air"], N, p=[.62, .26, .12])
carrier = rng.choice(["Carrier A", "Carrier B", "Carrier C", "Carrier D"], N, p=[.32, .28, .22, .18])
hub = rng.choice(["Hub 1", "Hub 2", "Hub 3", "Hub 4", "Hub 5"], N, p=[.22, .2, .24, .18, .16])

base_dist = {"Road": 450, "Rail": 900, "Air": 1400}
distance = np.array([max(30, rng.gamma(2.2, base_dist[m] / 2.2)) for m in mode]).round(0)
weight = (rng.gamma(2.0, 180, N) + 15).round(1)
units = np.maximum(1, (weight / rng.uniform(2, 8, N)).round()).astype(int)

speed = {"Road": 45, "Rail": 38, "Air": 400}        # km/h average door to door
month = date.month.values
fuel_price = 88 + 0.9 * month + rng.normal(0, 1.2, N)  # Rs/litre, drifting upwards
dwell = rng.gamma(2.0, 2.0, N) + 1                      # hours at hub
dwell *= np.where(hub == "Hub 3", 2.1, 1.0)
dwell *= np.where(month >= 10, 1.35, 1.0)
dwell = dwell.round(1)
dow = date.dayofweek.values
dwell *= np.where(dow == 0, 1.25, 1.0)                  # Monday backlog
dwell = dwell.round(1)

transit = distance / np.array([speed[m] for m in mode]) + dwell + rng.normal(0, 1.5, N)
transit = np.clip(transit, 2, None)
carrier_delay = np.where(carrier == "Carrier C", 5.5, 0) + np.where(month >= 10, 2.5, 0)
promised = distance / np.array([speed[m] for m in mode]) + 10
actual = transit + carrier_delay * rng.random(N)
on_time = (actual <= promised).astype(int)

rate = {"Road": 5.8, "Rail": 3.9, "Air": 21.0}          # Rs per km base
fuel = distance * np.array([rate[m] for m in mode]) * (fuel_price / 95) * 0.45
handling = weight * 1.6 + units * 4
labour = distance * 0.9 + dwell * 55
tolls = np.where(mode == "Road", distance * 0.55, 0)
other = rng.normal(0, 120, N)
cost = np.clip(fuel + handling + labour + tolls + other, 150, None)

df = pd.DataFrame({"shipment_id": np.arange(1, N + 1), "date": date, "month": month,
    "weekday": date.day_name().values, "mode": mode, "carrier": carrier, "hub": hub,
    "distance_km": distance, "weight_kg": weight, "units": units,
    "dwell_hours": dwell, "transit_hours": actual.round(1), "on_time": on_time,
    "fuel_cost": fuel.round(0), "handling_cost": handling.round(0),
    "labour_cost": labour.round(0), "toll_cost": tolls.round(0),
    "total_cost": cost.round(0)})
df["cost_per_km"] = (df.total_cost / df.distance_km).round(2)
df.to_csv("data/shipments.csv", index=False)
print(df.shape)
