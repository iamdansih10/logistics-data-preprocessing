# Advanced Data Analysis & Visualization in Logistics (Week 3)

Week 3 of my Logistics Data Analyst internship. This project explores a simulated
year of shipment data and uses Python (pandas, matplotlib, seaborn, scipy) to
find out what drives cost, where delays come from, and how peak season changes
performance. Earlier weeks:
[Week 1](https://github.com/iamdansih10/logistics-delivery-analysis) and
[Week 2](https://github.com/iamdansih10/logistics-data-preprocessing).

> **Data note:** the data is simulated (3,821 shipments, 19 columns). I built some
> patterns in on purpose (peak season, one slow carrier, one congested hub), so
> the findings show the *method*; they are not facts about a real company.

## Run it

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/analysis.py
```

## What it produces

| Output | Purpose |
|---|---|
| `fig1_monthly_trend.png` | Volume and average cost over the year |
| `fig2_distributions.png` | Shape of transit time; cost by mode |
| `fig3_correlation.png` | Which variables move together |
| `fig4_distance_cost.png` | Distance vs cost, trend line per mode |
| `fig5_bottlenecks.png` | Hub dwell time and carrier on-time rate |
| `fig6_dwell_heatmap.png` | Dwell time by weekday and hub |
| `descriptive_stats.csv`, `stats.json` | Mean, median, std, skew, correlations, model results |

## Headline findings

- Distance is the strongest cost driver (correlation 0.81 with cost).
- Peak season (Oct-Dec) doubles volume and drops on-time delivery from 78.5% to 55.4%.
- Hub 3 has about twice the dwell time of other hubs; Carrier C is on time only 54.7% of the time.
- Mondays have the longest dwell times (8.6 h vs about 6.7-7.2 h on other days).

## Limitations

Simulated data; correlation is not causation; a linear cost model is a simplification;
no real fuel, weather or traffic data; results depend on the patterns I chose to build in.
