# Logistics Data Cleaning & Preprocessing Pipeline (Week 2)

Week 2 of my Logistics Data Analyst internship: take messy shipment data and turn
it into something an analyst can actually trust. This builds on my Week 1 project,
[logistics-delivery-analysis](https://github.com/iamdansih10/logistics-delivery-analysis).

> **Data note:** the dataset is simulated. Its columns (order date, shipping mode,
> scheduled vs real shipping days, category) are modelled on the public
> [DataCo Smart Supply Chain dataset](https://data.mendeley.com/datasets/8gx2fvg2k6/5),
> and I deliberately injected typical real-world problems into it.

## Problems injected, and how each is handled

| Problem | Fix |
|---|---|
| 85 duplicate rows | `drop_duplicates()` |
| 14 spellings of 4 shipping modes, 24 spellings of 8 cities | strip / lower-case / mapping dictionary |
| Dates in 3 different formats | try each format in turn with `pd.to_datetime` |
| ~6% of weights stored in grams | unit column used to convert to kg |
| Impossible delivery days (negative, 90+) | business rule, set to missing |
| Extreme unit prices | IQR rule, capped (not deleted) |
| Missing weight, days, discount, city | group median / overall median / "Unknown" |
| Features on different scales | min-max and z-score scaling, one-hot encoding |

## Run it

```bash
pip install -r requirements.txt
python src/generate_raw_data.py
python src/preprocess.py
```

Outputs land in `outputs/`: `cleaned_shipments.csv`, `model_ready.csv`,
`cleaning_log.json` and a before/after chart.

## Key result

A basic delivery-time model trained on naively processed data scored R² = -0.12.
The same model on data cleaned by this pipeline scored R² = 0.75 (error cut from
1.86 to 0.83 days).

## Limitations

- Simulated data, so real data will be messier.
- `ground_truth.csv` exists only because the data is synthetic; real projects don't get an answer key.
- Date parsing assumes day-first formats; a real dataset could mix day-first and month-first.
- The IQR rule flagged more prices than I injected, so some legitimate expensive orders were capped.
- City and mode mappings are hand-written and will not scale without fuzzy matching.
