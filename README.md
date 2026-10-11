# Monthly Stock Aggregation & Technical Indicators

Converts 2 years of daily stock data (10 tickers) into monthly OHLC summaries with technical indicators, and writes one CSV per ticker. Built with **Python + pandas only** (no third-party technical-analysis libraries).

## Project structure

| data/
│   ── output_file.csv        # input: daily data, 10 tickers (5030 rows)
├── src/
│   ├── check.py               # verify download and integrity of input csv file
│   └── pipeline.py            # runs the full pipeline (resamples to monthly, calculates and adds indicators, generates output files)
├── output/
│   └── result_{SYMBOL}.csv    # 10 files  with 24 rows each
├── requirements.txt
└── README.md


Calculation logic and file-writing logic live in separate functions/modules.


## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          
pip install -r requirements.txt
python src/pipeline.py              
```

Output: `output/result_AAPL.csv`, `result_AMD.csv`, `result_AMZN.csv`, `result_AVGO.csv`, `result_CSCO.csv`, `result_MSFT.csv`, `result_NFLX.csv`, `result_PEP.csv`, `result_TMUS.csv`, `result_TSLA.csv`.

## Approach

1. **Load and validate** the daily CSV, parse dates, sort by `ticker, date`.
2. **Resample to monthly per ticker** (`groupby` + month grouping):
   - `open` = first trading day's open of the month
   - `high` = max of daily highs
   - `low` = min of daily lows
   - `close` = last trading day's close of the month
   - `start_date` / `end_date` = first / last trading day in the month
3. **Compute indicators on the monthly close** .
4. **Partition** by ticker and write `result_{SYMBOL}.csv`.


## Assumptions

1. **Month boundaries use trading days.** "Open" is the first trading day's open and "Close" is the last trading day's close of each calendar month, not an average.
2. **Data quality.** The input was checked for nulls, duplicate rows and identical row counts across tickers. No imputation was needed.
3. **Prices used.** OHLC uses the raw `open/high/low/close` columns. `adjclose` is not used for indicators.
4. **Volume.** monthly volume = sum of all daily volumes of the month
5. **Warm-up periods produce NaN.** With only 24 monthly points, a window of `n` gives `n - 1` leading empty values. These are left blank rather than filled or back-filled, to avoid inventing data. (SMA/EMA 20 has values only for the last 5 months)
6. **Window for Bollinger, Z-score & Donchian is 10 months**, since the assignment does not specify a window, using 20 would leave only 5 usable rows out of 24, hence 10 was used.
7. **Bollinger multiplier k = 1.9** as we are using 10 periods instead of standard 20 so k has been reduced from conventional 2 (as recommended in bollinger band rules : https://www.bollingerbands.com/bollinger-band-rules)
8. **Dates.** Output keeps `start_date` and `end_date` (first/last trading day) as the only date columns.


## Output columns

ticker, start_date, end_date, open, high, low, close, volume, sma_10, sma_20, ema_10, ema_20, bb_mid_10, bb_std_10, bb_upper_10, bb_lower_10, zscore_10, dc_upper_10, dc_lower_10, dc_mid_10

## Notes on design

- Fully vectorized pandas (`groupby`, `rolling`, `ewm`); no row-by-row loops.
- Functions for calculation (take a DataFrame, return a DataFrame) are kept separate from functions that touch the filesystem, so the logic is easy to test.
- Re-running the script overwrites the output files, so it is idempotent.
