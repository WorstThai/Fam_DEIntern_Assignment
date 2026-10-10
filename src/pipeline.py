from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 24


def load_data(path: str) -> pd.DataFrame: 
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_date"] = df["date"] 
    monthly = (
        df.groupby(["ticker", pd.Grouper(key="date", freq="ME")])
        .agg(
            start_date=("trade_date", "first"),
            end_date=("trade_date", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            adjclose=("adjclose", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .drop(columns="date") 
    )
    return monthly


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()                  #directly calculates sma


def ema(close: pd.Series, n: int) -> pd.Series:
    if len(close) < n:
        return pd.Series(np.nan, index=close.index) #if total rows lesser than n

    seeded = pd.Series(np.nan, index=close.index)
    seeded.iloc[n - 1] = close.iloc[:n].mean()      # nth entry = mean of first n
    seeded.iloc[n:] = close.iloc[n:]                # real closes after first n
    return seeded.ewm(alpha=2 / (n + 1), adjust=False).mean() #ewm directly calculates ema


def add_moving_averages(df: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    df = df.sort_values(["ticker", "start_date"]).reset_index(drop=True)    #sorting jic
    grouped = df.groupby("ticker")["close"]         
    for n in windows:
        df[f"sma_{n}"] = grouped.transform(lambda s: sma(s, n))             #sma_10 and sma_20
        df[f"ema_{n}"] = grouped.transform(lambda s: ema(s, n))             #ema_10 and ema_20
    return df


def bollinger_bands(df, n=10, k=1.9):
    grouped = df.groupby("ticker")["close"]

    df[f"bb_mid_{n}"] = grouped.transform(lambda s: s.rolling(n).mean())
    df[f"bb_std_{n}"] = grouped.transform(lambda s: s.rolling(n).std())
    df[f"bb_upper_{n}"] = df[f"bb_mid_{n}"] + k * df[f"bb_std_{n}"]
    df[f"bb_lower_{n}"] = df[f"bb_mid_{n}"] - k * df[f"bb_std_{n}"]
    return df


def zscore(df, n=10):
    df[f"zscore_{n}"] = (df["close"] - df[f"bb_mid_{n}"]) / df[f"bb_std_{n}"]
    return df


def donchian_channels(df, n=10):
    df[f"dc_upper_{n}"] = df.groupby("ticker")["high"].transform(lambda s: s.rolling(n).max())
    df[f"dc_lower_{n}"] = df.groupby("ticker")["low"].transform(lambda s: s.rolling(n).min())
    df[f"dc_mid_{n}"] = (df[f"dc_upper_{n}"] + df[f"dc_lower_{n}"]) / 2
    return df


def reorder_columns(df):
    COLUMN_ORDER = [
    "ticker", "start_date", "end_date", "open", "high", "low", "close", "volume",
    "sma_10", "sma_20", "ema_10", "ema_20",
    "bb_mid_10", "bb_std_10", "bb_upper_10", "bb_lower_10",
    "zscore_10",
    "dc_upper_10", "dc_lower_10", "dc_mid_10"
    ]
    return df[COLUMN_ORDER]


def checks(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    # 1. SMA & EMA
    # df = add_moving_averages(df)
    print("sma & ema shape:", df.shape)  # expected:(240, 13)

    # Check: NaNs per ticker
    # Expected: sma_10 & ema_10 = 9 NaNs, sma_20 & ema_20 = 19 NaNs
    # print(df.groupby("ticker")[["sma_10", "ema_10", "sma_20", "ema_20"]]
    #          .apply(lambda g: g.isna().sum()))

    # setting up one ticker (AAPL) for row level checks on ema and sma
    aapl = df[df.ticker == "AAPL"].reset_index(drop=True)

    # Check: EMA if starts with the SMA, so they match on the first countable (non NaN) row (month 10)
    print("ema sarts with sma (row 9) : ",
          np.isclose(aapl.loc[9, "ema_10"], aapl.loc[9, "sma_10"]))

    # Check: manually checking ema calculation
    # EMA_t = (close_t - EMA_{t-1}) * k + EMA_{t-1}, with k = 2 / (n + 1)
    k = 2 / (10 + 1)
    manual_ema = (aapl.loc[10, "close"] - aapl.loc[9, "ema_10"]) * k + aapl.loc[9, "ema_10"]
    print("ema manual calculation : ", np.isclose(manual_ema, aapl.loc[10, "ema_10"]))

 
    # 2. Bollinger Bands
    # df = bollinger_bands(df)
    print("bollinger shape:", df.shape)

    # Check: NaNs per ticker
    # Expected: bb_mid_10, bb_std_10, bb_upper_10, bb_lower_10 -> 9 NaNs each
    # print(df.groupby("ticker")[["bb_mid_10", "bb_std_10", "bb_upper_10", "bb_lower_10"]]
    #          .apply(lambda g: g.isna().sum()))



    # 3. Z-score
    # df = zscore(df)
    print("z-score shape:", df.shape)

    # Check: no infinite z-scores (guards against division by a zero std)
    # Expected: 0
    # print("z-score infinity counter : ", np.isinf(df["zscore_10"]).sum())


    
    # 4. Donchian Channels
    # df = donchian_channels(df)
    print("donchian shape:", df.shape)

    # Check: channel bounds contain the current month's high/low
    # (the rolling window includes the current month)
    # Expected: True, True
    # valid = df.dropna(subset=["dc_upper_10"])
    # print("donchian upper lower : ")
    # print((valid["dc_upper_10"] >= valid["high"]).all())
    # print((valid["dc_lower_10"] <= valid["low"]).all())

    # Check: number of non-NaN channel values per ticker
    # Expected: 15 per ticker
    # print(df.groupby("ticker")["dc_upper_10"].count())

    print(df)

    return df


def write_partitions(df: pd.DataFrame, out_dir: str = "output") -> list[Path]:

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written = []
    for symbol, group in df.groupby("ticker", sort=True):
        if len(group) != EXPECTED_ROWS:
            raise ValueError(
                f"{symbol}: expected {EXPECTED_ROWS} rows, got {len(group)}"
            )

        file_path = out_path / f"result_{symbol}.csv"
        group.sort_values("start_date").to_csv(file_path, index=False)
        written.append(file_path)

    return written


def verify_outputs(paths: list[Path], expected_rows: int = EXPECTED_ROWS) -> None:
    assert len(paths) == 10, f"Expected 10 files, found {len(paths)}"
    for p in paths:
        n = len(pd.read_csv(p))
        assert n == expected_rows, f"{p.name}: {n} rows, expected {expected_rows}"
    print(f"Verified {len(paths)} files, {expected_rows} rows each")


def run_pipeline(input_path: str) -> pd.DataFrame:
    cdf = load_data('data/output_file.csv')     # loading data
    cdf = resample_monthly(cdf)                 # resamping from daily to monthly
    cdf = add_moving_averages(cdf)              # add sma and ema
    cdf = bollinger_bands(cdf)                  # add bollinger bands
    cdf = zscore(cdf)                           # add z-score
    cdf = donchian_channels(cdf)                # add donchian channels
    cdf = checks(cdf)                           # perform checks on all operations
    cdf = reorder_columns(cdf)                  # reorder columns for consistency
    write_partitions(cdf, 'output')             # writing 10 csv files, one for each ticker
    
    return cdf


def main() -> None:
    root = Path(__file__).resolve().parent.parent    # pipeline.py to src to repo root
    cdf = run_pipeline(str(root/'data/output_file.csv'))

    paths = write_partitions(cdf, str(root/'output'))
    verify_outputs(paths)


if __name__ == "__main__":
    main()


