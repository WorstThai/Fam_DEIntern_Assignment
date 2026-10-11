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

    # 1. Shape: 10 tickers and 24 months
    assert df.shape[0] == 240, f"expected 240 rows, got {df.shape[0]}"

    # 1. SMA & EMA checks
    # NaNs per ticker: sma_10/ema_10 = 9, sma_20/ema_20 = 19
    nans = df.groupby("ticker")[["sma_10", "ema_10", "sma_20", "ema_20"]].apply(lambda g: g.isna().sum())
    assert (nans[["sma_10", "ema_10"]] == 9).all().all(), "sma_10/ema_10 NaN count != 9"
    assert (nans[["sma_20", "ema_20"]] == 19).all().all(), "sma_20/ema_20 NaN count != 19"

    # EMA starts with SMA (check for AAPL on row 9)
    aapl = df[df.ticker == "AAPL"].reset_index(drop=True)
    assert np.isclose(aapl.loc[9, "ema_10"], aapl.loc[9, "sma_10"]), "ema_10 does not start with sma_10"

    # Manual EMA calculation:
    k = 2 / (10 + 1)
    manual_ema = (aapl.loc[10, "close"] - aapl.loc[9, "ema_10"]) * k + aapl.loc[9, "ema_10"]
    assert np.isclose(manual_ema, aapl.loc[10, "ema_10"]), "manual ema_10 mismatch"

    # 2. Bollinger Bands: 9 NaNs per ticker
    bb_nans = df.groupby("ticker")[["bb_mid_10", "bb_std_10", "bb_upper_10", "bb_lower_10"]].apply(lambda g: g.isna().sum())
    assert (bb_nans == 9).all().all(), "bollinger NaN count != 9"

    # 3. Z-score: check for infinite values
    assert np.isinf(df["zscore_10"]).sum() == 0, "infinite z-score found"

    # 4. Donchian Channels
    valid = df.dropna(subset=["dc_upper_10"])  #removing NaNs for temporary checking
    assert (valid["dc_upper_10"] >= valid["high"]).all(), "dc_upper_10 < high"
    assert (valid["dc_lower_10"] <= valid["low"]).all(), "dc_lower_10 > low"
    assert (df.groupby("ticker")["dc_upper_10"].count() == 15).all(), "donchian non-NaN count not 15"

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
    print(f"Verified output : {len(paths)} files, {expected_rows} rows each")


def run_pipeline(input_path: str) -> pd.DataFrame:
    cdf = load_data('data/output_file.csv')     # loading data
    cdf = resample_monthly(cdf)                 # resamping from daily to monthly
    cdf = add_moving_averages(cdf)              # add sma and ema
    cdf = bollinger_bands(cdf)                  # add bollinger bands
    cdf = zscore(cdf)                           # add z-score
    cdf = donchian_channels(cdf)                # add donchian channels
    cdf = reorder_columns(cdf)                  # reorder columns for consistency
    
    return cdf


def main() -> None:
    root = Path(__file__).resolve().parent.parent    # pipeline.py to src to repo root
    cdf = run_pipeline(str(root/'data/output_file.csv'))

    checks(cdf)

    paths = write_partitions(cdf, str(root/'output'))
    verify_outputs(paths)


if __name__ == "__main__":
    main()


