
import sys
import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# --- Config & Setup ---
# Project root relative to this file: .../Find_Transform/Branch_example/exp_example/estimator.py
# Root is 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import strategy module from current directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy

from alpha_digging.smoothing import (
    log_move_adaptive_extrema,
    log_move_extrema,
    smooth_close,
)

# Updated Data Configuration using absolute path
DATA_BASE_DIR = Path("/home/liumx/momentum_transformer/qc_workspace/data/crypto/binance/minute")
SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
OUTPUT_DIR = Path(__file__).resolve().parent

START_DATE = "2025-10-01"
END_DATE = "2025-10-30"
WARMUP_DAYS = 1
MIN_AMPLITUDE_PCT = 0.01
RESAMPLE_MIN = 15
DETECTOR_CONFIG = {
    "method": "log_move_adaptive",
    "base_log_move": float(np.log1p(0.01)),
    "lookback": 30,
    "vol_mult": 5.0,
}

# --- Helper Functions (Copied/Adapted from run_peak_analysis.py) ---

def load_minute_data(data_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    COLUMNS = ["ms_offset", "open", "high", "low", "close", "volume"]
    frames = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        zip_path = data_dir / f"{day:%Y%m%d}_trade.zip"
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                day_df = pd.read_csv(fh, header=None, names=COLUMNS)
        base_ts = pd.Timestamp(day.date())
        day_df["timestamp"] = base_ts + pd.to_timedelta(day_df["ms_offset"], unit="ms")
        frames.append(day_df)
    
    if not frames:
        # Fallback for testing if no data
        print(f"Warning: No real data found in {data_dir}. Generating dummy data.")
        dates = pd.date_range(start_date, end_date, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": 100 + np.random.randn(len(dates)).cumsum(),
            "high": 100, "low": 100, "close": 100, "volume": 1000
        })
        df["close"] = df["open"] # Simple
        df["high"] = df["open"] + 1
        df["low"] = df["open"] - 1
        return df

    data = pd.concat(frames, ignore_index=True)
    data.sort_values("timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def resample_ohlcv(data: pd.DataFrame, resample: int) -> pd.DataFrame:
    if resample == 1:
        return data.copy()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return (
        data.set_index("timestamp")[list(agg.keys())]
        .resample(f"{resample}min")
        .agg(agg)
        .dropna()
        .reset_index()
    )

def _filter_by_amplitude(series, peaks, troughs, min_pct, is_log=False):
    if not peaks and not troughs:
        return peaks, troughs
    values = series.to_numpy()
    tagged = [(i, "p") for i in peaks] + [(i, "t") for i in troughs]
    tagged = sorted({(i, t) for i, t in tagged if 0 <= i < len(values)}, key=lambda x: x[0])
    keep = set()
    for (idx1, _), (idx2, _) in zip(tagged, tagged[1:]):
        p1, p2 = values[idx1], values[idx2]
        if is_log:
            move = abs(np.exp(p2 - p1) - 1.0)
        else:
            if p1 == 0: continue
            move = abs(p2 - p1) / abs(p1)
        if move >= min_pct:
            keep.add(idx1)
            keep.add(idx2)
    return [i for i in peaks if i in keep], [i for i in troughs if i in keep]

def _alt_distances_and_anchor(peaks, troughs, timestamps, warmup_end):
    tagged = [(i, "p") for i in peaks] + [(i, "t") for i in troughs]
    tagged = sorted({(i, t) for i, t in tagged if 0 <= i < len(timestamps)}, key=lambda x: x[0])
    if len(tagged) < 2: return np.array([]), []
    
    filtered = []
    last_type = None
    for idx, t in tagged:
        if last_type is not None and t == last_type: continue
        filtered.append((idx, t))
        last_type = t
        
    positions = [idx for idx, _ in filtered if warmup_end is None or timestamps.iloc[idx] >= warmup_end]
    if len(positions) < 2: return np.array([]), []
    
    times = timestamps.iloc[positions].to_numpy()
    deltas = (np.diff(times) / np.timedelta64(1, "m")).astype(float)
    anchor = positions[:-1]
    return deltas, anchor

def _alt_positions_with_types(peaks, troughs, timestamps, warmup_end):
    tagged = [(i, "p") for i in peaks] + [(i, "t") for i in troughs]
    tagged = sorted({(i, t) for i, t in tagged if 0 <= i < len(timestamps)}, key=lambda x: x[0])
    if len(tagged) < 2: return [], []
    
    filtered = []
    last_type = None
    for idx, tp in tagged:
        if last_type is not None and tp == last_type: continue
        filtered.append((idx, tp))
        last_type = tp
        
    if warmup_end is not None:
        filtered = [(idx, tp) for idx, tp in filtered if timestamps.iloc[idx] >= warmup_end]
    if len(filtered) < 2: return [], []
    
    positions = [idx for idx, _ in filtered]
    types = [tp for _, tp in filtered]
    return positions, types

def _rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/period, adjust=False).mean()
    roll_down = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - 100 / (1 + rs)

def compute_factor_series(df_r):
    close = df_r["close"].astype(float)
    high = df_r["high"].astype(float)
    low = df_r["low"].astype(float)
    volume = df_r["volume"].astype(float)
    log_close = np.log(close)
    log_ret = log_close.diff().fillna(0.0)
    
    factors = {}
    
    # Volatility
    vol_map = {}
    for lb in [30, 60]:
        vol = log_ret.rolling(lb, min_periods=2).std().fillna(0.0)
        factors[f"vol_logret_lb{lb}_bps"] = vol * 1e4 + 1e-9
        vol_map[lb] = factors[f"vol_logret_lb{lb}_bps"]

    # Trend
    trend_map = {}
    for lb in [30, 60]:
        mom = (log_close - log_close.shift(lb)).abs().fillna(0.0)
        factors[f"abs_trend_lb{lb}_bps"] = mom * 1e4 + 1e-9
        trend_map[lb] = factors[f"abs_trend_lb{lb}_bps"]

    # Rel Vol
    for lb in [20, 40]:
        ma_vol = volume.rolling(lb, min_periods=1).mean()
        rel_vol = (volume / (ma_vol + 1e-9)).fillna(1.0)
        factors[f"rel_vol_lb{lb}"] = rel_vol + 1e-9
        
    rel_vol20 = factors.get("rel_vol_lb20")
    rel_vol40 = factors.get("rel_vol_lb40")

    # Interactions (trend_x_relvol, etc.)
    if rel_vol20 is not None:
        for lb in [30, 60]:
            if lb in trend_map:
                factors[f"trend_x_relvol_lb{lb}_rv20"] = trend_map[lb] * rel_vol20
            if lb in vol_map:
                factors[f"vol_x_relvol_lb{lb}_rv20"] = vol_map[lb] * rel_vol20

    if rel_vol40 is not None:
        for lb in [30, 60]:
            if lb in trend_map:
                factors[f"trend_x_relvol_lb{lb}_rv40"] = trend_map[lb] * rel_vol40
            if lb in vol_map:
                factors[f"vol_x_relvol_lb{lb}_rv40"] = vol_map[lb] * rel_vol40

    return factors

# --- Main Logic ---

def process_symbol(symbol: str, strategy_str: str) -> Dict:
    print(f"Processing Symbol: {symbol} with Strategy: {strategy_str}")
    data_dir = DATA_BASE_DIR / symbol
    
    # 1. Load Data
    data = load_minute_data(data_dir, START_DATE, END_DATE)
    
    # 2. Peak Detection (Fixed: resample15 | none | log_move_adapt_lb30_m5.0)
    df_r = resample_ohlcv(data, RESAMPLE_MIN)
    ts = df_r["timestamp"]
    log_close = np.log(df_r["close"].astype(float))
    
    # Detector
    peaks, troughs = log_move_adaptive_extrema(
        log_close,
        base_log_move=DETECTOR_CONFIG["base_log_move"],
        lookback=DETECTOR_CONFIG["lookback"],
        vol_mult=DETECTOR_CONFIG["vol_mult"],
    )
    
    # Filter
    peaks, troughs = _filter_by_amplitude(log_close, peaks, troughs, MIN_AMPLITUDE_PCT, is_log=True)
    
    warmup_end = ts.min().normalize() + pd.Timedelta(days=WARMUP_DAYS)
    
    print(f"[{symbol}] Detected {len(peaks)} peaks and {len(troughs)} troughs.")
    
    # 4. Transform & Evaluate
    if "|" in strategy_str:
        parts = strategy_str.split("|")
        factor_name = parts[-1].strip()
    else:
        factor_name = strategy_str.strip()
        
    factors = compute_factor_series(df_r)
    
    if factor_name not in factors:
        print(f"❌ [{symbol}] Factor '{factor_name}' not found.")
        return {"cv": float('inf')}
        
    # Compute CV
    dist, anchor = _alt_distances_and_anchor(peaks, troughs, ts, warmup_end)
    
    if len(dist) < 3:
        print(f"❌ [{symbol}] Insufficient intervals.")
        return {"cv": float('inf')}
        
    # Prepare inputs for strategy
    # Strategy needs factors aligned at anchor points
    # We pass the full arrays sliced by anchor
    factors_aligned = {k: v.to_numpy()[anchor] for k, v in factors.items()}
    
    # --- CALL STRATEGY DIRECTLY ---
    # No scanning. The strategy is the single source of truth for the transformation.
    denominator = strategy.get_denominator(factors_aligned, dist)
    
    # Filter valid results
    mask = np.isfinite(dist) & np.isfinite(denominator) & (denominator > 1e-9)
    
    if mask.sum() < 3:
        print(f"❌ [{symbol}] Strategy returned insufficient valid denominators.")
        return {"cv": float('inf')}
        
    ratio_vals = dist[mask] / denominator[mask]
    anchor_used = np.array(anchor)[mask]
    
    mu = np.mean(ratio_vals)
    std = np.std(ratio_vals, ddof=1)
    cv = std / mu if mu else float('inf')
    
    print(f"✅ [{symbol}] Strategy CV = {cv:.4f}")
    
    best_series = {
        "factor": factor_name,
        "cv": cv,
        "ratio": ratio_vals,
        "anchor_idx": anchor_used
    }
    
    return {
        "cv": cv,
        "df_r": df_r,
        "peaks": peaks,
        "troughs": troughs,
        "warmup_end": warmup_end,
        "best_series": best_series
    }


def run_analysis(strategy_str):
    print(f"Running Multi-Symbol Analysis for Strategy: {strategy_str}")
    
    results_map = {}
    cv_list = []
    
    for symbol in SYMBOLS:
        res = process_symbol(symbol, strategy_str)
        if res["cv"] != float('inf'):
            results_map[symbol] = res
            cv_list.append(res["cv"])
    
    if not cv_list:
        print("❌ No valid results for any symbol.")
        return float('inf')
        
    avg_cv = np.mean(cv_list)
    print(f"\n📊 Final Average CV: {avg_cv:.4f}")
    
    # Plotting
    _plot_multi_untransformed(results_map, OUTPUT_DIR / "best_untransformed.png")
    _plot_multi_transformed(results_map, OUTPUT_DIR / "best_transformed.png")
    
    return avg_cv

def _plot_multi_untransformed(results_map, filename):
    symbols = list(results_map.keys())
    n = len(symbols)
    if n == 0: return
    
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1: axes = [axes]
    
    for ax, symbol in zip(axes, symbols):
        res = results_map[symbol]
        df_r = res["df_r"]
        peaks = res["peaks"]
        troughs = res["troughs"]
        warmup_end = res["warmup_end"]
        
        ts = df_r["timestamp"]
        log_close = np.log(df_r["close"].astype(float))
        
        ax.plot(ts, log_close, color="gray", alpha=0.6, label="Log Close")
        
        p_idx = [i for i in peaks if ts.iloc[i] >= warmup_end]
        t_idx = [i for i in troughs if ts.iloc[i] >= warmup_end]
        
        ax.scatter(ts.iloc[p_idx], log_close.iloc[p_idx], color="red", marker="^", label="Peaks")
        ax.scatter(ts.iloc[t_idx], log_close.iloc[t_idx], color="green", marker="v", label="Troughs")
        ax.set_title(f"{symbol.upper()} - Untransformed")
        ax.legend(loc="upper right")
        
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)
    print(f"Saved {filename}")

def _plot_multi_transformed(results_map, filename):
    symbols = list(results_map.keys())
    n = len(symbols)
    if n == 0: return
    
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1: axes = [axes]
    
    for ax, symbol in zip(axes, symbols):
        res = results_map[symbol]
        df_r = res["df_r"]
        peaks = res["peaks"]
        troughs = res["troughs"]
        warmup_end = res["warmup_end"]
        best_series = res["best_series"]
        
        ts = df_r["timestamp"]
        log_close = np.log(df_r["close"].astype(float))
        
        positions, types = _alt_positions_with_types(peaks, troughs, ts, warmup_end)
        if len(positions) < 2: continue
        
        anchor_idx = best_series["anchor_idx"]
        ratio_vals = best_series["ratio"]
        
        pos_map = {pos: i for i, pos in enumerate(positions)}
        
        seg_starts = []
        seg_ends = []
        seg_types_end = []
        ratios = []
        
        for a, r in zip(anchor_idx, ratio_vals):
            if a in pos_map:
                idx = pos_map[a]
                if idx + 1 < len(positions):
                    seg_starts.append(positions[idx])
                    seg_ends.append(positions[idx+1])
                    seg_types_end.append(types[idx+1])
                    ratios.append(r)
        
        if not ratios: continue

        xs = [0.0]
        ys = [log_close.iloc[seg_starts[0]]]
        labs = [types[0]]
        
        for r, end_idx, t_end in zip(ratios, seg_ends, seg_types_end):
            x_next = xs[-1] + r
            xs.append(x_next)
            ys.append(log_close.iloc[end_idx])
            labs.append(t_end)
            
        ax.plot(xs, ys, color="steelblue", alpha=0.8)
        
        for x, y, t in zip(xs, ys, labs):
            c = "red" if t == "p" else "green"
            m = "^" if t == "p" else "v"
            ax.scatter(x, y, color=c, marker=m, s=20)
            
        ax.set_title(f"{symbol.upper()} - Transformed (CV={best_series['cv']:.3f})")
        
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)
    print(f"Saved {filename}")

if __name__ == "__main__":
    # Test run
    run_analysis("alt | trend_x_relvol_lb60_rv40")