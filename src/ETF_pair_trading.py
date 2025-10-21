import pandas as pd
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict

def _hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    return float(model.params[1])

def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd

def generate_signals(a: pd.Series, b: pd.Series, lookback: int = 60,
                     entry_z: float = 2.0, exit_z: float = 0.5) -> pd.DataFrame:
    betas = []
    for i in range(len(a)):
        if i < lookback:
            betas.append(np.nan)
        else:
            betas.append(_hedge_ratio(a.iloc[i-lookback+1:i+1], b.iloc[i-lookback+1:i+1]))
    beta_s = pd.Series(betas, index=a.index)

    spread = a - beta_s * b
    z = zscore(spread, window=lookback)
    long_entry = z < -entry_z
    short_entry = z > entry_z
    exit_signal = z.abs() <= exit_z

    pos = np.zeros(len(a))
    for i in range(1, len(a)):
        if long_entry.iloc[i]: pos[i] = 1
        elif short_entry.iloc[i]: pos[i] = -1
        elif exit_signal.iloc[i]: pos[i] = 0
        else: pos[i] = pos[i-1]
    pos_s = pd.Series(pos, index=a.index)

    return pd.DataFrame({"a": a, "b": b, "beta": beta_s, "spread": spread, "z": z, "pos": pos_s})

def backtest(df: pd.DataFrame, tc_bps: float = 2.0, slippage_bps: float = 1.0):
    ra = df["a"].pct_change().fillna(0.0)
    rb = df["b"].pct_change().fillna(0.0)
    beta = df["beta"].fillna(method="ffill").fillna(1.0)

    pair_ret = df["pos"] * (ra - beta * rb)
    pos_shift = df["pos"].shift(1).fillna(0.0)
    turnover = (df["pos"] - pos_shift).abs()
    cost = turnover * ((tc_bps + slippage_bps) / 10000.0)

    net_ret = pair_ret - cost
    equity = (1 + net_ret).cumprod().rename("equity")

    ann_mean = float(net_ret.mean() * 252)
    ann_vol = float(net_ret.std() * np.sqrt(252))
    sharpe = float(ann_mean / ann_vol) if ann_vol > 0 else float("nan")
    mdd = float((equity / equity.cummax() - 1.0).min())
    win_rate = (net_ret[net_ret != 0] > 0).mean() if (net_ret != 0).sum() > 0 else float("nan")

    metrics = {
        "ann_mean_return_%": ann_mean * 100,
        "ann_vol_%": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_%": mdd * 100,
        "turnover_per_day": float(turnover.mean()),
        "win_rate_%": float(win_rate * 100 if win_rate == win_rate else np.nan)
    }
    return equity, metrics
