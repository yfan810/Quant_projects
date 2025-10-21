# ETF Pair Trading

A minimal, production-style template to test a pair trading strategy on two ETFs.

## Quick Start

1.  `pip install -r requirements.txt`
2.  Open `notebooks/etf_pair_trading_minimal.ipynb`
3.  Run all cells.

## Strategy

-   Regress A on B (rolling) to estimate hedge ratio.
-   Build spread = A - beta\*B, compute z-score.
-   Entry: z \> +2 short A/long B; z \< −2 long A/short B.
-   Exit: \|z\| ≤ 0.5.
-   Include transaction cost & slippage.
