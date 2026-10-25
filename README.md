# MACD Mean-Reversion System: research summary

**Verdict: REJECT.** This research tested whether MACD extremes on 11 liquid US ETFs mean-revert profitably after realistic costs. They don't. The A5 base rule loses money before any parameter search. The best of 12,000 searched variations does not survive multiple-testing control or the sealed out-of-sample test. An ML sizing overlay doesn't rescue it. The prop-firm route is both inadmissible and non-viable.

> Research artifact only. Not financial advice, no guaranteed performance, not production code. Live trading is out of scope.

Everything lives in **[`research.ipynb`](research.ipynb)**, a single self-contained notebook with no external Python modules. Every number below comes from its executed outputs (run 2026-10-25, config hash shown in Section 0).

---

## Repository layout

```
research.ipynb            all code, config, assumptions, results and the viability report (Sections 0–7)
data/events/events.csv    scheduled macro/FOMC event calendar (rebuilt from public sources if missing)
data/cache/nb/            derived data cache (daily bars, yfinance/FRED downloads); safe to delete
```

**Raw inputs** (outside the repo): ETF 1-minute bars in `/Users/andrew/main/trading/data/etfs/{T}_clean_1min.csv` and NQ futures 1-minute bars in `/Users/andrew/main/trading/data/NQ.1m.OHLCV.data/`. The notebook downloads yfinance daily data (adjusted closes, ^VIX, `=F` futures) and FRED DTB3/VIXCLS once and caches them.

**How to run:** open the notebook and *Run All*.
- With a warm cache it takes about 9 minutes; from scratch, about 35.
- Set `RECOMPUTE_DATA = True` to rebuild all derived data.
- Set `FAST_MODE = True` for a roughly 2-minute smoke run at reduced sizes. Its results are not for reporting.

Requirements: Python 3.13, numpy, pandas, numba, scipy, scikit-learn, statsmodels, plotly, yfinance, pyarrow, requests, bs4 (bs4 only to rebuild events).

---

## Pipeline / workflow

| § | Stage | What it does |
|---|---|---|
| 0 | Assumptions & plan | Inline `CFG` config (single source of truth); assumptions A1–A15; data contract; decision gates G0–G7; leakage checklist; environment and seed report |
| 1 | Data import & audit | 1-min → daily RTH bars. Splice repair: the vendor's pitrading (≤2022-03-04) and IEX (≥2022-03-07) segments are adjusted on different bases, so prices are rebased to the yfinance total-return close. Bad-bar replacement, IEX high/low union, audit table, VIX cross-check, event calendar, NQ front-month build |
| 2 | Baseline MACD (IS only) | Causal indicators, numba backtest engine (signal at close t, fill at open t+1, ATR stops, 1% risk, costs/borrow/financing), no-look-ahead perturbation test, base 12/26/9 backtest, bucket analysis (day-of-week, month, VIX regime, SPY trend, volume regime, event window, event impact), event-effect permutation tests |
| 3 | Parameter search | 12,000 random variations simulated **only up to 2018-12-31**. Selection on train (≤2014) and validation (2015–18) with neighbourhood-shrunk Sharpe → 5 candidates **registered (hashed) before any test access**. Then Deflated Sharpe, White RC / Hansen SPA, CSCV PBO, Holm/BH, one-at-a-time stability, seed stability, cost sensitivity, and a placebo suite (random signals, joint shuffled returns, lagged signals, random parameters, random event flags). Finally the sealed test 2019-01-02..2026-09-21 |
| 4 | Multi-split stress | IS 1–5 years and first-half splits, with IS-only re-selection (procedure test) vs the fixed primary. 10,000-path Monte Carlo (trade bootstrap and stationary block bootstrap), Newey–West t-tests, SPY buy-and-hold comparison |
| 5 | ML overlay | Random forest (out / base / 1.5× size) on base-strategy signals. Triple-barrier R labels, purged + embargoed expanding walk-forward (2008–2026), thresholds chosen on an inner validation year, leakage audit, importance, calibration, confusion matrices. Base vs ML vs SPY backtest with MC and a paired bootstrap; also run on the registered primary |
| 6 | Prop-firm simulation | Topstep 50K/100K/150K Combine rules (PROVISIONAL), ETF → futures mapping (ES, NQ, YM, RTY, GC, SI/SIL, CL, HG, ZB, ZN, ZT), intraday adaptation, contract sizing with caps/fees/slippage, 10,000-path Combine Monte Carlo, IS-chosen / OOS-validated optimisation, ETF↔futures correlation in stress windows |
| 7 | Viability report | Gates, consolidated PASS/FAIL/WARN/BLOCKED checklist, comparison table and a report generated from computed values |

**Sample labels:**
- **IS** = 2004-01-14..2018-12-31 (after the 260-day warm-up).
- **OOS** = 2019-01-02..2026-09-21, the sealed test window.
- **OOS-split** = a multi-split remainder window.

---

## Data findings (Section 1)

- **Splice artifact.** Raw vendor files jump at 2022-03-07 (SPY −5.8%, DIA −7.4%, TLT −14.2%, IEF −12.2%, SHY −11.9%). The two segments are dividend-adjusted as of different dates, so a naive backtest would book these as fake crashes. After rebasing, every splice gap matches yfinance's adjusted gap (SPY −0.148% vs −0.143%).
- **Bad bars.** 240 bars were inconsistent with consolidated data and were replaced by yfinance. Every extreme return (>15%, e.g. USO April 2020) is confirmed by yfinance.
- **IEX volume** covers a single venue, about 1–2.5% of consolidated volume (vs 77–95% for the pitrading segment). Volume features are therefore normalised within each source segment, and they are noisier after 2022.
- **Events.** "myforexfunds" is a defunct prop firm, not a FRED dataset. The calendar was built from federalreserve.gov (FOMC) and ALFRED vintage dates (CPI, NFP, GDP, PCE, Retail Sales, PPI, Jobless Claims). It holds dates and impact tiers only, with no surprise values.

---

## Results

### Baseline 12/26/9, in-sample (Section 2)
| | Sharpe [95% CI] | CAGR | Max DD | Trades | Hit rate | Profit/trade |
|---|---|---|---|---|---|---|
| Base portfolio | −0.347 [−0.83, 0.11] | 0.83% | −2.98% | 1,027 | 47.1% | −0.047% |
| SPY buy & hold | 0.426 [−0.02, 0.90] | 7.62% | −55.2% | — | — | — |

- The base rule has **no IS edge**: NW p = 0.159, and only USO has a positive per-ETF Sharpe.
- **Buckets:** of 104 adequately sized bucket tests, 15 were nominally significant vs 5.2 expected by chance, but **0 survive BH correction**.
- **Events:** no event type changes the edge significantly (permutation p 0.26–0.93, BH q ≥ 0.90). Any event effects are associational only.

### Parameter search & overfitting controls (Section 3)
- 12,000 variations, 1,765 eligible.
- **Registered primary** (id 4115): MACD 10/30/7, `macd_z` z-window 40, entry 1.5 / exit 1.0, 2.5 ATR stop, 4 ATR target, 20-bar max hold, long-only, own-SMA200 aligned, VIX < 30, relvol < 1, entries only in event windows.
- The primary's validation Sharpe was 1.58.

| Control (IS) | Result | Pass? |
|---|---|---|
| Deflated Sharpe (N = 12,000 / effective N ≈ 388) | 0.17 / 0.53 (need ≥ 0.95) | ✗ |
| Hansen SPA consistent p (5 seeds) | 0.63–0.78 | ✗ |
| White Reality Check p | 0.78–0.82 | ✗ |
| CSCV PBO (eligible / all) | 0.73 / 0.39 | ✗ |
| Holm / BH discoveries | 0 / 0 of 11,227 (raw p<0.05: 308 vs 561 expected) | ✗ |
| Seed stability | Alternative universes pick different configs (4–5 of 16 parameters shared). Only long-only and `macd_z` recur | weak |

The top validation Sharpe (1.58) is below what the best of 12,000 null trials is expected to show (≈ 2.03).

### Sealed out-of-sample test, 2019-01-02..2026-09-21
| | Sharpe [95% CI] | CAGR | Ann. excess | Alpha (t) | Beta | Max DD | Hit rate | Profit/trade | Exposure |
|---|---|---|---|---|---|---|---|---|---|
| Registered primary | 0.243 [−0.46, 0.93] | 2.85% | 0.13% | 0.05% (0.27) | 0.005 | −0.52% | 62.1% | 0.107% | 0.04 |
| Base 12/26/9 | −0.271 [−0.90, 0.37] | 2.39% | −0.32% | −0.45% (−1.11) | 0.008 | −2.19% | 50.7% | −0.037% | 0.10 |
| ML-sized base | −0.132 [−0.77, 0.52] | 2.54% | −0.17% | −0.35% (−0.79) | 0.012 | −2.24% | 51.9% | −0.010% | 0.09 |
| SPY buy & hold | 0.796 [0.14, 1.54] | 17.5% | 15.3% | — | 1 | −33.7% | — | — | 1.0 |

CAGRs include T-bill interest on idle cash. Strategy exposure is tiny, so "annual excess" is the relevant return.

- **Primary:** its NW t-test is not significant (t = 0.67, p = 0.50). In Monte Carlo, P(Sharpe ≤ 0) = 0.24 and P(Sharpe above SPY) = 0.10.
- **Placebos (OOS):** the primary does **not** beat random signals (p = 0.63), shuffled returns (p = 0.34), random parameters (p = 0.19) or random event flags (p = 0.37). The OOS WRC / SPA p-values across the 5 candidates plus base are 0.66 / 0.61.
- **Cost sensitivity:** the primary's OOS Sharpe falls from 0.34 at zero cost to −0.14 at 5× cost.
- **IS → OOS rank carry-over:** Spearman 0.16 across the grid, and 0.04 among eligible variations.

### Multi-split stress (Section 4)
| Split (IS → OOS remainder) | Re-selected #1, OOS-split Sharpe | Fixed primary (overlaps selection data) | SPY |
|---|---|---|---|
| IS 1y | 0.05 | 0.52 | 0.56 |
| IS 2y | −0.22 | 0.60 | 0.56 |
| IS 3y | −0.58 (significantly negative) | 0.63 | 0.55 |
| IS 4y | −0.38 | 0.60 | 0.59 |
| IS 5y | 0.35 | 0.61 | 0.81 |
| First half | −0.06 | 0.73 | 0.71 |

- When the selection procedure is re-run on each IS window, its pick is positive in 2 of 6 splits, significant in 0 of 6, and beats SPY in 0 of 6.
- The fixed primary looks good on these windows only because they overlap the data it was selected on.

### ML overlay (Section 5)
- **Classifier:** barely informative. Out-of-fold log loss is 1.07–1.09, *worse* than the class-prior baseline (0.99–1.02). Balanced accuracy is 0.36–0.41, against 0.33 for chance.
- **Base strategy, OOS:** ML Sharpe −0.13 vs base −0.27. The paired Sharpe-difference CI is [−0.07, 0.34] and the NW p-value of the difference is 0.28, so there is **no demonstrated improvement**. In the 2008–18 development folds, ML was slightly worse (−0.22 vs −0.19).
- **Registered primary, OOS:** ML 0.34 vs 0.24, with paired CI [−0.01, 0.21], so not significant.
- **Leakage audit:** all 19 folds pass (purge, embargo, thresholds from the prior year, no shuffling, point-in-time perturbation).

### Prop-firm simulation, Topstep (Section 6, PROVISIONAL rules)
- **Admissibility:** Topstep requires being flat by 15:10 CT, so the swing strategy is **inadmissible**. Only an intraday entry-day adaptation was simulated.
- **Sizing:** at $300–$750 risk per trade, only 1–53% of trades can be sized with listed contracts (micros exist only from 2019; SI and HG caps are 0).
- **Best IS configuration** (primary at $200): 0% pass and ~100% timeout. It barely trades.
- **Base at $750 risk:** 0.01% pass vs 44% MLL failure OOS, losing about $1,400 per year.
- **Gate:** P(pass) ≥ 25% is not met anywhere. Larger accounts (100K/150K) also show 0% pass.
- **Mapping:** equity-index ETFs track their futures closely (QQQ vs local NQ RTH correlation 0.994). Stress breaks were flagged for GLD/GC and USO/CL (2008), SHY/ZT (COVID), and a persistent TLT→ZB duration mismatch (beta about 0.67).

---

## Decision gates

| Gate | Criterion | Outcome |
|---|---|---|
| G0 Data integrity | splice repaired, extreme returns confirmed | **PASS** |
| G1 IS baseline edge | base Sharpe > 0, NW p < 0.05 | **FAIL** (−0.35, p = 0.16) |
| G2 Search significance | DSR ≥ 0.95, SPA p < 0.05, PBO < 0.5, ≥1 Holm | **FAIL** (0.17, 0.64, 0.73, 0) |
| G3 Sealed OOS | primary Sharpe CI lower bound > 0 | **FAIL** (0.24 [−0.46, 0.93]) |
| G4 Multi-split | ≥4/6 significant and ≥4/6 beat SPY | **FAIL** (0/6, 0/6) |
| G5 ML improvement | paired Sharpe-diff CI > 0 | **FAIL** ([−0.07, 0.34]) |
| G6 Prop viability | OOS P(pass) ≥ 25% and > P(fail) | **FAIL** (≈0%) |
| G7 Paper trading | only if G1–G6 pass | **NOT REACHED** |

**Leakage/overfitting checklist highlights:**
- **PASS:** no-look-ahead perturbation test, registration before test access (5 logged accesses), ML purge and embargo audit.
- **FAIL:** survivorship (an ex-post universe of 11 surviving ETFs).
- **WARN:**
  - Costs are assumed, not measured.
  - Fractional shares and no market impact.
  - IEX volume after 2022.
  - Shutdown-delayed event dates.
  - Prop rules are provisional.
  - The test window was already examined by an earlier version of this study.

---

## Viability & recommendations

**REJECT.** Do not paper-trade, fund or submit this to a prop firm. The 2019–2026 window has now been examined several times and is no longer a clean holdout, so any follow-up needs fresh data.

**Failure modes observed:**
- **Regime shifts:** IS Sharpe collapses OOS, and re-selections are unstable.
- **Overfitting and multiple testing:** the DSR, SPA, PBO and Holm results above.
- **Cost drag:** CPER spreads and SHY/IEF at the leverage cap.
- **Correlation breaks** in the futures mapping.
- **Tiny exposure and capacity:** about 4% average gross exposure for the primary.

**Next experiments, in priority order:**
1. Test frozen, pre-registered rules on a **new universe** (sector/country ETFs). Reopen only if the pre-cost spread has a CI > 0.
2. Run a paired comparison against simple mean-reversion rules (N-day low in an uptrend, RSI(2)) to see whether MACD adds anything.
3. Shrink the search to ≤ 20 economically motivated variants, and report DSR including every trial already run.
4. Compute all volume features from consolidated volume.
5. For any prop-firm use, design a native RTH-intraday strategy instead of translating swing trades.

**Paper-trading gates, if a future spec ever passes 1–3:**
- Frozen hashes of code, config and universe.
- Realised costs ≤ assumed after 50 fills.
- Replay parity (0 signal mismatches).
- A kill switch at the Monte Carlo p05 drawdown.
- No size-up until the Sharpe CI lower bound is > 0.
- Human approval before any capital.

---

## Key assumptions (full list in notebook Section 0)

- **Execution:** signal at close t, fill at open t+1. Each trade risks 1% of sleeve equity to an ATR(14) stop, with notional ≤ 2× equity.
- **Costs:** commission + half-spread + slippage in per-ETF bps, plus short borrow and a financing spread. All are assumptions.
- **Portfolio:** equal-weight sleeves; cash earns the 3-month T-bill rate; Sharpe is computed on excess returns.
- **Timing:** VIX close at t and scheduled event dates are treated as known before the open at t+1.
- **Prop rules:** Topstep values transcribed 2026-10-20. They must be verified before any real pass/fail conclusion.
- **Reproducibility:** named, seeded RNG streams (global seed 20261020), so results are deterministic given the cached data.
- **Disclosures:** after the first full execution, two display/robustness fixes were made, and the second execution's numbers are reported:
  - table formatting;
  - the prop optimiser excludes configurations that size no trades.

  Neither fix changes any strategy or selection result.
