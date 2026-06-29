# ✈️ Predicting US Domestic Flight Departure Delays

> **Beyond the Schedule: Integrating Weather and the Knock-On Effect for Superior Delay Prediction**

A rigorous data science report built in R Markdown that moves from a scheduled-features-only benchmark to a multi-source, causally motivated model stack — demonstrating that **the two dominant physical drivers of departure delays are inbound aircraft performance and origin weather**, not the flight schedule itself.

*Authors: Erel Edry, Idan Hazan, Nadav Zvulun, Reem Mizrahi — Introduction to Data Science*

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Key Features & Analysis](#-key-features--analysis)
- [Data Source](#-data-source)
- [Prerequisites & Dependencies](#-prerequisites--dependencies)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Key Insights & Results](#-key-insights--results)

---

## 📖 Project Description

Flight delays cost the US economy upward of **$28 billion annually** (Ball et al., *Total Delay Impact Study*, NEXTOR 2010) through crew overtime, fuel burn, gate occupancy, and the downstream burden of missed connections. Critically, delays do not stay local: a single disrupted flight propagates across a hub-and-spoke network, cascading through later legs flown by the same aircraft — the well-documented **knock-on (domino) effect**.

This project addresses a core limitation of prior delay-prediction work: models built exclusively on *scheduled* features (carrier, distance, time-of-day) are bounded by a hard data ceiling. Our scheduled-only OLS baseline explains roughly **2–3% of delay variance** (held-out R²), not because the model is misspecified, but because the two most powerful physical drivers — **inbound aircraft arrival performance** and **origin airport weather** — are simply absent from that feature set.

### Core Objectives

- **Quantify** the marginal predictive lift from adding NOAA weather variables and knock-on features on top of a scheduled-only baseline.
- **Model delay occurrence** (binary: > 15 min) via logistic regression and **delay magnitude** (continuous) via OLS and a Gamma GLM with log link.
- **Characterise the domino curve** — the non-linear, accelerating relationship between inherited inbound lateness and outbound delay risk.
- **Address OLS assumptions** explicitly through residual diagnostics, log-transformation of the skewed knock-on predictor, interaction terms, and a variance-stabilising GLM.

---

## 🔬 Key Features & Analysis

### 1. Hypothesis-Driven Framework
The analysis is structured around four explicit causal hypotheses:
- **H1 — Carrier Type:** Low-cost carriers, operating with tighter turnaround buffers, exhibit higher baseline delay rates than legacy majors.
- **H2 — Time-of-Day Accumulation:** Delay probability increases through the day as schedule slack is exhausted (morning → evening gradient).
- **H3 — Adverse Weather:** Bad weather at the origin (wind, precipitation, present-weather codes) independently inflates departure delays.
- **H4 — Knock-On Dominance:** Inbound arrival performance is the single strongest predictor, dwarfing all scheduled features.

### 2. Three-Variable Knock-On Encoding
Rather than a simple binary flag, the inbound effect is captured by three complementary variables:
- **`prev_arr_delay`** — inbound arrival delay in minutes (continuous); set to 0 for first-leg flights.
- **`has_prev_leg`** — binary flag distinguishing "no inbound constraint" from "on-time inbound" (separates structural zeros from true zeros).
- **`Prev_Leg`** — three-level factor: *First leg of day / Inbound on-time / Inbound late (>15 min)*.

### 3. NOAA Weather Integration
Daily NOAA summaries are joined to the DOT records by origin airport and date:
- `Bad_Weather` — binary flag derived from `ORIGIN_PRESENT_WEATHER_CODES`.
- `ORIGIN_WIND_SPEED_KNOTS` — continuous crosswind/headwind proxy.
- `TMAX`, `TMIN`, `PRCP` — daily temperature range and precipitation.

### 4. Model Stack (Four Models)
| Model | Target | Notes |
|---|---|---|
| **Base OLS** | `DEPARTURE_DELAY` (continuous) | Scheduled features only — clean benchmark |
| **Logistic Regression** | `is_delayed` (binary) | Full feature set; threshold calibrated to empirical delay rate |
| **Extended OLS** | `DEPARTURE_DELAY` | Log-transformed knock-on + three two-way interactions; mean-centered for interpretability |
| **Gamma GLM (log link)** | `DEPARTURE_DELAY` (shifted) | Handles right skew and heteroscedasticity; multiplicative coefficients |

### 5. Log-Transformation & Interaction Effects
`prev_arr_delay` has a long positive tail and structural zeros; it is shifted by a constant $c$ and log-transformed to compress the tail and linearise the diminishing marginal impact of each extra inherited minute. Three interaction terms are tested:
- `log_prev_arr_delay × Carrier_Type` — does knock-on propagate differently by carrier?
- `log_prev_arr_delay × Bad_Weather` — does adverse weather compound inbound lateness?
- `Carrier_Type × Bad_Weather` — differential carrier weather vulnerability.

### 6. Residual Diagnostics
- **Residuals vs. Fitted** (Fig 5) — LOESS trend near zero; heteroscedastic fan-out (bounded below, unbounded above).
- **Histogram of Residuals** (Fig 6) — strong right skew; normal prediction intervals too narrow in the tail (point estimates remain valid via CLT).

### 7. Calibrated Classification Threshold
With a base delay rate well below 50%, the logistic classifier threshold is fixed at the **empirical delay rate** rather than the naive 0.5 cutoff, which would label almost every flight as on-time.

### 8. Feature Importance (Standardised Coefficients)
OLS is re-fit on z-standardised inputs to produce scale-free coefficients. `prev_arr_delay` dominates all other predictors by a wide margin; evening time-blocks form a second tier; weather variables cluster mid-tier; distance and carrier type are weakest — consistent with delays being a ground-side phenomenon.

---

## 🗄️ Data Source

### Primary Dataset — 2015 DOT Flight On-Time Performance
- **Provider:** US Department of Transportation (DOT) Bureau of Transportation Statistics
- **Scope:** ~5.8 million domestic US flights operated in 2015
- **Standard:** Flights delayed by more than **15 minutes** past scheduled departure are classified as delayed (standard DOT threshold)
- **Exclusions:** Cancellations are dropped (structurally distinct from delays; `DEPARTURE_DELAY` is missing by design)
- **Analysis sample:** A uniform random sample of **200,000 complete-case** flights (complete across all model features)

**Key scheduled features used from DOT:**

| Feature | Type | Description |
|---|---|---|
| `DEPARTURE_DELAY` | Continuous | Actual minus scheduled departure (minutes) |
| `SCHEDULED_DEPARTURE` | Integer (HHMM) | Scheduled departure time |
| `DISTANCE` | Continuous | Route distance in miles |
| `AIRLINE` | Categorical | Full airline name |
| `MONTH` | Factor | Calendar month (1–12) |
| `DAY_OF_WEEK` | Factor | Day of week (1–7) |
| `PREV_LEG_ARR_DELAY` | Continuous | Arrival delay of the inbound aircraft leg |

### Secondary Dataset — NOAA Daily Weather Summaries
- **Provider:** National Oceanic and Atmospheric Administration (NOAA)
- **Join key:** Origin airport × calendar date
- **Features:** `ORIGIN_PRESENT_WEATHER_CODES`, `ORIGIN_WIND_SPEED_KNOTS`, `TMAX`, `TMIN`, `PRCP`

### Engineered Features

| Feature | Description |
|---|---|
| `is_delayed` | Binary target: 1 if `DEPARTURE_DELAY > 15 min` |
| `Time_Block` | Four-level ordered factor derived from `SCHEDULED_DEPARTURE` |
| `Carrier_Type` | Binary: *Major* (6 legacy carriers) vs *Low-Cost* |
| `Bad_Weather` | Binary: 1 if `ORIGIN_PRESENT_WEATHER_CODES` is non-missing |
| `prev_arr_delay` | Knock-on: inbound delay in minutes; 0 for first legs |
| `has_prev_leg` | Binary flag separating first-leg zeros from true on-time zeros |
| `Prev_Leg` | Three-level factor: *First leg / On-time inbound / Late inbound* |

---

## 📦 Prerequisites & Dependencies

### R Version
- **Recommended:** R ≥ 4.2.0
- **Renderer:** RStudio ≥ 2022.07 (for knitting to PDF) or `rmarkdown` ≥ 2.20

### Required R Packages

```r
install.packages(c(
  "tidyverse",   # Data manipulation and ggplot2 visualisation
  "patchwork",   # Composing multi-panel figures (Figs 2, 5-6)
  "knitr",       # kable() tables and chunk options
  "rmarkdown",   # Document rendering
  "scales"       # Axis label formatting (percent_format)
))
```

### System Dependencies (PDF output only)

The document knits to **PDF via XeLaTeX**. Ensure a TeX distribution is installed:

```bash
# On Ubuntu/Debian
sudo apt-get install texlive-xetex texlive-fonts-recommended

# On macOS (via Homebrew)
brew install --cask mactex

# Via R (cross-platform, minimal install)
install.packages("tinytex")
tinytex::install_tinytex()
```

---

## 🗂️ Project Structure

The R Markdown file follows a linear analytical narrative across four named sections and one appendix:

```
log_try_nadav.Rmd
│
├── Setup chunk            — Global knitr options, library loading, random seed (42)
│
├── data-load chunk        — Raw CSV ingestion, feature engineering, 200K sampling
│
├── modelling chunk        — Train/test split (80/20), Base OLS, Logistic Regression,
│                            ROC/AUC, classification metrics
│
├── Section 1 — Introduction
│   └── Motivation, knock-on background, baseline benchmark (OLS R²)
│
├── Section 2 — Data Overview
│   └── Sampling rationale, delay definition, feature group descriptions
│
├── Section 3 — Methods and Results
│   ├── fig-importance     — Feature importance bar chart (scaled coefficients)
│   ├── fig-roc-domino     — Side-by-side ROC curve + Domino Effect curve (Figs 2–3)
│   └── metrics-table      — Logistic regression held-out metrics (Table 1)
│
├── Model Development: Advanced Modeling & Goodness-of-Fit (Appendix)
│   ├── ext-ols            — Log transform, mean-centering, Extended OLS + interactions
│   │                        → Table 2: key interaction coefficients
│   ├── s5-diagnostics     — Residuals vs Fitted + Histogram (Figs 5–6)
│   ├── s5-glm-dynamic     — Gamma GLM with log link → Table 3: GLM coefficients
│   └── benchmark-table    — Three-model comparison: R², RMSE, MAE (Table 4)
│
└── Section 4 — Limitations and Future Work
    └── Missing features, temporal scope, sub-daily weather, future model directions
```

### Data Directory

```
data/
└── flight_nadav_with_weather.csv   # Pre-joined DOT + NOAA dataset (not included in repo)
```

> **Note:** The source CSV (`data/flight_nadav_with_weather.csv`) must be present in the working directory before knitting. It contains the pre-joined DOT 2015 flight records and NOAA daily weather summaries.

---

## ▶️ How to Run

### Option 1 — RStudio (Recommended)

1. **Clone or download** this repository and open the project in RStudio.

2. **Place the data file** in a `data/` subdirectory of the project root:
   ```
   project-root/
   ├── log_try_nadav.Rmd
   └── data/
       └── flight_nadav_with_weather.csv
   ```

3. **Install dependencies** (one-time):
   ```r
   install.packages(c("tidyverse", "patchwork", "knitr", "rmarkdown", "scales"))
   ```

4. **Knit the document:**
   - Open `log_try_nadav.Rmd` in RStudio.
   - Click the **Knit** button (or press `Ctrl+Shift+K` / `Cmd+Shift+K`).
   - Select **Knit to PDF** for the final formatted report.

### Option 2 — Command Line

```r
# From the R console, with the working directory set to the project root:
rmarkdown::render(
  input       = "log_try_nadav.Rmd",
  output_format = "pdf_document",
  output_file   = "flight_delay_report.pdf"
)
```

```bash
# Or from the terminal using Rscript:
Rscript -e "rmarkdown::render('log_try_nadav.Rmd', output_format='pdf_document')"
```

### Reproducibility

All stochastic operations use **`set.seed(42)`**, applied at three points:
1. Global setup (chunk options).
2. The 200K uniform random sample from the cleaned pool.
3. The 80/20 train/test split.

Running the document end-to-end with the same CSV will produce bit-identical figures, tables, and inline statistics.

---

## 📊 Key Insights & Results

### Finding 1 — The Scheduled-Only Ceiling

> A baseline OLS model using only distance, time-of-day, day-of-week, and carrier type explains just **~2–3% of delay variance** on the held-out test set (R² ≈ 0.02–0.03). This is not a modelling failure — it reflects a fundamental data gap: the two dominant physical drivers are absent.

### Finding 2 — Knock-On Dominance (H4 Confirmed)

The feature importance analysis on z-standardised inputs shows `prev_arr_delay` with a scaled coefficient **several times larger** than any other predictor. The mechanism is intuitive: because each departure depends on the timely arrival of the inbound aircraft, a late inbound compresses the turnaround window, and any shortfall converts directly into an outbound delay.

> **The Domino Curve (Figure 3):** First-leg flights (no inbound constraint) have a baseline delay risk of ~X%. That risk rises steeply and non-linearly with inherited inbound lateness — approximately **30 minutes of inbound delay roughly doubles baseline risk**, with the curve accelerating toward near-certain delay beyond ~60–90 minutes.

### Finding 3 — Carrier Type Differential (H1 Confirmed)

Low-cost carriers exhibit a meaningfully higher delay rate than major legacy carriers. The logistic model returns a **negative coefficient on `Carrier_TypeMajor`**, consistent with larger scheduled turnaround buffers at legacy carriers absorbing disruption more effectively.

### Finding 4 — Weather as an Independent Signal (H3 Confirmed)

Wind speed carries a **positive coefficient** across both OLS and GLM — crosswind and headwind conditions create pushback/runway delays. Low-cost carriers absorb a **larger weather penalty** (confirmed by the `Carrier_Type × Bad_Weather` interaction), consistent with their tighter schedules leaving less buffer to absorb weather-driven ground time.

### Finding 5 — Evening Accumulation (H2 Confirmed)

Evening time-blocks form the second-strongest predictor tier after knock-on features, reflecting cumulative schedule slippage as the day progresses — the later the departure block, the higher the inherited delay burden across the network.

### Model Performance Summary

| Model | R² | RMSE (min) | MAE (min) |
|---|---|---|---|
| Base OLS (scheduled only) | ~0.02–0.03 | — | — |
| Extended OLS (log + interactions) | Significant lift | Lower | Lower |
| **GLM (Gamma, log link)** | **Best** | **Lowest** | **Lowest** |

> The GLM wins on all three held-out regression metrics. Its theoretical motivation — Gamma's variance ∝ mean² aligning with the heteroscedastic fan-out observed in residual diagnostics — is partially attenuated by the additive shift applied to keep the response positive, but the empirical improvement is unambiguous.

**Logistic Regression — Classification Metrics (20% hold-out):**

| Accuracy | AUC | Precision | Recall | MSE |
|---|---|---|---|---|
| Reported dynamically in knitted output | > 0.631 baseline | Reported dynamically | Reported dynamically | Reported dynamically |

> The AUC improvement over the 0.631 scheduled-only baseline confirms that weather and knock-on features contribute **independent, non-redundant predictive signal** for delay occurrence.

### Key Methodological Contributions

- **Mean-centering of `log_prev_arr_delay`** before fitting interaction terms, so main effects (e.g., `Bad_Weather`) remain interpretable at the *average* level of inbound lateness rather than at zero — resolving a negative-coefficient artifact in the uncentered model.
- **Calibrated classification threshold** at the empirical delay rate (not 0.5), ensuring the classifier is optimised for the actual class imbalance in the data.
- **Same-data model comparison** — all three regression models are evaluated on the identical 200K hold-out, making Table 4 a clean apples-to-apples benchmark rather than figures carried over from different pipelines.

### Limitations & Future Directions

- Airport ground congestion and ATC ground-stops are absent from DOT records.
- The 2015-only scope misses multi-year drift and network evolution.
- NOAA weather features are **daily summaries** — sub-daily timing of conditions is not captured.
- **Future work:** ensemble non-linear models (Random Forest, XGBoost) to capture higher-order weather–network interactions; multi-year data; hourly METAR weather joined to exact departure times; FAA ASPM congestion features.

---

## 📚 References

- Ball, M. et al. (2010). *Total Delay Impact Study*. NEXTOR, University of Maryland.
- AhmadBeygi, S., Cohn, A., Guan, Y., & Belobaba, P. (2008). Analysis of the potential for delay propagation in passenger airline networks. *Journal of Air Transport Management*, 14(5), 221–236.

---

*Report rendered with R Markdown · PDF via XeLaTeX · All results are fully reproducible with `set.seed(42)`*
