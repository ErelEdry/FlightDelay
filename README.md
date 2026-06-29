# Predicting Flight Departure Delays

Final project for Introduction to Data Science.
Erel Edry, Idan Hazan, Nadav Zvulun, Reem Mizrahi.

All code is in `log_try_nadav.Rmd` (knits to PDF).

## What the code does

The Rmd runs the whole analysis in order:

- Reads `flight_nadav_with_weather.csv`, which is the 2015 DOT flight dataset
  pre-joined with NOAA daily weather by origin airport and date. Drops
  cancelled flights (missing `DEPARTURE_DELAY`) and any row with a missing
  value across the model features, then draws a uniform 200,000-flight sample.
- Builds the features: `is_delayed` (DEPARTURE_DELAY > 15 min, the DOT
  threshold), `Time_Block` (Morning / Afternoon / Evening / Night from the
  scheduled hour), `Carrier_Type` (Major vs Low-Cost based on a fixed list of
  six legacy carriers), `Bad_Weather` (non-missing origin present-weather code),
  and the three-part knock-on encoding -- `prev_arr_delay` (inbound arrival
  delay in minutes, zeroed for first legs), `has_prev_leg` (flag separating
  structural zeros from true on-time inbounds), and `Prev_Leg` (three-level
  factor: first leg / on-time inbound / late inbound).
- Splits 80/20 into train and test (seed 42) and fits four models on that
  split: a base OLS on scheduled features only (distance, time-block, day,
  carrier) as a clean benchmark; a logistic regression on the full feature set
  for delay occurrence; an extended OLS that replaces raw `prev_arr_delay` with
  a log-transformed, mean-centered version and adds three two-way interactions
  (knock-on × carrier, knock-on × weather, carrier × weather); and a Gamma GLM
  with a log link to handle the right-skewed, heteroscedastic residuals.
- Evaluates all models on the same 200,000-flight hold-out so comparisons are
  apples-to-apples. The ROC curve and AUC are built by hand. The logistic
  classification threshold is set at the empirical delay rate rather than 0.5.
- Produces the figures and tables shown in the report: feature importance bar
  chart (scaled OLS coefficients), ROC curve, domino-effect curve (inbound
  delay vs departure delay probability), residual diagnostics, and a three-model
  benchmark table (R², RMSE, MAE).

The analysis is organised around four hypotheses -- carrier type, time-of-day
accumulation, adverse weather, and knock-on dominance -- and finds that inbound
aircraft performance (`prev_arr_delay`) is by far the strongest predictor,
several times larger than any scheduled feature once coefficients are
standardised.

`tidyverse` is used for all data work; models are fit with base-R `lm` and
`glm`; figures are composed with `patchwork`.

## Data

The dataset is **not** included here. Place the pre-joined file in a `data/`
folder next to the Rmd:

- `data/flight_nadav_with_weather.csv`

This file contains 2015 DOT on-time records with NOAA daily weather columns
(`ORIGIN_PRESENT_WEATHER_CODES`, `ORIGIN_WIND_SPEED_KNOTS`, `TMAX`, `TMIN`,
`PRCP`) and the knock-on column `PREV_LEG_ARR_DELAY` already merged in.

## Run

1. Install R and the required packages:

   ```r
   install.packages(c("tidyverse", "patchwork", "knitr", "rmarkdown", "scales"))
   ```

2. Place the data file as above.

3. Open `log_try_nadav.Rmd` in RStudio and click Knit to PDF. If you do not
   have LaTeX installed, run `tinytex::install_tinytex()` once first.

   From the command line:

   ```r
   rmarkdown::render("log_try_nadav.Rmd", output_format = "pdf_document")
   ```

`set.seed(42)` is applied at setup, at the 200K sample, and at the train/test
split, so all figures and inline numbers are fully reproducible.
