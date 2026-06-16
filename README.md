# Predicting Flight Departure Delays

Final project for Introduction to Data Science.
Erel Edry, Idan Hazan, Nadav Zvulun, Reem Mizrahi.

All code is in `report.Rmd` (knits to PDF).

## What the code does
The Rmd runs the whole analysis in order:
- Reads `flights.csv` and `airlines.csv`, draws a 100,000-flight sample, and
  drops cancelled flights (missing `DEPARTURE_DELAY`).
- Builds the features: `is_delayed` (>15 min), `Time_Block` (Morning/Afternoon/
  Evening/Night from the scheduled hour), and `Carrier_Type` (Major vs Low-Cost).
- Fits two models with the same predictors -- a linear regression for how long a
  delay is (with IQR outlier trimming) and a logistic regression for whether a
  flight is delayed.
- Evaluates them: builds the ROC curve and AUC by hand, and sets the
  classification cutoff at the delay rate instead of 0.5.
- Produces the figures and tables shown in the report.

Following the course rules, tidyverse is used for the data work and base R for
the models.

## Data
The US DOT 2015 dataset is staff-provided and **not** included here.
Put these two files in a `data/` folder next to `report.Rmd`:
- `data/flights.csv`
- `data/airlines.csv`

## Run
1. Install R and the `tidyverse` package.
2. Place the data files as above.
3. Open `report.Rmd` in RStudio and click Knit (PDF needs LaTeX --
   run `tinytex::install_tinytex()` once if you don't have it).

`set.seed(42)` makes the sample and all numbers reproducible.
