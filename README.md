# Is the river safe to swim in today?

**Live page:** [swim-safety.streamlit.app](https://swim-safety.streamlit.app) shows each site's risk today.

This project predicts whether a water sample at one of England's 20 river bathing sites will fail its bacteria limits, using recent rain, river flow and upstream sewage spills, so a swimmer can judge whether the river is safe that day. The dataset was built from five public sources (about 1,000 samples, 4 million rain and flow readings, 73,000 sewage spills), not downloaded ready-made.

![Failure rate by rain in the previous 48 hours](results/figures/1_failure_rate_by_rain.png)

- **Rain is the main driver.** About 10% of samples fail after a dry 48 hours, and 74% after more than 10 mm.
- **A one-line rule is as good as the models.** 'Fail if more than 2.7 mm fell in the previous 48 h' scores PR-AUC 0.48 on the unseen 2026 season; logistic regression, random forests and gradient boosting can't be told apart from it.
- **About a quarter of failures come in dry weather**, and no weather model caught any of them. What works is checking the site's last result: alerting when it rained **or** the last known result failed catches 48 of 62 failures instead of 40, including 6 of the 17 dry-weather ones, for a small loss in precision (33% vs 37%).
- **Sewage spills upstream hint at extra risk beyond rain** (odds ratio about 1.6, permutation p = 0.06), but with only 10 failures at the Thames sites it's a lead to follow up rather than a result.

## The question

Given the rain and the sewage spills upstream in the last few days, will today's sample at a designated river bathing site fail?

- **One row = one Environment Agency (EA) bathing-water sample** at one of England's 20 designated river bathing sites, 2021–2026.
- **Label:** fail if E. coli > 1000 or intestinal enterococci > 400 cfu/100 ml, the inland 'good' boundaries from the bathing water standards.

## Data

| Data | Source | Used for |
| --- | --- | --- |
| Bathing-water samples and site register | [EA Bathing Water Quality API](https://environment.data.gov.uk/bwq/) | Labels, site list and locations |
| Rainfall and river flow (15-minute) | [EA Hydrology API](https://environment.data.gov.uk/hydrology/) | Rain and flow features |
| Sewage spill alerts (start/stop, April 2022 onwards) | [Thames Water Open Data](https://data.thameswater.co.uk/s/), discharge alerts v2 | Spill features for the 4 Thames sites |
| Hourly air temperature, sunlight and sunshine (ERA5 reanalysis) | [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api), CC BY 4.0 | Temperature and sunlight features (UV in sunlight kills bacteria) |
| Flow-direction grid of the Thames basin | [POOPy](https://github.com/AlexLipp/POOPy) by Alex Lipp, archived on Zenodo | Deciding which overflows are upstream of each site |

Only Thames Water publishes spill *history*; other water companies share only what is spilling now. So spill features exist for the 4 Thames sites, and the spills question is tested there.

## Results in brief

Trained on 2021–2025 (664 samples), tested on the unseen 2026 season (398 samples, 62 failures). Thresholds were set on training data only.

| Rule or model (2026 test season) | PR-AUC (95% range) | Failures caught | Samples flagged | Precision |
| --- | --- | --- | --- | --- |
| Random guessing | 0.16 | | | |
| Rain rule (> 2.7 mm in 48 h) | 0.48 (0.36–0.61) | 40 of 62 | 107 | 37% |
| Best model: logistic regression, all features | 0.54 (0.42–0.65) | 29 of 62 | 64 | 45% |
| **Two-part rule: rain rule OR last known result failed** | | **48 of 62** | 146 | 33% |

No model's PR-AUC is significantly different from the rain rule's (paired bootstrap). The two-part rule is what the live page uses. The idea for it came from studying the 2026 misses, so its cleaner evidence is 2021–2025, where it caught 145 of 177 failures against 106 for rain alone.

[RESULTS.md](RESULTS.md) has the rest: every model tried, calibration, error analysis, the threshold choice, predicting bacteria counts, and the spills experiment.

## How this compares with the EA's own forecasts

The Environment Agency already issues daily [pollution risk forecasts](https://environmentagency.blog.gov.uk/2024/02/23/bathing-water-classifications-and-short-term-pollution/) at bathing waters where it can (171 of 424 sites in 2023) and shows warnings on [Swimfo](https://environment.data.gov.uk/bwq/profiles/). Swimfo is the official source, and this project doesn't replace it.

None of the 20 river sites here gets a rain-based forecast, though. The EA's archive of daily risk records ([stp-risk-prediction](https://environment.data.gov.uk/bwq/)) marks every one of them as a non-forecast site, so the risk is only raised when the EA already knows about a pollution incident. The archive holds 77 raised-risk records across all years, and most of them come from a single incident on the River Teme in Ludlow, which ran for 52 days in August and September 2026.

The archive covers 16 of the 20 sites and is patchy before 2025, so the comparison uses the 699 of 1,062 samples taken on a day with an EA record:

| | Samples flagged | Failures caught (of 126) | Precision |
|---|---|---|---|
| EA warning in force | 10 | 2 | 20% |
| Rain rule (> 2.7 mm in 48 h) | 194 | 81 | 42% |
| Two-part rule | 271 | 102 | 38% |

The EA doesn't try to forecast rain-driven pollution at these sites, so what the table measures is a gap in coverage. Official warnings were in force for 2 of the 126 failures; a rule anyone can check flagged 102. `scripts/check_ea_warnings.py` reproduces the table.

## Data decisions

Every choice made while building the dataset and models, and why, is listed in [DATA_DECISIONS.md](DATA_DECISIONS.md): which sites, the failure label, how rain, flow and spills were linked to each sample, the upstream rule, the leakage checks and the train/test split.

## Limitations

- **The spills experiment is small:** 179 samples and 10 failures, so treat the spills result as a lead.
- **Spill data covers Thames Water only**, because no other company publishes spill history.
- **A spill alert suggests a spill but doesn't confirm it**, and monitors can be offline.
- **2026 was cleaner than the training years** (15.6% failing vs 26.7%), so thresholds and probabilities learned from training data run high for 2026.
- **About a quarter of failures follow dry weather**, which no feature here explains.
- **Upstream links rely on a 50 m flow-direction grid.** A path running alongside the river within 300 m of a site but joining it further downstream would be linked wrongly. `results/figures/5_upstream_links_check.png` maps every link within 50 km so this can be checked by eye.
- **The spills likelihood-ratio test adjusts for rain but not site** (sites with no failures make a site-adjusted model unstable). The permutation test shuffles spills within site and rain group, so its p-value is the headline.
- **Relative flow is scaled by each station's median over all training seasons**, so early samples are scaled partly with later flow. This uses no labels or test data.
- **Alerts in the repeated hour when clocks go back** are treated as summer time, so they may sit up to 1 hour early.
- **Six sites appear only in the 2026 test set**, so results there reflect performance on new sites.
- **Wolvercote's 2022 season** comes before most Thames monitors were connected, so it is excluded from the spills experiment.

## Reproduce it

You need Python 3.11+ on Linux or WSL. Everything is rebuilt from the public APIs, and downloads are cached in `data/raw/`, so reruns are fast.

1. Clone the repository and install the requirements:

   ```bash
   git clone https://github.com/Ash679-ux/swim-safety
   cd swim-safety
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run every step in order:

   ```bash
   ./run_all.sh
   ```

To start from a later step, use `./run_all.sh 06`, or run the steps one at a time:

```bash
python scripts/04_samples.py           # clean samples and site table
python scripts/05_hydrology.py         # rain gauges and flow stations
python scripts/05b_weather.py          # air temperature and sunlight at each site
python scripts/06_spills.py            # Thames spill history -> spill events
python scripts/07_link_upstream.py     # which overflows are upstream of each site (+ map check)
python scripts/08_features.py          # features, with a leakage test
python scripts/09_models.py            # main model
python scripts/10_spills_experiment.py # spills experiment
python scripts/11_figures.py           # figures
python scripts/12_error_analysis.py    # calibration and error analysis
python scripts/13_ecoli_regression.py  # bacteria counts as a continuous target
```

Scripts 01–03 are the initial data audit and are not needed to rebuild the results. The first full run downloads several million readings and takes about 10–15 minutes.

## Today's risk page

`app.py` is a one-page Streamlit app, live at [swim-safety.streamlit.app](https://swim-safety.streamlit.app), that applies the two-part rule to every site right now: it flags a site if more than 2.7 mm of rain fell in the last 48 hours at the nearest live rain gauges, or if the site's latest sample (within 14 days) failed. It reads `sites.csv` in the repo root (site names and coordinates, written by step 04 and committed so the page runs without the pipeline):

```bash
streamlit run app.py
```

The live rain comes from the EA real-time flood-monitoring API, whose gauges aren't always the ones the models were trained on. Outside the bathing season there are no new samples, so only the rain part can flag a site.

## Tests

`pytest` runs 38 unit tests of the core logic (`tests/test_funcs.py`): the failure label, pairing spill start/stop alerts, rain and weather windows (including that readings after a sample are never used), spills 'as known at the time', the site-history feature (including a check that the label can't leak into it) and the fast PR-AUC used for the bootstrap. They also run on GitHub on every push.

## How this was built

The code was written with AI assistance (Claude). I made the decisions and checked them: the question, which data sources to use, how sites are linked to rain gauges, flow stations and spill outlets, the time-based train/test split and the alert threshold. I ran the pipeline on the real data myself and checked each result before it went into this README. DATA_DECISIONS.md explains the reasoning behind each choice.

## Licence and data

Code: MIT licence (see `LICENSE`). Data: EA bathing-water, hydrology and flood-monitoring data under the Open Government Licence v3.0; Thames Water open data under its portal terms; Open-Meteo weather data under CC BY 4.0; the flow-direction grid from the POOPy project. Raw data isn't committed; the scripts download it.

## Repository layout

```
app.py            Streamlit page: each site's risk today (streamlit run app.py)
sites.csv         site names and coordinates for app.py (written by step 04)
run_all.sh        runs the pipeline in order (./run_all.sh, or ./run_all.sh 06 to start at step 06)
DATA_DECISIONS.md every data and modelling choice, and why
RESULTS.md        full results: all models, calibration, error analysis, spills
scripts/          numbered pipeline steps (01-13), check_sample_hours.py (figure 6, sample-time
                  check), check_ea_warnings.py (EA warnings comparison) and shared code (common.py downloads, funcs.py tested core logic,
                  plotstyle.py figure style)
tests/            unit tests (pytest)
.github/workflows CI: runs the tests on every push
data/raw/         untouched API downloads (not committed; rebuilt by the scripts)
data/interim/     cleaned tables: samples, sites, stations, rain, flow, spills, links
data/processed/   features.csv, one row per sample
results/          model results, predictions and figures (figures committed)
```
