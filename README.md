# Is the river safe to swim in today?

Predicting whether a river bathing-water sample in England will fail its bacteria limits, from recent rain, river flow and upstream sewage spills. Every dataset was collected and linked from public APIs, not downloaded as a ready-made table.

**Main findings**

- **Rain is the main driver.** About 10% of samples fail after a dry 48 hours, 27% after 4–10 mm of rain, and 74% after more than 10 mm.
- **A one-line rain rule does most of the work.** "Fail if more than 2.7 mm fell in the previous 48 h" scores PR-AUC 0.48 on the unseen 2026 season. A logistic regression with rain, flow, season and site history scores 0.55, but the uncertainty ranges overlap.
- **Upstream sewage spills hint at extra risk beyond rain, but the data can't confirm it.** At Thames Water sites, spills 25–50 km upstream over the previous 2–3 days are linked to higher odds of failure (odds ratio about 1.6, permutation p = 0.06). With only 10 failures, this is a lead for more data rather than a firm result.

## The question

Given the rain and the sewage spills upstream in the last few days, will today's sample at a designated river bathing site fail?

- **One row = one Environment Agency (EA) bathing-water sample** at one of England's 20 designated river bathing sites, 2021–2026.
- **Label:** fail if E. coli > 1000 or intestinal enterococci > 400 cfu/100 ml, the inland "good" boundaries from the bathing water standards.

## Data

| Data | Source | Used for |
| --- | --- | --- |
| Bathing-water samples and site register | [EA Bathing Water Quality API](https://environment.data.gov.uk/bwq/) | Labels, site list and locations |
| Rainfall and river flow (15-minute) | [EA Hydrology API](https://environment.data.gov.uk/hydrology/) | Rain and flow features |
| Sewage spill alerts (start/stop, April 2022 onwards) | [Thames Water Open Data](https://data.thameswater.co.uk/s/), discharge alerts v2 | Spill features for the 4 Thames sites |
| Flow-direction grid of the Thames basin | [POOPy](https://github.com/AlexLipp/POOPy) by Alex Lipp, archived on Zenodo | Deciding which overflows are upstream of each site |

Only Thames Water publishes spill *history*; other water companies share only what is spilling now. So spill features exist for the 4 Thames sites, and the spills question is tested there.

## Results

![Failure rate by rain in the previous 48 hours](results/figures/1_failure_rate_by_rain.png)

### Main model: all 20 sites, tested on the unseen 2026 season

Trained on 2021–2025 (664 samples), tested on 2026 (398 samples, 62 failures). Thresholds were chosen on the training data only.

| Model | PR-AUC (95% range) | ROC-AUC | Failures caught | Samples flagged | Precision |
| --- | --- | --- | --- | --- | --- |
| Random guessing | 0.16 | 0.50 | | | |
| Site failure rate only | 0.28 (0.20–0.37) | 0.70 | 53 of 62 | 239 | 22% |
| Rain rule (> 2.7 mm in 48 h) | 0.48 (0.36–0.61) | 0.75 | 40 of 62 | 107 | 37% |
| Logistic regression, rain + season | 0.51 (0.39–0.63) | 0.76 | 27 of 62 | 55 | 49% |
| Logistic regression, all features | 0.55 (0.42–0.67) | 0.83 | 32 of 62 | 72 | 44% |
| Random forest, all features | 0.54 (0.42–0.67) | 0.83 | 36 of 62 | 84 | 43% |

![Precision-recall curves on the 2026 test season](results/figures/2_precision_recall_2026.png)

The models' small edge over the rain rule comes from knowing each site's history: PR-AUC 0.60 vs 0.50 on familiar sites, but about the same (0.46–0.49) on the 6 sites first sampled in 2026. For a site with no history, the rain rule is as good as anything tried here.

### Spills experiment: 4 Thames sites, 2023–2026

179 samples, 10 failures. With so few failures, comparing prediction scores would be close to meaningless, so the question was asked statistically: does adding upstream spill hours improve a model that already knows rain?

| Check | Result |
| --- | --- |
| Within similar rain | Light rain (0.5–4 mm): 2 of 4 failed after a spill vs 1 of 35 without. Wet (> 4 mm): 2 of 11 vs 3 of 24 |
| Likelihood-ratio test, spills added to a rain model | Odds ratio 1.60 per log(1 + spill hours), 95% CI 0.90–2.83, p = 0.13 |
| Permutation test (spills shuffled within site and rain group) | p = 0.056 |
| Same test with river flow also in the model | Odds ratio 2.37, 95% CI 1.23–4.56, p = 0.015 (one of two models tried, so supporting evidence only) |
| Cross-validated PR-AUC | Rain only 0.39; rain + spills 0.35 |

![Spill odds ratios by distance band and time window](results/figures/3_spill_odds_ratios.png)

In plain terms: under the rain model, a sample taken after 6 hours of upstream spilling has about 2.5 times the odds of failing, but the plausible range runs from no effect to about 7 times. The signal is strongest for spills 25–50 km upstream over the previous 48–72 hours, which fits river travel times of very roughly 25 km a day. Within 10 km it is weakest, because only 1–4 overflows sit that close to each site. More seasons of data would settle the question.

## Data decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Sites | The 20 sites the EA register labels `RiverBathingWater` | The register's own label; lakes, estuaries and coastal sites behave differently and mostly lack upstream overflows |
| Years | Samples from 2020 onwards (in practice 2021: only Ilkley and Wolvercote before 2024) | Spill monitoring is patchy before then |
| Label | Fail if E. coli > 1000 or intestinal enterococci > 400 cfu/100 ml | E. coli alone missed failures at sites like Wolvercote |
| Counts outside the lab's range | `<` treated as a pass; `>` at or above a limit treated as a fail | Below or above any limit respectively |
| Duplicates | 30 reissued records removed (newest copy kept) | Ilkley's 2021 season was reissued in May 2022 and would have counted twice |
| Flagged, not dropped | Discountable samples (3, all fail) and samples < 4 days after the previous one (113) | They are real pollution events |
| Sample time zone | Treated as UK clock time, converted to UTC | The API gives no time zone; if wrong, samples sit up to 1 h early, which can't leak future data |
| Rain | Average of the 3 nearest EA gauges (within 20 km); a gauge counts only with 80% of readings present in the window; "Suspect" readings kept | Averaging limits the effect of one bad gauge; dropping Suspect readings (16%) would leave gaps |
| Flow | Nearest EA flow station on the same river (within 40 km); none for Chester (no Dee station) | Flow from a tributary or another river is worse than none |
| Spills | Thames Water start/stop alerts paired into spills; zero-length spills dropped; monitor offline periods kept as a feature | EDM alerts indicate spills rather than confirm them |
| Upstream rule | Follow the flow-direction grid downstream from each overflow; upstream if the path passes within 300 m of the site | Excludes downstream overflows automatically; distances run 5–10% long because the path zigzags across 50 m grid cells |
| No leakage | Every feature uses only data from before the sample time, checked by a built-in test | The model may only see what was known when the sample was taken |
| Split | Train 2021–2025, test 2026 | Tests on a season the model has never seen |
| Thresholds | Chosen on training data only (out-of-fold) | Keeps test results honest |

## Limitations

- **The spills experiment is small:** 179 samples and 10 failures. Treat it as a lead, not a finding.
- **Spill data covers Thames Water only**, because no other company publishes spill history.
- **Spill alerts indicate spills, they don't confirm them**, and monitors can be offline.
- **2026 was cleaner than the training years** (15.6% failing vs 26.7%), so thresholds set on training data flag more often than 2026 needed.
- **Six sites appear only in the 2026 test set**, so results there reflect performance on new sites.
- **Wolvercote's 2022 season** comes before most Thames monitors were connected, so it is excluded from the spills experiment.

## Reproduce it

Python 3.11+ on Linux or WSL. Everything is rebuilt from the public APIs; downloads are cached in `data/raw/`, so reruns are fast.

```bash
git clone https://github.com/Ash679-ux/swim-safety
cd swim-safety
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/04_samples.py           # clean samples and site table
python scripts/05_hydrology.py         # rain gauges and flow stations
python scripts/06_spills.py            # Thames spill history -> spill events
python scripts/07_link_upstream.py     # which overflows are upstream of each site
python scripts/08_features.py          # features, with a leakage test
python scripts/09_models.py            # main model
python scripts/10_spills_experiment.py # spills experiment
python scripts/11_figures.py           # figures
```

Scripts 01–03 are the initial data audit and are not needed to rebuild the results. The first full run downloads several million readings and takes about 10–15 minutes.

## Repository layout

```
scripts/        numbered pipeline steps (01-11) and shared helpers (common.py)
data/raw/       untouched API downloads (not committed; rebuilt by the scripts)
data/interim/   cleaned tables: samples, sites, stations, rain, flow, spills, links
data/processed/ features.csv, one row per sample
results/        model results, predictions and figures (figures committed)
```
