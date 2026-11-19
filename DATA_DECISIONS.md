# Data decisions

Every choice made while building the dataset and models, and why.

| Decision | Choice | Why |
| --- | --- | --- |
| Sites | The 20 sites the EA register labels `RiverBathingWater` | The register's own label; lakes, estuaries and coastal sites behave differently and mostly lack upstream overflows |
| Years | Samples from 2020 onwards (in practice 2021: only Ilkley and Wolvercote before 2024) | Spill monitoring is patchy before then |
| Label | Fail if E. coli > 1000 or intestinal enterococci > 400 cfu/100 ml | E. coli alone missed failures at sites like Wolvercote |
| Counts outside the lab's range | `<` treated as a pass; `>` at or above a limit treated as a fail | Below or above any limit respectively |
| Duplicates | 30 reissued records removed (newest copy kept) | Ilkley's 2021 season was reissued in May 2022 and would have counted twice |
| Flagged, not dropped | Discountable samples (3, all fail) and samples < 4 days after the previous one (113) | They are real pollution events |
| Sample time zone | Treated as UK clock time, converted to UTC | The API gives no time zone; if wrong, samples sit up to 1 h early, which can't leak future data. Checked (scripts/check_sample_hours.py): no date-only records, every sample between 07:00 and 19:00 and 96% between 10:00 and 15:00, a working-day pattern in every year |
| Rain | Average of the 3 nearest EA gauges (within 20 km); a gauge counts only with 80% of readings present in the window; "Suspect" readings kept | Averaging limits the effect of one bad gauge; dropping Suspect readings (16%) would leave gaps |
| Flow | Nearest EA flow station on the same river (within 40 km); none for Chester (no Dee station) | Flow from a tributary or another river is worse than none |
| Spills | Thames Water start/stop alerts paired into spills; zero-length spills dropped; monitor offline periods kept as a feature | EDM alerts indicate spills rather than confirm them |
| Upstream rule | Follow the flow-direction grid downstream from each overflow; upstream if the path passes within 300 m of the site | Excludes downstream overflows automatically; distances run 5–10% long because the path zigzags across 50 m grid cells |
| No leakage | Every feature uses only data from before the sample time, checked by a built-in test | The model may only see what was known when the sample was taken |
| Split | Train 2021–2025, test 2026 | Tests on a season the model has never seen |
| Thresholds | Chosen on training data only, from leave-one-season-out predictions | Keeps test results honest and matches the test design |
| Site history feature | Each training sample's site failure rate is learned from the other training seasons only, rebuilt inside every cross-validation fold | A leave-one-out rate leaks the label: failing samples always get a slightly lower rate than passing ones at the same site, which tree models learn as a fake pattern |
| Model comparison | Paired bootstrap of the PR-AUC difference against the rain rule, on the same test resamples | Answers "is it really better?" directly, rather than comparing separate intervals |
| Previous result feature | The site's last result taken at least 2 days earlier in the same season (checked by a test in 08_features.py) | Lab cultures take about 2 days, so a newer result wouldn't be known yet; in the test season this uses earlier 2026 results, just as a live forecast would |
