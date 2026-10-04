# DENCAST

PySpark implementation of

> Corizzo R., Pio G., Ceci M., Malerba D.
> **DENCAST: distributed density-based clustering for multi-target regression**
> *Journal of Big Data* 6:43 (2019) — [10.1186/s40537-019-0207-2](https://doi.org/10.1186/s40537-019-0207-2)

Only DENCAST is implemented: no K-means, ARIMA or LSTM baselines. Where the
paper and the original Scala code disagree, the paper wins — the divergences
are listed under [Differences from the Scala code](#differences-from-the-scala-code).

Project layout follows **Cookiecutter Data Science**, the pipeline is
orchestrated with **DVC**, and every run is tracked with **MLflow** on
**DagsHub**.

---

## The algorithm

DENCAST is DBSCAN's notion of a cluster computed with different machinery, plus
a prediction step DBSCAN does not have.

```
labeled objects
      │
      │  1. LSH            random hyperplanes → r-bit signatures → permuted
      ▼                    sorted lists → candidate pairs → exact cosine filter
neighborhood graph ⟨V,E⟩
      │
      │  2. core objects   degree ≥ minPts
      ▼
      │  3. propagation    core objects send their cluster id along the edges;
      ▼                    every node keeps the maximum received; stop when
clusters                   propagations < labelChangeRate · |E|
      │
      │  4. prediction     an unlabeled object inherits the cluster of the most
      ▼                    similar labeled object; its targets are the
predictions                similarity-weighted average of that cluster's targets
```

Two properties worth keeping in mind:

* **Steps 2–3 only ever see object ids and edges**, never feature vectors.
  Propagating one integer per edge instead of a 214-dimensional vector is two
  orders of magnitude less network traffic per iteration; the features are
  joined back only for prediction.
* **There is no merging phase on a single machine.** Other distributed
  density-based methods cluster each partition locally and reconcile on the
  driver; here the clusters emerge from the propagation itself.

DENCAST is *inductive*: the model is built once and new objects are routed
through it. "Unlabeled" means hidden from the algorithm, not unknown to us —
targets are withheld on purpose so predictions can be scored.

---

## Quickstart

Dependencies are declared in `pyproject.toml` and installed into a virtual
environment; there is no `requirements.txt`.

```bash
uv sync --extra dev       # create .venv from pyproject.toml + uv.lock

cp .env.example .env      # DagsHub credentials (optional)

uv run dvc repro          # run the whole DAG
uv run dvc dag            # show the graph
uv run dvc metrics show   # see the results
uv run pytest             # tests
```

Extras: `--extra dev` adds ruff and pytest, `--extra notebooks` adds
JupyterLab, matplotlib and pandas.

Every stage in `dvc.yaml` is invoked through `uv run`, so the pipeline resolves
its own environment. A bare `python -m ...` would be resolved through `PATH`
instead, and since `dvc repro` runs each stage in a subprocess that inherits
the caller's `PATH`, the stages could silently execute under a different
interpreter than the one `dvc` itself is running from.

Requires Java 8/11/17 on the PATH (PySpark 3.5 does not support Java 21).
Without a `.env`, MLflow logs locally to `./mlruns` and everything still works
offline.

---

## Pipeline

```
extract ──► features ──┬──► train
                       └──► evaluate
```

| stage | command | what it does |
|---|---|---|
| `extract` | `python -m dencast.data.extract_data` | reads the `COPY` block of the pg_dump into `data/raw/` |
| `features` | `python -m dencast.features` | assembles the feature/target arrays, normalizes, writes Parquet |
| `train` | `python -m dencast.modeling.train` | fits one reference window, saves the model and cluster diagnostics |
| `evaluate` | `python -m dencast.modeling.predict` | rolls over the test split, aggregates the errors |

Each stage declares only the `params.yaml` blocks it uses, so editing a
clustering parameter re-runs `train` and `evaluate` but not the extraction of a
300 MB dump.

`train` exists because DENCAST is refit on a fresh window for every test day:
there is no single model artifact in the usual sense, so this stage fits the
first window of the split to give something inspectable. The real experiment is
`evaluate`.

### Experiments

```bash
dvc exp run -S clustering.min_pts=5 -S lsh.min_sim=0.97
dvc exp run -S lsh.r=11
dvc exp show
dvc metrics diff
```

---

## Configuration

Everything lives in `params.yaml`.

| parameter | paper | meaning |
|---|---|---|
| `lsh.r` | r | random hyperplanes = signature length in bits |
| `lsh.num_permutations` | numPerm | how many sorted lists to build |
| `lsh.b` | B | positional neighbours per object per list |
| `lsh.min_sim` | minSim | cosine threshold for an edge to exist |
| `clustering.min_pts` | minPts | minimum degree for a core object |
| `clustering.label_change_rate` | LCR | stop when propagations < LCR · \|E\| |
| `evaluation.window_size` | — | days of training before each test day |

### Tuning: look at the cluster count, not the RMSE

The cluster count is a far better guide than the error. It is structural, it
barely depends on which test day was picked, and Table 4 of the paper gives a
target: **618 clusters** for PV Italy ST at 30 days.

Measured on `2012-06-13`, 10 013 training objects, with the paper's `r=5`:

| minPts | minSim | edges | core | noise | **clusters** | RMSE | RMSE macro |
|---|---|---|---|---|---|---|---|
| 3 | 0.80 | 54 290 | 9907 | 22 | 242 | 0.1314 | 0.1285 |
| 3 | 0.90 | 36 666 | 8836 | 618 | 392 | 0.1250 | 0.1223 |
| 3 | 0.95 | 22 092 | 6897 | 2013 | **720** | 0.1448 | 0.1390 |
| 3 | 0.97 | 14 890 | 4981 | 3494 | **733** | 0.1262 | 0.1180 |
| 3 | 0.98 | 10 175 | 3292 | 5274 | **698** | 0.1143 | **0.1067** |
| 3 | 0.99 | 4 605 | 1160 | 7960 | 438 | 0.1810 | 0.1772 |
| 5 | 0.98 | 16 313 | 2805 | 5489 | 411 | 0.2007 | 0.1945 |
| 10 | 0.95 | 67 423 | 6271 | 2196 | 158 | 0.1520 | 0.1456 |

Full grid in [`reports/grid_pvitaly_st_30d.tsv`](reports/grid_pvitaly_st_30d.tsv).

Three things follow:

**`minPts` is the decisive knob.** At `minPts=5` the cluster count never leaves
the 150–420 band; at `minPts=3` it reaches 700+, and the paper's 618 sits
between `minSim=0.98` (698) and `0.99` (438). The paper never reports which
point of its grid won, so this is the only way to locate it.

**The cluster count is unimodal in `minSim`.** 242 → 392 → 720 → 733 → 698 →
438. At low thresholds merging dominates (connected components swallow
everything); at high thresholds noise dominates (too few core objects survive).
The peak is in between.

**`r=5` is a poor choice, and the paper's own Table 1 says so.** With 5 bits
there are 32 distinct signatures, so on a 10 000-object window ~313 objects
share each one; adjacency in the sorted list then means nothing and LSH recall
collapses to ~10 candidates per object. Raising `r` on the same day, with
everything else fixed:

| r | candidates/obj | core | noise | clusters | RMSE |
|---|---|---|---|---|---|
| 5 | 10.4 | 765 | 8307 | 241 | 0.1801 |
| 11 | 12.9 | 1579 | 7217 | 369 | 0.1322 |
| 15 | 19.3 | 2739 | 5519 | 297 | **0.1080** |
| 20 | 30.7 | 5530 | 2468 | 250 | 0.1131 |

Table 1 of the paper reports a 37% lower RMSE at `r=11` than at `r=5` for 27%
more runtime, and then sets `r=5` anyway.

### Choosing `min_sim` on a new dataset

Measure the distribution of pairwise cosine similarities first — it takes five
minutes and tells you whether there is any signal to threshold on. On the
synthetic set shipped here, `min_sim=0.90` lets 89% of all pairs through, the
graph comes out nearly complete, and predictions collapse to the global mean.

---

## Evaluation protocol

The paper's protocol, reproduced here:

```
854 days in PV Italy  →  10% = 85 days per split
5 splits × 85 days × 17 plants  =  7 225 per-(plant, day) RMSEs, then averaged
```

`references/pv_italy_*.txt` are the paper's own splits. Note they are **not
disjoint**: any two share about ten days, and across the ten files 850 day-slots
cover only 542 distinct days — 36% is literal duplication, since the same day
with the same 30-day window produces the same prediction.

### Two RMSEs, on purpose

```
rmse         pooled: all predictions together, one square root
rmse_macro   one RMSE per (plant, day), then a plain average   ← the paper's
```

The Scala code aggregates the second way:

```scala
predictionsByDay.groupByKey().map { calculateErrors(...) }
totalRMSE = dailyErrors.map(_._4).reduce(_+_) / dailyErrors.count()
```

Because the square root is concave, the macro-average is at most the pooled
figure, with equality only when every group errs equally. On plants that range
from RMSE 0.076 to 0.152 the gap is real, so the two are not interchangeable
and comparisons with the published numbers must use `rmse_macro`.

### Report the spread

LSH is randomized: different seeds give different graphs, different clusters and
different numbers. Measured on the synthetic dataset with the split held fixed:

| seed | RMSE | clusters | noise |
|---|---|---|---|
| 0 | 0.0996 | 15 | 20 |
| 1 | 0.0917 | 29 | 12 |
| 2 | 0.0980 | 36 | 13 |
| 3 | 0.0986 | 33 | 18 |
| | **0.0970 ± 0.0036** | | |

The predictions move ~4%, already the size of the gaps the paper reports
between competing methods. **The structure underneath moves far more**: the
cluster count goes from 15 to 36 between runs differing only in the random
hyperplanes. Averaging over a cluster smooths that out, so RMSE badly
understates how unstable the clustering is. Anything built on the cluster
structure — sizes, membership, the noise flag — inherits that instability
without RMSE ever warning you.

Set `evaluation.seeds: [0, 1, 2, 3, 4]` in `params.yaml`; the evaluate stage
reports mean and standard deviation.

---

## MLflow on DagsHub

DagsHub gives every repository an MLflow-compatible tracking server, so nothing
in the code is DagsHub-specific: it is a tracking URI plus basic auth.

```bash
cp .env.example .env
# MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
# MLFLOW_TRACKING_USERNAME=<owner>
# MLFLOW_TRACKING_PASSWORD=<token from dagshub.com/user/settings/tokens>
```

Logged per run: every `params.yaml` value; the structural metrics (clusters,
core objects, noise fraction, iterations, edges); per-day RMSE as a step series,
so the run page shows the spread across days rather than just the average; and
`reports/metrics.json` plus `reports/per_day_metrics.csv` as artifacts.

### DVC remote on DagsHub

DagsHub also serves DVC storage over the S3 protocol:

```bash
uv run python scripts/setup_dagshub_remote.py   # reads .env
uv run dvc push
```

The endpoint goes to `.dvc/config`; the token goes to `.dvc/config.local`,
which is gitignored.

---

## Layout

```
├── data
│   ├── external        the pg_dump files shipped with the paper
│   ├── raw             tables extracted from the dumps
│   ├── interim
│   └── processed       feature/target arrays, Parquet
├── dencast
│   ├── config.py       paths + typed params.yaml
│   ├── data/           stage: extract
│   ├── features.py     stage: features, plus the temporal split
│   ├── lsh.py          step 1: signatures, permutations, sorted lists, edges
│   ├── clustering.py   steps 2–3: core objects, propagation
│   ├── metrics.py      RMSE/MAE, pooled and macro
│   ├── spark_session.py
│   ├── tracking.py     MLflow/DagsHub
│   └── modeling
│       ├── train.py    stage: train
│       └── predict.py  step 4 + stage: evaluate
├── models              fitted model, graph, cluster sizes
├── references          the paper's official splits, original config.properties
├── reports             metrics, per-day results, tuning grid
├── scripts             one-off helpers (synthetic data, bike sharing prep)
├── scripts             one-off helpers (DagsHub remote, synthetic data)
├── tests               run with `uv run pytest`
├── dvc.yaml            the pipeline
├── params.yaml         every knob
├── pyproject.toml      dependencies, entry points, ruff/pytest config
└── uv.lock             the resolved environment -- commit it
```

---

## Differences from the Scala code

**Prediction.** Algorithm 4 compares the test object against *every labeled
object*, inherits the cluster of the most similar one, and averages that
cluster's targets weighted by similarity. The Scala code compares against the
*centroids* of the clusters and returns the centroid's target, which makes its
prediction step equivalent to K-means. This port follows the paper.

**Cosine similarity.** `Cosine.apply` in the Scala computes
`dotProduct(v2, v2) / (l2(v1) * l2(v2))` — the ratio of the norms, with the
angle between the vectors never entering the calculation. This port computes
the actual cosine.

**Similarity weighting.** The per-object weighting in the Scala prediction
reduces to `(pred * sim) / sim`, a single term, so the weights cancel. Here the
weighted average runs over all members of the cluster, as in Eq. 2.

**Edge weights.** The Scala graph is `RDD[(Node, Node)]`: the cosine decides
whether an edge exists and is then discarded. This port keeps the similarity on
the edge — four bytes per edge, and the propagation ignores it, but anything
richer (weighted density, attachment strength) needs it.

**Sliding window.** `SlidingRDD(signatures, b, b)` uses windowSize = step = b,
i.e. non-overlapping blocks: an object sees only its block-mates and never its
neighbour across a boundary. This port pairs `i` with `i+1..i+b`, which gives
overlapping coverage and better recall.

---

## Implementation notes

**No GraphX.** It is not exposed in PySpark and GraphFrames does not cover
`aggregateMessages`, so the message passing is written with DataFrame joins and
aggregations. The structure is the same: a map over the edges, a reduce over the
destinations.

**No Python UDFs.** Signatures, cosine, propagation and the weighted average are
all Spark SQL and run inside the JVM. That avoids serializing every row to a
Python interpreter, and as a side effect the code also runs where PySpark's
Python workers are broken.

**The weighted average is two running sums**, `sum(sim·target)` as a vector and
`sum(sim)` as a scalar, divided once at the end. Both are associative and
commutative, so partitions accumulate independently and merge in any order —
cluster members never have to be gathered in one place. This is what the paper
means by the `+` operator between vectors being computed distributedly.

**One spot does not scale.** `_candidate_pairs_for_permutation` numbers the
sorted signatures with `row_number()` over an unpartitioned window, which moves
all rows to one partition. On a cluster the sorted RDD would be numbered with
`zipWithIndex`; that needs Python workers, hence the SQL form.

**Lineage truncation.** The propagation loop rebuilds its plan on top of the
previous iteration, so `state` is checkpointed each round; without it Spark
re-derives the whole history at every action and overflows the stack.

---

## Reproducibility gaps in the paper

Worth knowing before trying to match the published numbers:

* the winning `(minPts, minSim)` is never reported — the grid is given, the
  chosen point is not, for any of the 42 dataset × setting × window cells;
* which dataset the LSH tuning of Table 1 was run on is not stated;
* the "independent split" used for tuning is not identified, and since the
  splits overlap it is not independent in the strict sense;
* no variance is reported anywhere, while LSH randomness alone moves the
  results by about as much as the gaps between the compared methods;
* the text says five splits, the repository ships ten date files per dataset.
