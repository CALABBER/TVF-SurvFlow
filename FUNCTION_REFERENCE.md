# TVF-SurvFlow — Function Reference

Per-function documentation for `TVF-SurvFlow_functions.R`: what each function does, its
parameters, what it returns, and any caveats worth knowing (useful when re-reading the
code or preparing to explain it).

**Conventions used below**
- `df` — a data frame in counting-process form with columns `id`, `tstart`, `tstop`,
  `status`, plus covariates. (Some functions also accept the simpler `time`/`status` form.)
- `outdir` — output directory; the model functions expect the `models/`,
  `summaries_for_each_model/`, and `metrics/` subfolders created by `create_output_folder()`.
- `tvf` — logical; `TRUE` trains on time-varying features, `FALSE` on the collapsed form.
- `splitted_data` — the list returned by `split_data()`, i.e. `list(train, test)`.
- `state` — internal label, `"TVF"` or `"no_TVF"`, used in output filenames.

> ⚠️ **Global dependency (applies to all model functions):** the model wrappers read the
> formula objects `formula_with_tvf_without_cluster` and
> `formula_without_tvf_without_cluster` from the **global environment**, not from their
> arguments. These must exist in the workspace before any model function is called.

---

## 1. Data handling

### `create_output_folder(outdir)`
Creates the output directory and its three standard subfolders.
- **Parameters:** `outdir` — target directory path.
- **Creates:** `models/`, `summaries_for_each_model/`, `metrics/` (recursively; no error if
  they already exist).
- **Returns:** a named character vector of the three subfolder paths.

### `split_data(df, train_prop = 0.9, seed = 8888)`
Splits data into train/test **by subject `id`**, so all rows for a given subject stay
together (essential for time-varying data).
- **Parameters:** `df`; `train_prop` — fraction of *subjects* in the training set;
  `seed` — RNG seed for the sampled ids.
- **Returns:** `list(train, test)`, each with unused factor levels dropped.
- **Note:** requires an `id` column.

### `replace_with_na(data, perc, exclude = c("status", "time"))`
Introduces missing values completely at random, per column, for robustness experiments.
- **Parameters:** `data`; `perc` — percent of values set to `NA` per column;
  `exclude` — columns to leave untouched.
- **Returns:** a copy of `data` with `NA`s inserted.

### `replace_and_impute_data(data, perc, exclude = c("status", "time"))`
Runs `replace_with_na()` and then imputes: **median** for numeric columns, **mode** (most
frequent level) for factor/character columns.
- **Parameters:** as above.
- **Returns:** the imputed data frame (character columns coerced back to factors).
- **Dependency:** `dplyr`.
- **Note:** the `exclude` argument is passed through to `replace_with_na()` only implicitly;
  imputation itself is applied to all numeric/factor columns.

---

## 2. Model dispatcher

### `model(df, model_type, outdir, tvf, splitted_data)`
Routes to the correct model-fitting function based on a string key, printing a status line.
- **Parameters:** `model_type` — one of
  `"rfsrc"`, `"gbm"`, `"gbm3"`, `"deepsurv"`, `"deeppamm"`, `"enet"`, `"lasso"`,
  `"ridge"`, `"coxph"`, `"clatree"`.
- **Returns:** whatever the dispatched model function returns.
- **Note:** `"enet"`, `"lasso"`, `"ridge"` all call `pcox_func()` with `alpha =`
  `0.5 / 1 / 0` respectively. `"deeppamm"` is dispatched without the `tvf` argument.

---

## 3. Model wrappers

Each wrapper follows the same pattern: pick the formula from `tvf`, fit the model, compute
C-index + variable importance + survival curves, write a PDF report to
`summaries_for_each_model/`, save the fitted object to `models/`, and return a result list.

### `coxph_func(df, outdir, tvf, splitted_data, seed = 8888)`
Standard Cox proportional-hazards model (`survival::coxph`).
- **Returns:** `list(fit, tuned, concord, vimp)` (`tuned` is an empty list — no tuning).
- ⚠️ **Bug:** `model_coxph <- list(fit = fit, ...)` is built **before** `fit` is created on
  the next line, so the saved object references an undefined/￼stale `fit`. Move the
  assignment to after the `coxph()` call.

### `pcox_func(df, outdir, tvf, splitted_data, alpha, lambda = 0.1, seed = 8888, n_indiv = 30)`
Penalized Cox via `pcoxtime` — ridge (`alpha = 0`), lasso (`alpha = 1`), elastic net
(`alpha = 0.5`). Computes linear predictors, concordance (with a `survival::concordance`
fallback), coefficient-based variable importance grouped by original variable, and both
individual and median survival curves.
- **Parameters:** `alpha` — elastic-net mixing; `lambda` — penalty strength;
  `n_indiv` — number of individual survival curves to draw.
- **Returns:** `list(fit, tuned, concord, coef_df, vimp_df, vimp_plot, surv_plot,
  test_df, train_df)`.
- **Note:** `model_type` (`"ridge"/"lasso"/"enet"`) is derived from `alpha` for the output
  filename.

### `rfsrc_func(df, outdir, tvf, splitted_data, seed = 8888)`
Random survival forest via `satpred` (`rfsrc.satpred`), tuned over a grid of
`mtry × nodesize × ntree` with `modtune()`/`modfit()`.
- **Returns:** `list(fit, tuned, concord, vimp)`.
- **Note:** intended for the **no-TVF** setting.

### `gbm_func(df, outdir, tvf, splitted_data, seed = 8888)`
Gradient boosting via `satpred` (`gbm.satpred`), tuned over
`shrinkage × n.trees × n.minobsinnode × interaction.depth`.
- **Returns:** `list(fit, tuned, concord, vimp)`.
- **Note:** uses the no-TVF formula explicitly; intended for the no-TVF setting.

### `gbm3_func(df, outdir, tvf, splitted_data, seed = 8888)`
Gradient boosting via the `gbm3` back-end (`gbm3.satpred`), with a `trace()` patch to fix
the `error.method` argument. Supports TVF.
- **Returns:** nothing explicit (writes PDF + saves model as side effects).
- ⚠️ **Bugs:** (a) `params_gbm3` is **used before it is defined** in the TVF branch;
  (b) in the no-TVF branch `tuned_gbm3` is never created but is passed to `modfit()`;
  (c) `save(gbm3_fit, ...)` refers to a variable that is actually named `fit_gbm3`.
  Reorder the grid definition and fix the variable names before relying on this function.

### `clatree_func(df, outdir, tvf, splitted_data, seed = 8888)`
Left-truncation-aware classification/survival tree via `LTRCtrees::LTRCIT` (+ `partykit`
controls). Produces partial-dependence plots (age, treatment), permutation variable
importance (drop in C-index), per-node/per-patient survival curves, and the C-index.
- **Returns:** `list(fit, concord, test_df, train_df, vimp_df)`.
- ⚠️ **Caveat:** the no-TVF branch currently assigns the **with-TVF** formula (then sets
  `tstart = 0`); confirm this is the intended behavior.

### `deepsurv_func(df, outdir, tvf, splitted_data, seed = 8888)`
DeepSurv via `satpred` (`deepsurv.satpred`), tuned over
`dropout × learning_rate × epochs` with a small hidden-layer grid.
- **Returns:** `list(fit, tuned, curves, concord, vimp)`.
- **Requires:** Keras/TensorFlow backend. Intended for the no-TVF setting.

### `deeppamm_func(data, outdir, splitted_data, epochs = 1000, batch_size = 128, seed = 1223)`
Deep piecewise-exponential additive model via `deeppamm` (R6) + `pammtools`. Builds a
`Y ~ s(time) + <covariates> + deep(<covariates>)` formula, trains with early-stopping and
LR-reduction callbacks, then computes cumulative-hazard risk scores, the C-index
(`survcomp::concordance.index`), and individual survival curves.
- **Parameters:** `data`; `epochs`; `batch_size`; `seed`.
- **Returns:** `list(fit, concordance, risk_scores, survival_curves, plot, pdf_file,
  intervals)`.
- ⚠️ **Caveats:** uses the **global `df`** to derive covariate names instead of its own
  `data` argument; `max_time` is **hardcoded to 587**. Both should be parameterized.
  Requires Keras/TensorFlow.

---

## 4. Wrapper / bridge methods (for the comparison layer)

### `predictSurvProb.pcoxtime(object, newdata, times, ...)`
S3 method so that `pcoxtime` fits can be used by `pec`/`riskRegression` (which have no
native `predictSurvProb` for `pcoxtime`). Reconstructs survival probabilities from the
linear predictor and a Cox baseline hazard.
- **Returns:** a matrix of survival probabilities (rows = subjects, cols = `times`).

### `DeepPAMM_predict(object, newdata, times)`
Converts DeepPAMM's native survival-probability output (on its own interval grid) into the
standard probability matrix expected by `pec`, interpolating onto the requested `times`.
- **Returns:** an (unnamed) probability matrix.

### `predictRisk.DeepPAMM(object, newdata, times, ...)`
Risk = `1 - predictSurvProb.DeepPAMM(...)`.
- ⚠️ **Note:** depends on `predictSurvProb.DeepPAMM`, which is present only as a
  commented-out block (two candidate versions, one for `Score()`, one for `pec()`). One must
  be uncommented for the DeepPAMM comparison path to work.

### `clean_vimp_df(df)`
Normalizes the various variable-importance frames returned by different models into a common
two-column `(Variable, Importance)` shape (handles `terms/Overall` and `type/importance`
naming).
- **Returns:** a cleaned data frame, or `NULL` if input is `NULL`.
- **Dependency:** `dplyr`.

---

## 5. Comparison & metrics

### `plot_survival_metrics(df, models_list, outdir, eval_times = seq(50, 550, 50), tp_roc = 50)`
Computes and plots time-dependent ROC, AUC-over-time, and Brier-score-over-time for all
models using `riskRegression::Score` (and `pec::pec`), writing a PDF to `metrics/`.
- **Returns:** PDF written as a side effect.
- ⚠️ **Caveats:** does **not** support TVFs yet; internally **rebuilds `models_list` from
  global `res_*` objects** (`res_rfsrc`, `res_gbm`, …, `DeepPAMM_object`), so those must
  exist in the workspace. Refactor to use the `models_list` argument.

### `compute_pec_all(df, res_rfsrc, res_gbm, res_deepsurv, res_coxph, res_enet, res_ridge, res_lasso, res_deeppamm, times = seq(50, 500, 50))`
Prediction Error Curves across all models via `pec::pec`.
- **Returns:** the `pec` result object (and plots it).
- ⚠️ **Bug:** references `res_rigde` (typo for `res_ridge`) when assembling the model list,
  which will error. Does not support TVFs yet.

---

## 6. Post-processing

### `postprocess_results(fit, tuned, train_df, test_df, modelname, outdir, state)`
Shared reporting step used by several model wrappers: computes individual survival curves,
concordance, and permutation variable importance (via the `satpred` getters
`get_indivsurv`, `get_survconcord`, `get_varimp`), then arranges them into a per-model PDF.
- **Returns (invisibly):** `list(curves, concord, vimp)`.

### `postprocess_comparison(results, outdir = <path>, filename = "comparison_all_perc_RFSRC.pdf")`
Loops over a list of results (e.g. from the missing-data experiments), producing a combined
PDF with survival curves, tuning, variable importance, and a concordance-vs-missingness
table.
- **Returns (invisibly):** the concordance table.
- ⚠️ **Note:** `outdir` defaults to a hardcoded absolute path — override it when calling.

---

## Quick caveats checklist (for a fast re-read)

| Function | Thing to remember |
|---|---|
| all model wrappers | need global `formula_*` objects defined first |
| `coxph_func` | `model_coxph` built before `fit` exists — bug |
| `gbm3_func` | grid used before defined; wrong save-var name — bugs |
| `clatree_func` | no-TVF branch uses the with-TVF formula |
| `deeppamm_func` | uses global `df`; `max_time` hardcoded to 587 |
| `plot_survival_metrics` | overrides arg with global `res_*`; no TVF |
| `compute_pec_all` | `res_rigde` typo; no TVF |
| `postprocess_comparison` | hardcoded default `outdir` |
| DeepSurv / DeepPAMM | require Keras/TensorFlow (Python) |
