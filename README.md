# TVF-SurvFlow
R package for standardized and user-friendly framework for survival analysis with time-varying features.

## Models available
- CoxPH ( with and without TVFs)
- penalized Cox PH (lasso, enet, redige) (with and without TVFs)
- Gradient Boosted Model (with and without TVFs)
- Classification tree (with and without TVFs)
- Random Survival Forest (without TVFs)
- DeepSurv (without TVFs)
- DeepPAMM (without TVFs)

## Features
- possibility to choose to include or not TVFs
- automatically generated plots for each model with
  - concordance index
  - individual survival curves
  - variable of importance
  - tuning plot (if applicable)

 ![rfsrc](images/RFSRC.png)

- Possibility to directly compare models with comparison tools
  - ROC curve at one or several time points
  - AUC over time
  - Brier score over time
  - Variable of importance

![metrics](images/metrics.png)
![varimp](images/varimp_with_gbm3_satpred.png)

## Toy data set used
CGD - Chronic Granulomatous Disease

The International Chronic Granulomatous Disease Cooperative Study Group. A
controlled trial of interferon gamma to prevent infection in chronic granulomatous disease. _The New England Journal of Medicine_, (324):509–516, 1991.

