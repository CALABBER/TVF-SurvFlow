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
  - tuning plot (if applicable) 
- Possibility to directly compare models with compairson tools
-   ROc curve at one or several time points
-   AUC over time
-   Brier score over time

## Toy data set used
CGD 

