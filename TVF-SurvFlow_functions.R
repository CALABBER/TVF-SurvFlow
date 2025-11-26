############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                  ENVIRONMENT SETTING AND LIBRARY LOADING 
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################
# 
# Sys.setenv(RETICULATE_PYTHON = "C:/Users/capuc/r-python-env39-13plus/Scripts/python.exe")
# library(reticulate)
# py_config()

library(survival)
library(randomForestSRC)
library(pec)
library(riskRegression)
library(satpred)
library(shellpipes)
library(gbm)
library(gbm3)
library(survivalmodels)
library(gridExtra)
library(grid)
library(ggplot2)
library(gridGraphics)
library(pcoxtime); pcoxtheme()
library(caret)
library(survminer)
library(tidyr)
library(dplyr)
library(survcomp)
library(keras)
library(tensorflow)
library(deeppamm)
library(pammtools)
library(reshape2)
library(data.table)
library(LTRCtrees)
library(partykit)
library(pdp)
library(purrr)
library(RColorBrewer)
satpredtheme()



############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                  DATA HANDLING 
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################

# ----------------------------------------------------------------------------------------------------
#          CREATE OUTPUT FOLDER
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     creates an output folder
# 
# PARAMETERS
#     folder path
# 
# RETURNS
#     folder paths
# ----------------------------------------------------------------------------------------------------
create_output_folder <- function(outdir) {
  subfolders <- c("models", "summaries_for_each_model", "metrics")
  if (!dir.exists(outdir)) {
    dir.create(outdir, recursive = TRUE)
  }
  for (sub in subfolders) {
    dir.create(file.path(outdir, sub), showWarnings = FALSE)
  }
  paths <- setNames(file.path(outdir, subfolders), subfolders)
  return(paths)
}


# ----------------------------------------------------------------------------------------------------
#          SPLIT TRAIN/TEST
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     splits the data set into a train and test set
# 
# PARAMETERS
#     data frame, train-test proportion, seed
# 
# RETURNS
#     a train and test data sets
# ----------------------------------------------------------------------------------------------------
split_data <- function(df, train_prop = 0.9, seed = 8888) {
  set.seed(seed)
  ids <- unique(df$id)
  train_ids <- sample(ids, size = floor(length(ids) * train_prop))
  output <- list(
    train = droplevels(df[df$id %in% train_ids, ]),
    test  = droplevels(df[!df$id %in% train_ids, ])
  )
  return(output)
}



# ----------------------------------------------------------------------------------------------------
#          MISSING DATA RANDOMLY ADDED
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     introduces NA randomly in data set
# 
# PARAMETERS
#     data set, percentage of values to shift to NA per column, variable to eventually exclude
# 
# RETURNS
#     data set with NA
# ----------------------------------------------------------------------------------------------------
replace_with_na <- function(data, perc, exclude = c("status", "time")) {
  data_na <- data 
  for (col in setdiff(names(data_na), exclude)) {
    n <- nrow(data_na)
    n_na <- ceiling(n * perc / 100)  
    na_indices <- sample(1:n, n_na) 
    data_na[na_indices, col] <- NA
  }
  
  return(data_na)
}



# ----------------------------------------------------------------------------------------------------
#          MISSING DATA RANDOMLY ADDED + IMPUTATION
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     introduces NA randomly in data set then replace the NA values by the mean value across the column
# 
# PARAMETERS
#     data set, percentage of values to shift to NA per column, variable to eventually exclude
# 
# RETURNS
#     modified data set
# ----------------------------------------------------------------------------------------------------
replace_and_impute_data <- function(data, perc, exclude = c("status", "time")) {
  data_na <- replace_with_na(data, perc)
  data_imp <- data_na %>%
    mutate(across(where(is.numeric),
                  ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(across(where(is.factor),
                  ~ifelse(is.na(.),
                          as.character(names(sort(table(.), decreasing = TRUE))[1]),
                          as.character(.)))) %>%
    mutate(across(where(is.character), as.factor))
  return(data_imp)
}



############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                  MODELS 
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################
# ----------------------------------------------------------------------------------------------------
#          CALL MODELS
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     Chooses which model function to use for training
# 
# PARAMETERS
#     data frame, name of model (string), output directory, if tvf or no tvf (string), splitted data
# 
# RETURNS
#     calls the corresponding model function
# ----------------------------------------------------------------------------------------------------
model <- function(df, model_type, outdir, tvf, splitted_data){
  if (model_type == "rfsrc"){
    print("RandomForestSRC training...")
    return(rfsrc_func(df, outdir, tvf, splitted_data))
  }
  if (model_type == "gbm"){
    print("GradientBoostedModel training...")
    return(gbm_func(df, outdir, tvf, splitted_data))
  }
  if (model_type == "deepsurv"){
    print("DeepSurv training...")
    return(deepsurv_func(df, outdir, tvf, splitted_data))
  }
  if (model_type == "enet"){
    print("PenalizedCox enet training...")
    return(pcox_func(df, outdir, tvf, splitted_data, alpha=0.5))
  }
  if (model_type == "lasso"){
    print("PenalizedCox lasso training...")
    return(pcox_func(df, outdir, tvf, splitted_data, alpha=1))
  }
  if (model_type == "ridge"){
    print("PenalizedCox ridge training...")
    return(pcox_func(df, outdir, tvf, splitted_data, alpha=0))
  }
  if (model_type == "coxph"){
    print("CoxPH training...")
    return(coxph_func(df, outdir, tvf, splitted_data))
  }
  if (model_type == "deeppamm"){
    print("DeepPAMM training...")
    return(deeppamm_func(df, outdir, splitted_data))
  }
  if (model_type == "clatree"){
    print("Classification tree training...")
    return(clatree_func(df, outdir, tvf, splitted_data))
  }
  if (model_type == "gbm3"){
    print("GBM3 training...")
    return(gbm3_func(df, outdir, tvf, splitted_data))
  }
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 1 : RANDOM SURVIVAL FOREST
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     calls split data, 
#     creates a grid of parameters to make a tuning plot,
#     fit the model with a random survival forest,
#     calls the postprocess_results function,
#     saves the model as RData
# 
# PARAMETERS
#     data frame, output directory, if tvf or no tvf (string), splitted data, seed
# 
# RETURNS
#     a list made of the fitted model, the tuning plot
# ----------------------------------------------------------------------------------------------------
rfsrc_func <- function(df, outdir, tvf, splitted_data, seed = 8888) {
  train_df <- splitted_data$train
  test_df <- splitted_data$test

  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
    tuned_coxph <- list()
  } else {
    f <- formula_without_tvf_without_cluster
    state <- "no_TVF"
    print("without time-varying features")
  }
  
  params <- expand.grid(mtry = c(4, 5, 6), 
                        nodesize = c(5, 10, 15), 
                        ntree = c(800, 1000, 1200))
  
  tuned_rfsrc <- modtune(f, train_df,
                         param_grid = params,
                         modfun = rfsrc.satpred, 
                         parallelize = TRUE, 
                         seed = seed)
  
  fit <- modfit(tuned_rfsrc, return_data = FALSE)
  concord <- get_survconcord(fit, test_df)
  vimp <- get_varimp(fit, type = "perm", newdata = train_df, nrep = 20, modelname = "RandomForestSRC")
  postprocess_results(fit, tuned_rfsrc, train_df, test_df, modelname = "RandomForestSRC", outdir, state)
  
  model_rfsrc = list(fit = fit, tuned = tuned_rfsrc)
  file_path_save <- file.path(outdir, "models/model_rfsrc.RData")
  save(model_rfsrc, file = file_path_save)
  cat("\nModel saved to:", file_path_save, "\n")
  
  #return(list(fit = fit, tuned = tuned_rfsrc))
  result <- list(fit = fit, tuned = tuned_rfsrc, concord = concord, vimp = vimp)
  invisible(result)
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 2 : CLASSIFICATION TREE
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     calls split_data()
#     define if tvfs are taken into account
#     fit the model with a classification tree
#     identifies variables of importance (partial dependence plots and variable permutation)
#     gets individual survival curves for each node
#     computes c-index
#     saves everything as pdf
#     saves the model as RData
# 
# PARAMETERS
#     data frame, output directory, if tvf or no tvf (string), splitted data, seed
# 
# RETURNS
#     a list with fitted model, c-index, test and train df, variable of importance df
# ----------------------------------------------------------------------------------------------------
clatree_func <- function(df, outdir, tvf, splitted_data, seed = 8888) {
  set.seed(seed)
  train_df <- splitted_data$train
  test_df <- splitted_data$test
  
  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
    tuned_coxph <- list()
  } else {
    f <- formula_with_tvf_without_cluster
    state <- "no_TVF"
    train_df$tstart = 0
    test_df$tstart = 0
    print("without time-varying features")
  }
  
  ctrl <- ctree_control(
    minsplit = 10,       
    minbucket = 5,       
    mincriterion = 0.8   
  )
  
  clatree_fit <- LTRCIT(f, data=train_df, Control = ctrl)
  model_tree <- clatree_fit
  
  # Variable importance - Partial Dependance Plots
  age_partial_df <- pdp::partial(clatree_fit, pred.var = "age", train = train_df, plot = FALSE)
  treat_partial_df <- pdp::partial(clatree_fit, pred.var = "treat", train = train_df, plot = FALSE)
  p1 <- ggplot(age_partial_df, aes(x = age, y = yhat)) +
    geom_point(color = "steelblue", size = 2) +
    theme_minimal() +
    ggtitle("Partial Dependence of Age on Predicted Survival") +
    xlab("Age") + ylab("Predicted Risk")
  p2 <- ggplot(treat_partial_df, aes(x = treat, y = yhat)) +
    geom_point(color = "steelblue", size = 2) +
    theme_minimal() +
    ggtitle("Partial Dependence of Treatment on Predicted Survival") +
    xlab("Treatment") + ylab("Predicted Risk")
  
  # Variable Importance - Variable Permutation
  lp <- predict(clatree_fit, newdata = test_df)
  baseline_cindex <- survConcordance(Surv(tstart, tstop, status) ~ lp, data = test_df)$concordance
  permute_var <- function(varname) {
    df_perm <- test_df
    df_perm[[varname]] <- sample(df_perm[[varname]])  # shuffle values
    lp_perm <- predict(clatree_fit, newdata = df_perm)
    c_index_perm <- survConcordance(Surv(tstart, tstop, status) ~ lp_perm, data = df_perm)$concordance
    baseline_cindex - c_index_perm  # importance = drop in C-index
  }
  varnames <- setdiff(colnames(df), c("id", "tstart", "tstop", "status"))
  importance <- map_dbl(varnames, permute_var)
  importance_df <- data.frame(
    Variable = varnames,
    Importance = importance)
  p3 <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Variable Importance (Permutation)", x = "Variable", y = "Drop in C-index") +
    theme_minimal()

  # Predicted survival curve per node
  terminal_nodes <- predict(clatree_fit, newdata = test_df, type = "node")
  train_nodes <- predict(clatree_fit, type = "node")  # nodes for training data
  surv_list <- lapply(unique(terminal_nodes), function(node) {
    node_data <- train_df[which(train_nodes == node), ]  # <- use train_nodes here
    survfit(Surv(tstart, tstop, status) ~ 1, data = node_data)})
  names(surv_list) <- unique(terminal_nodes)
  patient_surv <- lapply(1:nrow(test_df), function(i) {
    node <- terminal_nodes[i]
    survfit_obj <- surv_list[[as.character(node)]]
    data.frame(
      id = test_df$id[i],
      time = survfit_obj$time,
      surv = survfit_obj$surv)})
  patient_surv_df <- dplyr::bind_rows(patient_surv)
  scurves_tree <- ggplot2::ggplot(patient_surv_df, aes(x = time, y = surv, group = id)) +
    ggplot2::geom_step(alpha = 0.3) +
    ggplot2::labs(title = "Predicted Survival Curves per Patient",
                  x = "Time",
                  y = "Survival Probability") +
    ggplot2::theme_minimal()
 
  # C-index
  test_lp <- as.numeric(predict(clatree_fit, newdata = test_df))  # ensure numeric
  cobj <- concordance(Surv(tstart, tstop, status) ~ test_lp, data = test_df)
  c_index <- cobj$concordance
  print(paste("C-index: ", c_index))

  # pdf summary
  pdf_file_summary <- paste0(outdir, "/summaries_for_each_model/ClassificationTree_", state, ".pdf")
  pdf(pdf_file_summary, width = 14, height = 8)
  grid.newpage()
  summary_text <- paste0("Model: Classification Tree - ", state, "\nConcordance: ", round(c_index, 6), "\n")
  grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"), gp = gpar(fontsize = 12))
  grid.arrange(p1, p2, p3, ncol = 3)
  grid.draw(scurves_tree) 
  dev.off()
  cat("\nPDF report saved to:", pdf_file_summary, "\n")
  
  # Save model as RData
  file_path_save <- paste0(outdir, "/models/model_tree_", state, ".RData")
  save(model_tree, file = file_path_save)
  cat("\nModel saved to:", file_path_save, "\n")
  
  result <- list(
    fit = clatree_fit,
    concord = c_index,
    test_df = test_df,
    train_df = train_df,
    vimp_df = importance_df
  )
  
  invisible(result)
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 3a : GRADIENT BOOSTED MODEL
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     calls split data, 
#     defines if tvfs are taken into account
#     creates a grid of parameters to make a tuning plot,
#     fit the model with a gradient boosted model,
#     saves the model as RData
#     calls the postprocess_results function
# 
# PARAMETERS
#     data frame, output directory, if tvf or no tvf (string), splitted data, seed
# 
# RETURNS
#     a list made of the fitted model, the tuning plot, c-index and variable of importance df   
# ----------------------------------------------------------------------------------------------------
gbm_func <- function(df, outdir, tvf, splitted_data, seed = 8888) {
  train_df <- splitted_data$train
  test_df <- splitted_data$test

  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
    tuned_coxph <- list()
  } else {
    f <- formula_without_tvf_without_cluster
    state <- "no_TVF"
    print("without time-varying features")
  }

  params_gbm <- expand.grid(
    shrinkage = seq(0.05, 0.1, length.out = 3),
    n.trees = c(800, 1000),
    n.minobsinnode = c(10, 15),
    interaction.depth = c(2, 4, 8)
  )
  
  tuned_gbm <- modtune(
    formula_without_tvf_without_cluster,
    train_df,
    distribution = "coxph",
    param_grid = params_gbm,
    modfun = gbm.satpred,
    parallelize = TRUE
  )
  
  fit <- modfit(tuned_gbm, return_data = TRUE, formula = formula_without_tvf_without_cluster)
  postprocess_results(fit, tuned_gbm, train_df, test_df, modelname = "GradientBoostedModel", outdir, state)
  
  model_gbm = list(fit = fit, tuned = tuned_gbm)
  file_path_save <- file.path(outdir, "models/model_gbm.RData")
  save(model_gbm, file = file_path_save)
  cat("\nModel saved to:", file_path_save, "\n")
  
  concord <- get_survconcord(fit, test_df)
  vimp <- get_varimp(fit, type = "perm", newdata = train_df, nrep = 20, modelname = "GradientBoostedModel")
  result <- list(fit = fit, tuned = tuned_gbm, concord = concord, vimp = vimp)
  invisible(result)
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 2b : GRADIENT BOOSTED MODEL 3
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     traces the gbm3 function to the one from satpred package,
#     calls split data, 
#     defines if tvfs are taken into account,
#     creates a grid of parameters to make a tuning plot,
#     fit the model with a gradient boosted 3 model,
#     makes scurves plots, 
#     makes variable of importance graphs, 
#     compute c-index,
#     saves everything as pdf,
#     saves the model as RData
# 
# PARAMETERS
#     data frame, output directory, if tvf or no tvf (string), splitted data, seed
# 
# RETURNS
#     -
# ----------------------------------------------------------------------------------------------------
gbm3_func <- function(df, outdir, tvf, splitted_data, seed = 8888) {

  trace("gbm3.satpred", at = 1, print = FALSE, quote({
            if (length(error.method) > 1) {
            error.method <- "auto"
          }}))
  
  train_df <- splitted_data$train
  test_df <- splitted_data$test
  
  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
   tuned_gbm3 <- modtune(f, train_df, distribution = "coxph", param_grid = params_gbm3, modfun = gbm3.satpred, parallelize = F
    )
  } else {
    f <- formula_without_tvf_without_cluster
    state <- "no_TVF"
    print("without time-varying features")
  }
  
  params_gbm3 <- expand.grid(shrinkage = seq(0.05, 0.1, length.out = 3)
                           , n.trees = c(200, 500), n.minobsinnode = 10
                           , interaction.depth = 1)

  fit_gbm3 <- modfit(tuned_gbm3, return_data = TRUE)
  scurves_gbm3 <- get_indivsurv(fit_gbm3, train_df)
  concord_gbm3 <- get_survconcord(fit_gbm3)
  print(concord_gbm3)
  vimp_gbm3 <- get_varimp(fit_gbm3, type = "perm", newdata = train_df, nrep = 20, modelname = "gbm3")

  p1 <- plot(scurves_gbm3) + ggtitle("Survival Curves - GBM3")
  if(length(tuned_gbm3)!=0){
    p2 <- plot(tuned_gbm3)   + ggtitle("Tuning Results - GBM3")
  }
  p3 <- plot(vimp_gbm3)    + ggtitle("Variable Importance - GBM3")
  
  pdf_file <- file.path(outdir, paste0("summaries_for_each_model/gbm3_", state, ".pdf"))
  pdf(pdf_file, width = 14, height = 8)
  grid.newpage()
  summary_text <- paste0(
    "Model: GBM3 TVF\n\n",
    "Concordance: ", round(concord_gbm3, 6), "\n\n")
  grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"), gp = gpar(fontsize = 12))
  if(length(tuned_gbm3)!=0){
    gridExtra::grid.arrange(p1, p2, p3, ncol = 3)
  } else {
    gridExtra::grid.arrange(p1, p3, ncol = 3)
  }
  
  file_path_save <- paste0(outdir, "/models/model_gbm3_", state, ".RData")
  save(gbm3_fit, file = file_path_save)
  cat("\nModel saved to:", file_path_save, "\n")
  
  dev.off()
  
  cat("\nPDF report saved to:", pdf_file, "\n")
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 4 : DEEPSURV
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     calls split data, 
#     creates a grid of parameters to make a tuning plot,
#     fit the model with a deep learning model,
#     plots the individual survival curves, 
#     plots the variable of importance,
#     plots the tuning plot,
#     computes the concordance index,
#     saves the model as RData,
#     saves the plots as pdf
# 
# PARAMETERS
#     data frame, output directory, if tvf or no tvf (string), splitted data
# 
# RETURNS
#     a list made of the fitted model, the tuning plot, c-index and variables of importance df  
# ----------------------------------------------------------------------------------------------------
deepsurv_func <- function(df, outdir, tvf, splitted_data, seed = 8888) {
  train_df <- splitted_data$train
  test_df <- splitted_data$test
  
  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
    tuned_coxph <- list()
  } else {
    f <- formula_without_tvf_without_cluster
    state <- "no_TVF"
    print("without time-varying features")
  }
  
  params_deepsurv <- expand.grid(
    dropout = c(0.1, 0.4),
    learning_rate = seq(0.1, 0.01, length.out = 3),
    epochs = c(50)
  )
  num_nodes <- list(n1 = c(32, 64))
  
  tuned <- modtune(
    formula = f,
    data = train_df,
    num_nodes = num_nodes,
    param_grid = params_deepsurv,
    modfun = deepsurv.satpred,
    lr_decay = 0.001,
    parallelize = F,
    seed = seed
  )
  
  fit <- modfit(
    tuned,
    return_data = TRUE,
    early_stopping = FALSE
  )
  
  get_survconcord(fit, test_df)

  scurves <- get_indivsurv(fit, test_df)
  concord <- get_survconcord(fit, test_df)
  vimp <- get_varimp(fit, type = "perm", newdata = train_df, nrep = 20, modelname = "deepsurv")
  cat("\nConcordance for DeepSurv - no TVF:\n")
  print(concord)
    
  p1 <- plot(scurves) + ggtitle("Survival Curves - DeepSurv")
  tuned$data <- as.data.frame(tuned$data)
  p2 <- plot(tuned) + ggtitle("Tuning Results - DeepSurv")
  p3 <- plot(vimp)    + ggtitle("Variable Importance - DeepSurv")
    
  pdf_file_summary <- file.path(outdir, "summaries_for_each_model/DeepSurv_no_TVF.pdf")
  model_path <- file.path(outdir, "models/model_deepsurv_no_TVF.RData")
    
  pdf(pdf_file_summary, width = 14, height = 8)
  grid.newpage()
  summary_text <- paste0("Model: DeepSurv - no TVF\nConcordance: ", round(concord, 6), "\n")
  grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"), gp = gpar(fontsize = 12))
  gridExtra::grid.arrange(p1, p2, p3, ncol = 3)
  dev.off()
  cat("\nPDF report saved to:", pdf_file_summary, "\n")
  model_deepsurv = list(fit = fit, tuned = tuned, curves = scurves, concord = concord, vimp = vimp)
  save(model_deepsurv , file = model_path)
    
  return(list(fit = fit, tuned = tuned, curves = scurves, concord = concord, vimp = vimp))
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 4 : Penalized COX PH (RIDGE, LASSO AND ENET)
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     calls split data, 
#     defines if tvfs are taken into account,
#     fits the data to be used by pcoxtime,
#     extract the type of model based on alpha to save it under the right name,
#     fit the model with a penalized cox model,
#     gets the linear predictors,
#     computes variable importance, concordance index, individual survival curves,
#     save the plots as pdf in the target directory
# 
# PARAMETERS
#     data frame, alpha (type of penalization), lambda (intensity of penalization), nb of individuals used to plot the survival curves, output directory, if tvf or no tvf (string), splitted data
# 
# RETURNS
#     a list made of the fitted model, the tuning plot, the concordance index, the dataframe of coefficients, the variable importance, the individual survival curves, the train and test data frame   
# ----------------------------------------------------------------------------------------------------
pcox_func <- function(df, outdir, tvf, splitted_data, alpha, lambda = 0.1, seed = 8888, n_indiv = 30) {
  set.seed(seed)
  
  train_df <- splitted_data$train
  test_df <- splitted_data$test

  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
    tuned_coxph <- list()
  } else {
    f <- formula_without_tvf_without_cluster
    state <- "no_TVF"
    print("without time-varying features")
    tuned_coxph <- list()
  }
  
  pcox_fit <- pcoxtime(f, data = train_df, lambda=0.1, alpha=alpha)
  
  alpha_num <- pcox_fit$alpha
  model_type <- if(alpha_num == 0) {
    "ridge"
  } else if(alpha_num == 1) {
    "lasso"
  } else {
    "enet"
  }
  message("Fitting pcoxtime (alpha = ", alpha, ", lambda = ", lambda, ") ...")
  
  test_lp <- predict(pcox_fit, newdata = test_df)
  test_df <- test_df %>%
    dplyr::mutate(row_id = dplyr::row_number(),
                  risk_score = as.numeric(test_lp))
  
  concord <- tryCatch({
    pcoxtime::concordScore(pcox_fit, newdata = test_df)
  }, error = function(e) {
    warning("pcoxtime::concordScore failed: ", conditionMessage(e), " - falling back to survival::concordance()")
    cs <- survival::concordance(Surv(time, status) ~ I(-test_df$risk_score), data = test_df)
    if (is.list(cs) && !is.null(cs$concordance)) cs$concordance else cs
  })
  
  cv <- concord
  
  pcox_coef <- stats::coef(pcox_fit)
  coef_df <- data.frame(
    variable = names(pcox_coef),
    coef = as.numeric(pcox_coef),
    stringsAsFactors = FALSE
  )
  
  # Create regex patterns for each variable + manually remove time and status
  covariates_names <- names(df)
  covariates_names <- setdiff(covariates_names, c("time", "status"))
  var_map <- setNames(
    as.list(paste0("^", covariates_names)),
    covariates_names)
  
  coef_df$type <- NA
  for (v in names(var_map)) {
    coef_df$type[grepl(var_map[[v]], coef_df$variable)] <- v
  }
  coef_df <- coef_df %>% dplyr::filter(!is.na(type))
  vimp_type_df <- coef_df %>%
    dplyr::group_by(type) %>%
    dplyr::summarise(importance = sum(abs(coef)), .groups = "drop") %>%
    dplyr::arrange(desc(importance))
  
  vimp_plot <- ggplot2::ggplot(vimp_type_df, ggplot2::aes(x = reorder(type, importance), y = importance)) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::labs(title = paste0("Variable Importance by Type (pcoxtime, alpha=", alpha, ")"),
                  x = "Variable Type", y = "Total Absolute Coefficient") +
    ggplot2::theme_minimal(base_size = 14)
  
  cox_base <- survival::coxph(f, data = train_df)
  basehaz_df <- survival::basehaz(cox_base, centered = FALSE)
  time_seq <- seq(0, max(basehaz_df$time), length.out = 50)
  if (n_indiv > nrow(test_df)) n_indiv <- nrow(test_df)
  sample_ids <- sample(test_df$row_id, n_indiv, replace = FALSE)
  surv_df <- do.call(rbind, lapply(sample_ids, function(i) {
    H_interp <- approx(basehaz_df$time, basehaz_df$hazard, xout = time_seq, rule = 2)$y
    data.frame(
      time = time_seq,
      surv = exp(-H_interp * exp(test_df$risk_score[i])),
      id = i,
      stringsAsFactors = FALSE
    )
  }))
  surv_all_df <- do.call(rbind, lapply(test_df$row_id, function(i) {
    H_interp <- approx(basehaz_df$time, basehaz_df$hazard, xout = time_seq, rule = 2)$y
    data.frame(
      time = time_seq,
      surv = exp(-H_interp * exp(test_df$risk_score[i])),
      id = i,
      stringsAsFactors = FALSE
    )
  }))
  median_df <- surv_all_df %>%
    dplyr::group_by(time) %>%
    dplyr::summarise(surv_med = stats::median(surv, na.rm = TRUE), .groups = "drop")
    #dplyr::summarise(surv_mean = mean(surv, na.rm = TRUE), .groups = "drop")
  surv_plot <- ggplot2::ggplot() +
    ggplot2::geom_line(data = surv_df, ggplot2::aes(x = time, y = surv, group = id), alpha = 0.7, linetype = "dashed", color = "gray66", size = 0.5) +
    ggplot2::geom_line(data = median_df, ggplot2::aes(x = time, y = surv_med), size = 1.2, color = "red", size = 0.1) +
    ggplot2::labs(title = paste0("Individual Survival Curves (pcoxtime - alpha=", alpha, ")"),
                  x = "Time", y = "Survival Probability") +
    ggplot2::theme_minimal(base_size = 14)
  
  pdf_file <- file.path(outdir, paste0("summaries_for_each_model/pcoxtime_", model_type, "_", state,  ".pdf"))
 
  pdf(pdf_file, width = 14, height = 8)
  grid::grid.newpage()
  summary_text <- paste0(
    "Model type: ", model_type, " ", state, "\n\n",
    "Concordance index: ", round(concord, 6)
  )
  grid::grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"),
                  gp = grid::gpar(fontsize = 16, fontface = "bold"))
  gridExtra::grid.arrange(vimp_plot, surv_plot, ncol = 2)
  dev.off()
  
  cat("\nPDF report saved to:", pdf_file, "\n")
  
  assign(paste0("model_", model_type, "_", state), list(
    fit = pcox_fit,
    tuned = list(alpha = alpha, lambda = lambda),
    concord = concord,
    coef_df = coef_df,
    vimp_plot = vimp_plot,
    surv_plot = surv_plot,
    test_df = test_df,
    train_df = train_df
  ))
  
  var_name <- paste0("model_", model_type, "_", state)
  file_path <- paste0(outdir, "/models/", var_name, ".RData")
  save(list = var_name, file = file_path)
  cat("\nModel saved to:", file_path, "\n")
  
  result <- list(
    fit = pcox_fit,
    tuned = list(alpha = alpha, lambda = lambda),
    concord = concord,
    coef_df = coef_df,
    vimp_df = vimp_type_df,
    vimp_plot = vimp_plot,
    surv_plot = surv_plot,
    test_df = test_df,
    train_df = train_df,
    concord = concord
  )
  print(paste("Concordance index: ", round(concord, 6)))
  
  invisible(result)
}



# ----------------------------------------------------------------------------------------------------
#          FUNCTION 5 : COX PROPORTIONAL HAZARDS
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     calls split data, 
#     defines if tvfs are taken into account,
#     fit the model with a Cox Proportional Hazard model,
#     calls the postprocess_results function,
#     saves model as RData
# 
# PARAMETERS
#     data frame, output directory, if tvf or no tvf (string), splitted data
# 
# RETURNS
#     a list made of the fitted model, the tuning plot, c-index and variable of importance df
# ----------------------------------------------------------------------------------------------------
coxph_func <- function(df, outdir, tvf, splitted_data, seed = 8888) {
  train_df <- splitted_data$train
  test_df <- splitted_data$test
  
  if (tvf == TRUE){
    f <- formula_with_tvf_without_cluster
    state <- "TVF"
    print("with time-varying features")
    tuned_coxph <- list()
  } else {
    f <- formula_without_tvf_without_cluster
    state <- "no_TVF"
    print("without time-varying features")
    tuned_coxph <- list()
  }
  
  model_coxph = list(fit = fit, tuned = tuned_coxph)
  fit <- coxph(formula = f, data = train_df)
  
  postprocess_results(fit=fit, tuned=tuned_coxph, train_df=train_df, test_df=test_df, modelname = "CoxPH", outdir=outdir, state)
  path <- paste0(outdir,"/models/model_coxph_", state, ".RData" )
  print(path)
  save(model_coxph , file = path)
  cat("\nModel saved to:", path, "\n")
  
  concord <- get_survconcord(fit, test_df)
  vimp <- get_varimp(fit, type = "perm", newdata = train_df, nrep = 20, modelname = "CoxPH")
  result <- list(fit = fit, tuned = tuned_coxph, concord = concord, vimp = vimp)
  invisible(result)
}

#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️️️️️
#⚠️ DOESN'T WORK WITH TVFs YET⚠️
#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️
# ----------------------------------------------------------------------------------------------------
#          FUNCTION 6 : DEEPPAMM 
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     Deep learning model for survival analysis.
#     Define the covariates as the ones in veteran dataset
#     set up time set up to the largest value observed
#     Splits data set with splitdata()
#     Creates a new instance of the class deeppamm
#     Builds a deeppamm model and make it with parametrs passed as arguments
#     Train model
#     Give survival predictions with the methods PredictCumHaz() and predictSurvProb()
#     Computes risk score with cumulated risk hazard to get the concordance index
#     Computes individual survival curves
#     Export curves + index as a pdf stored in outputs
#
# PARAMETERS
#    Data set, epochs, batch size, output directory, if tvf or no tvf (string), splitted data
# 
# RETURNS
#     a list of the fitted model, the concordance index, the survival curves, and its corresponding plot, the time intervals used (for plotting and compare), the odf file containing the plots
# ----------------------------------------------------------------------------------------------------
deeppamm_func <- function(data, outdir, splitted_data, epochs = 1000, batch_size = 128, seed = 1223){
  if (!"id" %in% colnames(data)) {
    data <- data %>% mutate(id = 1:nrow(data))}

  covariates <- names(df)
  covariates <- setdiff(covariates, c("time", "status"))

  deep_covariates <- paste(covariates, collapse = ", ")
  f_str <- paste0(
    "Y ~ s(time) + ", paste(covariates, collapse = " + "), " + deep(", deep_covariates, ")")
  f <- as.formula(f_str)
  max_time <- 587
  #max_time <- 700
  #max_time <- max(data$time, na.rm = TRUE)
  cuts <- seq(0.1, max_time, length.out = 100)
  times <- seq(0.01, max_time, length.out = 250)
  t_star <- tail(times, 1)
  # for comparison
  DEEPPAMM_times <- times
  
  set.seed(seed)
  tr <- splitted_data$train
  te <- splitted_data$test
  # tr$t0 <- 0
  # te$t0 <- 0
  
  DEEPPAMM <- deeppamm$new(
    formulas = f,
    deep_architectures = list(
      deep = function(x) x %>%
        layer_dense(32, activation = "relu") %>% 
        layer_dropout(0.5) %>%
        layer_dense(32, activation = "relu") %>% 
        layer_dropout(0.5) %>%
        layer_dense(16, activation = "relu") %>% 
        layer_dense(1)
    ),
    data = tr,
    
    trafo_fct = Surv(time, status) ~ .,
    cut = cuts,
    lr = 0.001,
    scale = TRUE
  )
  
  DEEPPAMM$make_model(make_params())
  
  DEEPPAMM$train(
    epochs = epochs, 
    batch_size = batch_size, 
    callbacks = list(
      callback_reduce_lr_on_plateau(min_lr = 0.0001, factor = 0.5, patience = 50L),
      callback_early_stopping(patience = 100L)
    )
  )
  
  # y_pred <- DEEPPAMM$predictSurvProb(data[te, ] %>% select(-id), intervals = times)
  # cumulative_hazards_pred <- DEEPPAMM$predictCumHaz(data[te, ] %>% select(-id), intervals = times)
  y_pred <- DEEPPAMM$predictSurvProb(te, intervals = times)
  cumulative_hazards_pred <- DEEPPAMM$predictCumHaz(te, intervals = times)
  
  risk_score <- cumulative_hazards_pred[, length(times)]
  # y_time   <- data$time[te]
  # y_status <- data$status[te]
  y_time   <- te$time
  y_status <- te$status
  
  c_index <- concordance.index(
    x = risk_score,
    surv.time = y_time,
    surv.event = y_status,
    method = "noether"
  )
  cat("C-index:", round(c_index$c.index, 4), "\n")
  
  #y_pred_long <- melt(y_pred)
  y_pred_long <- reshape2::melt(y_pred)
  
  colnames(y_pred_long) <- c("Individual", "TimeIndex", "Survival")
  y_pred_long$Time <- times[y_pred_long$TimeIndex]
  avg_surv <- y_pred_long %>%
    group_by(Time) %>%
    summarize(AverageSurvival = mean(Survival))
  surv_plot <- ggplot() +
    geom_line(data = y_pred_long, aes(x = Time, y = Survival, group = Individual),
              color = "gray40", alpha = 0.2, linetype = "dashed") +
    geom_line(data = avg_surv, aes(x = Time, y = AverageSurvival),
              color = "red", linewidth = 1.2) +
    labs(x = "Time", y = "Predicted Survival Probability",
         title = "Individual Predicted Survival Curves with Average") +
    theme_minimal()
  
  modelname <- "deeppamm"
  pdf_file <- file.path(outdir, paste0("/summaries_for_each_model/",modelname, ".pdf"))
  pdf(pdf_file, width = 12, height = 8)
  grid.newpage()
  summary_text <- paste0(
    "Model: ", modelname, "\n\n",
    "Concordance Index: ", round(c_index$c.index, 4), "\n"
  )
  grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"),
            gp = gpar(fontsize = 12))
  grid.arrange(surv_plot, ncol = 1)
  dev.off()
  
  cat("\nPDF report saved to:", pdf_file, "\n")
  
  model_deeppamm = list(fit = DEEPPAMM,
                        concordance = c_index,
                        risk_scores = risk_score,
                        survival_curves = y_pred_long,
                        plot = surv_plot,
                        pdf_file = pdf_file,
                        intervals = times)
  file_path_save <- file.path(outdir, "/models/model_deeppamm.RData")
  save(model_deeppamm , file = file_path_save)
  
  return(list(
    fit = DEEPPAMM,
    concordance = c_index,
    risk_scores = risk_score,
    survival_curves = y_pred_long,
    plot = surv_plot,
    pdf_file = pdf_file,
    intervals = times
  ))
}



############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                  WRAPPER FUNCTIONS 
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################

# ----------------------------------------------------------------------------------------------------
#          Define predictSurvProb method for pcoxtime
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#   Method created to be able to use PEC for methods coming from pcoxtime
#   Needed because for pcoxtime model there are no built in predictSurvProb method as they are directly computed without needing a method
# 
# PARAMETERS
#    pcoxtime instance, test data set, time sample
# 
# RETURNS
#     -
# ----------------------------------------------------------------------------------------------------
predictSurvProb.pcoxtime <- function(object, newdata, times, ...) {
  lp <- predict(object, newdata = newdata)
  cox_fit <- survival::coxph(Surv(time, status) ~ ., data = newdata)
  basehaz_fit <- survival::basehaz(cox_fit, centered = FALSE)
  sapply(times, function(t) {
    exp(-approx(basehaz_fit$time, basehaz_fit$hazard, xout = t, rule = 2)$y * exp(lp))
  })
}


# ----------------------------------------------------------------------------------------------------
#          Define DeepPAMM_predict method for pec::Score()
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#    Wrapper created to be able to use pec::Score() for DeepPAMM objects (used in survival_metrics())
#    Converts DeepPAMM's native prediction format (survival prob on its own interval grid) into the std probability matrix that pec needs
#    Handles time interpolation
#    Heart of DeepPAMM - pec bridge
#   
# PARAMETERS
#    pcoxtime instance, test data set, time sample
# 
# RETURNS
#     probability matrix
# ----------------------------------------------------------------------------------------------------
DeepPAMM_predict <- function(object, newdata, times) {
  pred_surv <- object$fit$predictSurvProb(new_data = newdata)
  interval_times <- seq(from = min(object$fit$cut),
                        to   = max(object$fit$cut),
                        length.out = ncol(pred_surv))
  pred_interp <- t(apply(pred_surv, 1, function(x) {
    approx(x = interval_times, y = x, xout = times, method = "linear", rule = 2)$y
  }))
  pred_interp <- as.matrix(pred_interp)
  colnames(pred_interp) <- NULL
  rownames(pred_interp) <- NULL
  return(pred_interp)
}


#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️️⚠️⚠️⚠️⚠️⚠️️
#⚠️ ️NEEDS TO BE UNCOMMENTED TO USE PEC()️ ⚠️
#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️⚠️⚠️⚠⚠️⚠️
# ----------------------------------------------------------------------------------------------------
#          Wrapper which holds trained model and class tag for DeepPAMM objects
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#    pec can automatically call PredictSurvProb.deeppamm() and predictRisk.deeppamm()
#    Connects the methods with the trained deeppamm model
#   
# PARAMETERS
#    -
# RETURNS
#    -
# ----------------------------------------------------------------------------------------------------
# DeepPAMM_object <- list(fit = res_deeppamm$fit)
# class(DeepPAMM_object) <- "DeepPAMM"


#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️️️️️
#⚠️     NOT OPTIMAL YET       ⚠️
#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️
# ----------------------------------------------------------------------------------------------------
#          Survival probability computation : S3 wrapper function that translates the S3 call into R6 call
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#    Method created to be able to use S3 class (used for computing PEC) with a R6 object (deeppamm),
#    Calls the corresponding predictSurvProb contained in DeepPAMM instance,
#    uncomment the one needed for either Score() or Pec()
#
# PARAMETERS
#    R6 instance, test data set, time frame used for comparison
# 
# RETURNS
#     correct format for using PEC
# ----------------------------------------------------------------------------------------------------
# TO DEFINE WHICH ONE TO PICK - works for Score()
# predictSurvProb.DeepPAMM <- function(object, newdata, times, ...) {
#   if (!is.function(object$fit$predictSurvProb)) {
#     stop("DeepPAMM object does not contain a callable predictSurvProb method.")
#   }
#   times <- as.numeric(times)
#   pred_surv <- object$fit$predictSurvProb(new_data = newdata)
# 
#   interval_times <- seq(from = min(object$fit$cut),
#                         to   = max(object$fit$cut),
#                         length.out = ncol(pred_surv))
# 
#   pred_interp <- t(apply(pred_surv, 1, function(x) {
#     approx(x = interval_times, y = x, xout = times, method = "linear", rule = 2)$y
#   }))
# 
#   pred_interp <- as.matrix(pred_interp)
#   colnames(pred_interp) <- NULL
#   rownames(pred_interp) <- NULL
#   pred_interp
# }

# TO DEFINE WHICH ONE TO PICK - works for pec, apparently more robust??
# predictSurvProb.deeppamm <- function(object, newdata, times, ...) {
#   if (is.environment(object) && exists("fit", envir = object)) {
#     object <- get("fit", envir = object)
#   }
# 
#   times <- as.numeric(times)
# 
#   if ("deeppamm" %in% class(object) && "predictSurvProb" %in% names(object)) {
#     pred_surv <- object$predictSurvProb(new_data = newdata, intervals = times)
#   } else if (!is.null(object$fit) && "deeppamm" %in% class(object$fit)) {
#     pred_surv <- object$fit$predictSurvProb(new_data = newdata, intervals = times)
#   } else {
#     stop("No callable predictSurvProb method found in this object or its fit element.")
#   }
# 
#   pred_surv <- as.matrix(pred_surv)
#   colnames(pred_surv) <- NULL
#   rownames(pred_surv) <- NULL
#   pred_surv
# }



# ----------------------------------------------------------------------------------------------------
#          Risk prediction : S3 wrapper function that translates the S3 call into R6 call
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#    Converts survival prediction into a risk prediction
# 
# PARAMETERS
#    R6 instance, test data set, time frame used for comparison
# 
# RETURNS
#     -
# ----------------------------------------------------------------------------------------------------
predictRisk.DeepPAMM <- function(object, newdata, times, ...) {
  1 - predictSurvProb.DeepPAMM(object, newdata, times)
}
clean_vimp_df <- function(df) {
  if (is.null(df)) return(NULL)
  df <- as.data.frame(df)
  if (all(c("terms", "Overall") %in% colnames(df))) {
    df <- df %>% rename(Variable = terms, Importance = Overall)
  }
  if (all(c("type", "importance") %in% colnames(df))) {
    df <- df %>% rename(Variable = type, Importance = importance)
  }
  df <- df %>% select(any_of(c("Variable", "Importance")))
  return(df)
}



############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                  COMPARISON AND METRICS 
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################

#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️️️️️
#⚠️ DOESN'T WORK WITH TVFs YET⚠️
#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️
# ----------------------------------------------------------------------------------------------------
#          SURVIVAL METRICS
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     Extract the fit argument of the results obtained for each model
#     compute roc curve, auc and brier score for each model
#     plots the scores
# 
# PARAMETERS
#    data frame, results obtained for each model, name of the output file, time points for roc curve
# 
# RETURNS
#     pdf with all survival metrics plots
# ----------------------------------------------------------------------------------------------------
plot_survival_metrics <- function(df, models_list,
                                  outdir, eval_times = seq(50, 550, 50), tp_roc = 50) {
  
  covariates <- names(df)
  covariates <- setdiff(covariates, c("time", "status"))
  covariates_str <- paste(covariates, collapse = " + ")
  f <- formula_without_tvf_without_cluster
    
  data_splitted <- split_data(df) # used for interpolation
  
  models_list <- list(
    "Random Forest" = res_rfsrc$fit,
    "GBM"           = res_gbm$fit,
    "DeepSurv"      = res_deepsurv$fit,
    "CoxPH"         = res_coxph$fit,
    "Elastic Net"   = res_enet$fit,
    "Ridge"         = res_ridge$fit,
    "Lasso"         = res_lasso$fit,
    "DeepPAMM"      = DeepPAMM_object)
  
  good_times <- eval_times[eval_times < 400]
  print(good_times)
  
  for (m in names(models_list)) {
    if (is.null(models_list[[m]]$formula)) {
      models_list[[m]]$formula <- formula_without_tvf_without_cluster
    }
  }
  
  score_all <- Score(
    object = models_list,
    formula = formula_without_tvf_without_cluster,
    data = data_splitted$test,
    plots = c("brier", "auc", "roc"),
    times = good_times, #instead of eval_times
    reference = FALSE
  )
  
  pec_result <- pec(
    object = models_list,
    formula = formula_without_tvf_without_cluster,
    data = data_splitted$test,
    times = good_times
  )
  
  n_models <- length(models_list)
  colors <- brewer.pal(min(max(n_models, 3), 9), "Set1")
  colors <- rep(colors, length.out = n_models)
  colors_named <- setNames(adjustcolor(colors, alpha.f = 0.7), names(models_list))
  colors_named_brier <- c("Reference" = adjustcolor("tomato2", alpha.f = 0.8),
                          colors_named)
  
  linetypes <- 1:9
  linetypes <- rep(linetypes, length.out = n_models)
  linetypes_named <- setNames(linetypes, names(models_list))
  linetypes_named_brier <- c("Reference" = 1, linetypes_named)
  
  output_file <- file.path(outdir, "metrics/survival_metrics.pdf")
  pdf(output_file, width = 14, height = 6)
  
  if (length(tp_roc) == 1) {
    par(mfrow = c(1, 3), mar = c(4, 4, 5, 2))
    plotROC(score_all, times = tp_roc, col = colors_named, lty = linetypes_named, lwd = 2, legend = F)
    mtext(paste0("ROC at time = ", tp_roc), side = 3, line = 1.2, cex = 1.1, font = 2)
    plotAUC(score_all, col = colors_named, lty = linetypes_named, lwd = 2, legend = F, ylim = c(0, 1))
    mtext("Time-dependent AUC", side = 3, line = 1.5, cex = 1.2, font = 2)
    score_all$score$model <- as.character(score_all$score$model)
    score_all$contrasts$model <- as.character(score_all$contrasts$model)
    score_all$contrasts$reference <- as.character(score_all$contrasts$reference)
    models_to_plot <- names(colors_named_brier) # for the naughty naughty legend that drove me crazy
    plotBrier(score_all, col = colors_named_brier, lty = linetypes_named_brier, lwd = 2, legend = F, ylim = c(0, 0.3))
    mtext("Brier score over time", side = 3, line = 1.5, cex = 1.2, font = 2)
    graphics::legend("topright",
                     legend = models_to_plot,
                     col = colors_named_brier,
                     lty = linetypes_named_brier,
                     lwd = 2)
  } else {
    par(mfrow = c(1, 2), mar = c(4, 4, 5, 2))
    plotAUC(score_all, col = colors_named, lty = linetypes_named, lwd = 2, legend = F, ylim = c(0, 1))
    mtext("Time-dependent AUC", side = 3, line = 1.5, cex = 1.2, font = 2)
    score_all$score$model <- as.character(score_all$score$model)
    score_all$contrasts$model <- as.character(score_all$contrasts$model)
    score_all$contrasts$reference <- as.character(score_all$contrasts$reference)
    models_to_plot <- names(colors_named_brier)
    plotBrier(score_all, col = colors_named_brier, lty = linetypes_named_brier, lwd = 2, legend = F, ylim = c(0, 0.3))
    mtext("Brier score over time", side = 3, line = 1.5, cex = 1.2, font = 2)
    graphics::legend("topright",
                     legend = models_to_plot,
                     col = colors_named_brier,
                     lty = linetypes_named_brier,
                     lwd = 2)
    par(mfrow = c(1, min(3, length(tp_roc))), mar = c(4, 4, 5, 2))
    for (t in tp_roc) {
      plotROC(score_all, times = t, col = colors_named, lty = linetypes_named, lwd = 2, legend = (t == tp_roc[1]))
      mtext(paste0("ROC at time = ", t), side = 3, line = 1.2, cex = 1.1, font = 2)
    }
  }
  dev.off()
  message("✅ PDF saved to: ", output_file)
}



#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️️️️️
#⚠️ DOESN'T WORK WITH TVFs YET⚠️
#️️️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️️
# ----------------------------------------------------------------------------------------------------
#          PREDICTION ERROR CURVE FOR MULTIPLE MODELS
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#    Compute the PEC for all models
#    Identifies how to handle each model regarding their survival prediction construction
# 
# PARAMETERS
#    list of results, test data set, time sequence
# 
# RETURNS
#     PEc plot
# ----------------------------------------------------------------------------------------------------
compute_pec_all <- function(df, res_rfsrc, res_gbm, res_deepsurv, res_coxph, res_enet, res_ridge, res_lasso, res_deeppamm, times = seq(50, 500, 50)) {
  
  models_list <- list(
    "RandomForestSRC" = res_rfsrc,
    "GBM"             = res_gbm,
    "DeepSurv"        = res_deepsurv,
    "CoxPH"           = res_coxph,
    "ElasticNet"      = res_enet,
    "Ridge"           = res_rigde,
    "Lasso"           = res_lasso, 
    "DeepPAMM"        = res_deeppamm
  )
  test_data <- split_data(df)$test
  pec_models <- list()
  covariates <- names(df)
  covariates <- setdiff(covariates, c("time", "status"))
  covariates_str <- paste(covariates, collapse = " + ")
  surv_formula <- as.formula(paste("Surv(time, status) ~", covariates_str))
  for (model_name in names(models_list)) {
    fit_obj <- models_list[[model_name]]$fit
    if (inherits(fit_obj, "rfsrc") || inherits(fit_obj, "gbm") ||
        inherits(fit_obj, "deepsurv") || inherits(fit_obj, "coxph") ||
        inherits(fit_obj, "deeppamm")) {
      pec_models[[model_name]] <- fit_obj
    } else if (inherits(fit_obj, "pcoxtime")) {
      class(fit_obj) <- c("pcoxtime", class(fit_obj))
      pec_models[[model_name]] <- fit_obj
    } else {
      warning("Unknown model type for PEC: ", model_name)
    }
  }
  pec_res <- pec(
    object = pec_models,
    formula = surv_formula,
    data = test_data,
    times = times,
    exact = FALSE,
    reference = F
  )
  print(pec_res)
  plot(pec_res, legend = F, main = "Prediction Error Curves (PEC)")
  model_names <- names(pec_res$AppErr)
  legend("topright",
         legend = names(models_list),
         col = 1:length(models_list),
         lty = 1,
         lwd = 2,
         bty = "n")
  
  return(pec_res)
}



############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                  POST-PROCESSING 
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################

# ----------------------------------------------------------------------------------------------------
#          POST-PROCESSING
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     gives individual survival curves, concordance score and variable of importance score and plots
#     save the plots as pdf and saved in output folder
# 
# PARAMETERS
#     fitted model, tuning plot, test data frame, model name (), target directory where the plots are to be stored
# 
# RETURNS
#     pdf report saved in target directory
# ----------------------------------------------------------------------------------------------------
postprocess_results <- function(fit, tuned, train_df, test_df, modelname, outdir, state) {
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  
  scurves <- get_indivsurv(fit, test_df)
  concord <- get_survconcord(fit, test_df)
  vimp <- get_varimp(fit, type = "perm", newdata = train_df, nrep = 20, modelname = modelname)
  cat("\nConcordance for", modelname, " ", state, ":\n")
  print(concord)
  
  p1 <- plot(scurves) + ggtitle(paste("Survival Curves -", modelname))
  if(length(tuned)!=0){
    p2 <- plot(tuned)   + ggtitle(paste("Tuning Results -", modelname))
  }
  p3 <- plot(vimp)    + ggtitle(paste("Variable Importance -", modelname))
  
  
  pdf_file <- file.path(outdir, paste0("summaries_for_each_model/", modelname, "_", state, ".pdf"))
  pdf(pdf_file, width = 14, height = 8)
  grid.newpage()
  summary_text <- paste0(
    "Model: ", modelname, "\n\n",
    "Concordance: ", round(concord, 6), "\n\n"
  )

  grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"), gp = gpar(fontsize = 12))
  if(length(tuned)!=0){
    gridExtra::grid.arrange(p1, p3, ncol = 2)
  } else {
    gridExtra::grid.arrange(p1, p3, ncol = 2)
  }
  
  dev.off()
  cat("\nPDF report saved to:", pdf_file, "\n")
  invisible(list(curves = scurves, concord = concord, vimp = vimp))
}



# ----------------------------------------------------------------------------------------------------
#          POST-PROCESSING - MISSING VALUES
# ----------------------------------------------------------------------------------------------------
# DESCRIPTION
#     gives individual survival curves, concordance score and variable of importance score and plots
#     save the plots as pdf and saved in output folder
# 
# PARAMETERS
#     results obtained after applying model function, target directory
# 
# RETURNS
#     pdf report saved in target directory
# ----------------------------------------------------------------------------------------------------
postprocess_comparison <- function(results,
                                   outdir = "C:/Users/capuc/Desktop/ETHZ/researchproject/scripts/satpred/outputs",
                                   filename = "comparison_all_perc_RFSRC.pdf") {
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  
  pdf_file <- file.path(outdir, filename)
  pdf(pdf_file, width = 14, height = 8)
  concord_table <- data.frame(
    Percentage = numeric(),
    Concordance = numeric())
  for (nm in names(results)) {
    res <- results[[nm]]
    if (is.null(res)) {
      message("⚠️ Skipping ", nm, " because training failed.")
      next}
    fit   <- res$fit
    tuned <- res$tuned
    train_df <- res$train
    test_df  <- res$test
    perc  <- res$percentage
    modelname <- paste0("simulating_", perc, "perc_NA_data_RFSRC")
    
    scurves <- get_indivsurv(fit, test_df)
    concord <- get_survconcord(fit, test_df)
    vimp    <- get_varimp(fit, type = "perm", newdata = train_df, nrep = 20, modelname = modelname)
    
    p1 <- plot(scurves) + ggtitle(paste("Survival Curves -", modelname))
    p2 <- plot(tuned)   + ggtitle(paste("Tuning Results -", modelname))
    p3 <- plot(vimp)    + ggtitle(paste("Variable Importance -", modelname))
    
    grid.newpage()
    summary_text <- paste0(
      "Model: ", modelname, "\n\n",
      "Concordance: ", round(concord, 6), "\n\n")
    grid.text(summary_text, x = 0.05, y = 0.95, just = c("left", "top"), gp = gpar(fontsize = 12))
    grid.arrange(p1, p2, p3, ncol = 3)
    
    concord_table <- rbind(concord_table, data.frame(Percentage = perc, Concordance = concord))}
  
  if (nrow(concord_table) > 0) {
    grid.newpage()
    grid.table(concord_table)
  } else {
    grid.newpage()
    grid.text("⚠️ No successful runs to display concordance.", gp = gpar(fontsize = 14))}
  
  dev.off()
  cat("\nCombined PDF report saved to:", pdf_file, "\n")
  invisible(concord_table)
}
