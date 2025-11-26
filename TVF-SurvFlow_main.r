source("C:\\Users\\capuc\\Desktop\\ETHZ\\researchproject\\scripts\\satpred\\satpred_CancerSurvival_functions_v3.2.3_DD - Copie.R")

library(survival)
outdir <- "/Users/capuc/Desktop/ETHZ/researchproject/scripts/satpred/figures"
create_output_folder(outdir = outdir)

cgd0=cgd0
dim(cgd0)
newcgd <- tmerge(data1=cgd0[, 1:13], data2=cgd0, id=id, tstop=futime)
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime1))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime2))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime3))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime4))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime5))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime6))
newcgd <- tmerge(newcgd, cgd0, id=id, infect = event(etime7))
df <- tmerge(newcgd, newcgd, id, enum=cumtdc(tstart))
names(df)[names(df) == "infect"] <- "status"
df <- df[, !names(df) %in% c("futime", "center","hos.cat", "enum","random" )]
var <- setdiff(colnames(df), c("id","tstart", "tstop", "status"))
formula_with_tvf <- as.formula(paste("Surv(tstart, tstop, status) ~",paste(var, collapse = " + "), "+ cluster(id)"))
formula_without_tvf <- as.formula(paste("Surv(tstart, tstop, status) ~",paste(var, collapse = " + "), "+ cluster(id)"))
formula_with_tvf_without_cluster <- as.formula(paste("Surv(tstart, tstop, status) ~",paste(var, collapse = " + ")))
formula_without_tvf_without_cluster <- as.formula(paste("Surv(tstop, status) ~",paste(var, collapse = " + ")))

# MODEL TRAINING ----------------------------------------------------------------------------------

splitted_data=split_data(df,train_prop = 0.8, seed=8888)

# TVF models
res_coxph_tvf <- model(df, "coxph", outdir, tvf = T, splitted_data)
res_enet_tvf <- model(df, "enet", outdir, tvf = T ,splitted_data)
res_ridge_tvf <- model(df, "ridge", outdir, tvf = T,splitted_data)
res_lasso_tvf <- model(df, "lasso", outdir, tvf = T,splitted_data)
res_clatree_tvf <- model(df, "clatree", outdir, tvf = T, splitted_data) 
res_gbm3_tvf <- model(df, "gbm3", outdir, tvf = T, splitted_data)

# no TVF
res_coxph_no_tvf <- model(df, "coxph", outdir, tvf = F, splitted_data)
res_enet_no_tvf <- model(df, "enet", outdir, tvf = F ,splitted_data)
res_ridge_no_tvf <- model(df, "ridge", outdir, tvf = F,splitted_data)
res_lasso_no_tvf <- model(df, "lasso", outdir, tvf = F,splitted_data)
res_gbm_no_tvf <- model(df, "gbm", outdir, tvf = F, splitted_data)
res_deepsurv_no_tvf <- model(df, "deepsurv", outdir, tvf=F, splitted_data)
res_rfsrc_no_tvf <- model(df, "rfsrc", outdir, tvf = F, splitted_data)
res_clatree_no_tvf <- model(df, "clatree", outdir, tvf = F, splitted_data)


# VARIMP ----------------------------------------------------------------------------------
library(purrr)
library(ggplot2)
# TVF
vimp_df_coxph_tvf <- res_coxph_tvf$vimp
vimp_df_enet_tvf <- res_enet_tvf$vimp_df
vimp_df_ridge_tvf <- res_ridge_tvf$vimp_df
vimp_df_lasso_tvf <- res_lasso_tvf$vimp_df
vimp_df_clatree_tvf <- res_clatree_tvf$vimp_df
vimp_df_gbm3_tvf <- res_gbm3_tvf$vimp
# no TVF
vimp_df_coxph_no_tvf <- res_coxph_no_tvf$vimp
vimp_df_enet_no_tvf <- res_enet_no_tvf$vimp_df
vimp_df_ridge_no_tvf <- res_ridge_no_tvf$vimp_df
vimp_df_lasso_no_tvf <- res_lasso_no_tvf$vimp_df
vimp_df_gbm_no_tvf <- res_gbm_no_tvf$vimp
vimp_df_deepsurv_no_tvf <- res_deepsurv_no_tvf$vimp # drop sign column
vimp_df_rfsrc_no_tvf <- res_rfsrc_no_tvf$vimp # drop sign column
vimp_df_clatree_no_tvf <- res_clatree_no_tvf$vimp_df
vimp_df_gbm3_no_tvf  <- res_gbm3_no_tvf$vimp

vimp_list <- list(
  # coxph_tvf      = vimp_df_coxph_tvf,
  # enet_tvf       = vimp_df_enet_tvf,
  # ridge_tvf      = vimp_df_ridge_tvf,
  # lasso_tvf      = vimp_df_lasso_tvf,
  # clatree_tvf    = vimp_df_clatree_tvf,
  # gbm3_tvf       = vimp_df_gbm3_tvf,
  coxph_no_tvf   = vimp_df_coxph_no_tvf,
  enet_no_tvf    = vimp_df_enet_no_tvf,
  ridge_no_tvf   = vimp_df_ridge_no_tvf,
  lasso_no_tvf   = vimp_df_lasso_no_tvf,
  gbm_no_tvf     = vimp_df_gbm_no_tvf,
  deepsurv_no_tvf= vimp_df_deepsurv_no_tvf,
  rfsrc_no_tvf   = vimp_df_rfsrc_no_tvf,
  clatree_no_tvf = vimp_df_clatree_no_tvf,
  gbm3_no_tvf    = vimp_df_gbm3_no_tvf
)

vimp_cleaned <- lapply(names(vimp_list), function(name) {
  df <- clean_vimp_df(vimp_list[[name]])
  if (is.null(df)) return(NULL)
  if (!"Importance" %in% names(df)) df$Importance <- 0
  colnames(df) <- c("Variable", name)
  return(df)
})

vimp_wide <- reduce(vimp_cleaned, full_join, by = "Variable")
vimp_wide <- vimp_wide %>%
  mutate(
    #gbm3_tvf = ifelse(Variable == "age", 1.0, gbm3_tvf),
    gbm3_no_tvf = ifelse(Variable == "age", 1.0, gbm3_no_tvf)
  )
write.csv(vimp_wide, file.path(outdir, "variable_importance_wide_all_models.csv"), row.names = FALSE)
vimp_long <- vimp_wide %>%
  { if ("mean_importance" %in% names(.)) select(., -mean_importance) else . } %>%
  pivot_longer(
    cols = -Variable,
    names_to = "model",
    values_to = "importance")

ggplot(vimp_long, aes(x = model, y = importance, fill = Variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Variable Importance by Model",
    x = "Model",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right")



# VARIMP - HEAT MAP ----------------------------------------------------------------------------------
library(reshape2)
data <- read.csv("/Users/capuc/Desktop/ETHZ/researchproject/scripts/satpred/figures/varimp.csv", row.names = 1)
data <- read.csv("/Users/capuc/Desktop/ETHZ/researchproject/scripts/satpred/figures/varimp_with_gbm3_satpred.csv", sep = ";", row.names = 1)
data_long <- reshape2::melt(as.matrix(data))
palette <- c("white", "#54278f", "#756bb1", "#9e9ac8", "#cbc9e2", "#f2f0f7")
ggplot(data_long, aes(x = Var2, y = Var1, fill = factor(value))) +
  geom_tile(color = "grey80") +
  scale_fill_manual(
    values = palette,
    name = "Importance rank",
    breaks = c(0, 1, 2, 3, 4, 5),
    labels = c("not important", "1", "2", "3", "4", "5")
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size =14),
    axis.text.y = element_text(
      size = 14       # <<< bigger variable names
    ),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 13),
    plot.title   = element_text(size = 18),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right"
  ) +
  ggtitle("Variable Importance Across Models")




# CONCORDANCE PLOT ----------------------------------------------------------------------------------
cv_enet_tvf <- c()
cv_lasso_tvf <- c()
cv_ridge_tvf <- c()
cv_coxph_tvf <- c()
cv_gbm3_tvf <- c(0.6131645, 0.6966538, 0.6787021)
cv_clatree_tvf <- c()

cv_enet_no_tvf <- c()
cv_lasso_no_tvf <- c()
cv_ridge_no_tvf <- c()
cv_coxph_no_tvf <- c()
cv_gbm3_no_tvf <- c()
cv_gbm_no_tvf <- c()
cv_deepsurv_no_tvf <- c()
cv_clatree_no_tvf <- c()
cv_rfsrc_no_tvf <- c()

seeds <- c(1, 8888, 567)
  
for(seed in seeds){
  splitted_data=split_data(df,train_prop = 0.8, seed=8888)
  res_gbm3_tvf <- gbm3_sat_func(df, outdir, tvf = T, splitted_data, seed = seed)
  cv_gbm3_tvf <- c(cv_gbm3_tvf, res_gbm3_tvf$concord)
  
}

# Run model training over all three seeds
for (seed in seeds){ 
  splitted_data=split_data(df,train_prop = 0.8, seed=567)
  head(splitted_data$train)
  res_enet_tvf <- pcox_func(df, outdir, tvf = T, splitted_data, alpha = 0.5, lambda = 0.1, seed = 8888, n_indiv = 30)
  cv_enet_tvf <- c(cv_enet_tvf, res_enet_tvf$concord)
  res_enet_no_tvf <- pcox_func(df, outdir, tvf = F, splitted_data, alpha = 0.5, lambda = 0.1, seed = 8888, n_indiv = 30)
  cv_enet_no_tvf <- c(cv_enet_no_tvf, res_enet_no_tvf$concord)
  res_ridge_tvf <- pcox_func(df, outdir, tvf = T, splitted_data, alpha = 0, lambda = 0.1, seed = 8888, n_indiv = 30)
  cv_ridge_tvf <- c(cv_ridge_tvf, res_ridge_tvf$concord)
  res_ridge_no_tvf <- pcox_func(df, outdir, tvf = F, splitted_data, alpha = 0, lambda = 0.1, seed = 8888, n_indiv = 30)
  cv_ridge_no_tvf <- c(cv_ridge_no_tvf, res_ridge_no_tvf$concord)
  res_lasso_tvf <- pcox_func(df, outdir, tvf = T, splitted_data, alpha = 1, lambda = 0.1, seed = 8888, n_indiv = 30)
  cv_lasso_tvf <- c(cv_lasso_tvf, res_lasso_tvf$concord)
  res_lasso_no_tvf <- pcox_func(df, outdir, tvf = F, splitted_data, alpha = 1, lambda = 0.1, seed = 8888, n_indiv = 30)
  cv_lasso_no_tvf <- c(cv_lasso_no_tvf, res_lasso_no_tvf$concord)
  res_clatree_tvf <- clatree_func(df, outdir, tvf = T, splitted_data, seed = 8888)
  cv_clatree_tvf <- c(cv_clatree_tvf, res_clatree_tvf$concord)
  res_clatree_no_tvf <- clatree_func(df, outdir, tvf = F, splitted_data, seed = 8888)
  cv_clatree_no_tvf <- c(cv_clatree_no_tvf, res_clatree_no_tvf$concord)
  res_gbm3_tvf <- gbm3_func(df, outdir, tvf = T, splitted_data, seed = 8888)
  cv_gbm3_tvf <- c(cv_gbm3_tvf, res_gbm3_tvf$concord)
  res_deepsurv_no_tvf <- deepsurv_func(df, outdir, tvf = F, splitted_data, seed = 8888)
  cv_deepsurv_no_tvf <- c(cv_deepsurv_no_tvf, res_deepsurv_no_tvf$concord)
  res_coxph_tvf <- coxph_func(df, outdir, tvf = T, splitted_data, seed = 8888)
  cv_coxph_tvf <- c(cv_coxph_tvf, res_coxph_tvf$concord)
  res_coxph_no_tvf <- coxph_func(df, outdir, tvf = F, splitted_data, seed = 8888)
  cv_coxph_no_tvf <- c(cv_coxph_no_tvf, res_coxph_no_tvf$concord)
  res_rfsrc_no_tvf <- rfsrc_func(df, outdir, tvf = F, splitted_data, seed = 8888)
  cv_rfsrc_no_tvf <- c(cv_rfsrc_no_tvf, res_rfsrc_no_tvf$concord)
  res_gbm_no_tvf <- gbm_func(df, outdir, tvf = F, splitted_data, seed = 8888)
  cv_gbm_no_tvf <- c(cv_gbm_no_tvf, res_gbm_no_tvf$concord)
 
}
  save(cv_enet_tvf, cv_enet_no_tvf, cv_ridge_tvf, cv_ridge_no_tvf, cv_lasso_tvf, cv_lasso_no_tvf, cv_clatree_tvf, cv_clatree_no_tvf, cv_gbm3_tvf, cv_gbm3_no_tvf, cv_deepsurv_no_tvf, cv_coxph_tvf, cv_coxph_no_tvf, cv_rfsrc_no_tvf, cv_gbm_no_tvf, 
       file = "/Users/capuc/Desktop/ETHZ/researchproject/scripts/satpred/figures/c_index_vectors_nogbm3notvf.RData")
  
  # PLOT ALL CV PLOTS ----------------------------------------------------------------------------------
  
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  
  # Create a named list of all c-index vectors
  cv_list <- list(
    "ENet_TVF" = cv_enet_tvf,
    "ENet_noTVF" = cv_enet_no_tvf,
    "Ridge_TVF" = cv_ridge_tvf,
    "Ridge_noTVF" = cv_ridge_no_tvf,
    "Lasso_TVF" = cv_lasso_tvf,
    "Lasso_noTVF" = cv_lasso_no_tvf,
    "Clatree_TVF" = cv_clatree_tvf,
    "Clatree_noTVF" = cv_clatree_no_tvf,
    "GBM3_TVF" = cv_gbm3_tvf,
    "DeepSurv_noTVF" = cv_deepsurv_no_tvf,
    "CoxPH_TVF" = cv_coxph_tvf,
    "CoxPH_noTVF" = cv_coxph_no_tvf,
    "RSF_noTVF" = cv_rfsrc_no_tvf,
    "GBM_noTVF" = cv_gbm_no_tvf
  )
  cv_df <- cv_list %>%
    lapply(function(x) data.frame(c_index = x)) %>%
    bind_rows(.id = "model")
  summary_df <- cv_df %>%
    group_by(model) %>%
    summarise(
      mean_c = mean(c_index),
      se = sd(c_index)/sqrt(n()),          # standard error
      lower = mean_c - 1.96*se,            # 95% CI lower
      upper = mean_c + 1.96*se             # 95% CI upper
    )
  ggplot(summary_df, aes(x = reorder(model, mean_c), y = mean_c)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3) +
    geom_text(aes(label = sprintf("%.3f", mean_c), y = mean_c + 0.01),  # adjust vertical offset
              size = 4) +
    coord_flip() +
    labs(title = "Average Concordance Index per Model",
         x = "Model",
         y = "Mean Concordance Index (C-index)") +
    theme_minimal(base_size = 14)
  
  # PLOT CV FOR MWITH TVF ----------------------------------------------------------------------------------
  
  cv_list_tvf <- list(
    "ENet_TVF" = cv_enet_tvf,
    "Ridge_TVF" = cv_ridge_tvf,
    "Lasso_TVF" = cv_lasso_tvf,
    "Clatree_TVF" = cv_clatree_tvf,
    "GBM3_TVF" = cv_gbm3_tvf,
    "CoxPH_TVF" = cv_coxph_tvf
  )
  
  # Convert to a long-format data frame
  cv_df_tvf <- cv_list_tvf %>%
    lapply(function(x) data.frame(c_index = x)) %>%
    bind_rows(.id = "model")
  
  summary_df_tvf <- cv_df_tvf %>%
    group_by(model) %>%
    summarise(
      mean_c = mean(c_index),
      se = sd(c_index)/sqrt(n()),          # standard error
      lower = mean_c - 1.96*se,            # 95% CI lower
      upper = mean_c + 1.96*se             # 95% CI upper
    )
  
  ggplot(summary_df_tvf, aes(x = reorder(model, mean_c), y = mean_c)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3) +
    #coord_flip() +
    geom_text(
      aes(label = sprintf("%.3f", mean_c), y = upper + 0.02),  # label above CI
      size = 4
    ) +
    scale_y_continuous(limits = c(0, 0.8)) +
    labs(title = "with TVF",
         x = "Model",
         y = "Mean Concordance Index (C-index)") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # NO TVF ----------------------------------------------------------------------------------
  
  cv_list_no_tvf <- list(
    "ENet_noTVF" = cv_enet_no_tvf,
    "Ridge_noTVF" = cv_ridge_no_tvf,
    "Lasso_noTVF" = cv_lasso_no_tvf,
    "Clatree_noTVF" = cv_clatree_no_tvf,
    #"GBM3_noTVF" = cv_gbm3_no_tvf,
    "DeepSurv_noTVF" = cv_deepsurv_no_tvf,
    "CoxPH_noTVF" = cv_coxph_no_tvf,
    "RSF_noTVF" = cv_rfsrc_no_tvf,
    "GBM_noTVF" = cv_gbm_no_tvf
  )
  
  # Convert to a long-format data frame
  cv_df_no_tvf <- cv_list_no_tvf %>%
    lapply(function(x) data.frame(c_index = x)) %>%
    bind_rows(.id = "model")
  
  summary_df_no_tvf <- cv_df_no_tvf %>%
    group_by(model) %>%
    summarise(
      mean_c = mean(c_index),
      se = sd(c_index)/sqrt(n()),          # standard error
      lower = mean_c - 1.96*se,            # 95% CI lower
      upper = mean_c + 1.96*se             # 95% CI upper
    )
  
  ggplot(summary_df_no_tvf, aes(x = reorder(model, mean_c), y = mean_c)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3) +
    #coord_flip() +
    geom_text(
      aes(label = sprintf("%.3f", mean_c), y = upper + 0.02),  # label above CI
      size = 4
    ) +
    scale_y_continuous(limits = c(0, 0.8)) +
    labs(title = "without TVF",
         x = "Model",
         y = "Mean Concordance Index (C-index)") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
