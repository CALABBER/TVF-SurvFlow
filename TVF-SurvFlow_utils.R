# ---------------------------------------------------------------------------
# TVF-SurvFlow_utils.R
#
# Pure utility functions extracted from TVF-SurvFlow_functions.R for unit
# testing. These depend only on base R + dplyr, so they load WITHOUT the heavy
# modelling stack (satpred / keras / tensorflow / randomForestSRC / ...).
#
# Keep these definitions in sync with TVF-SurvFlow_functions.R.
# ---------------------------------------------------------------------------

library(dplyr)

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
