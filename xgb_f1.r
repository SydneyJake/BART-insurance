library(caret)
library(xgboost)
library(tidyverse)

set.seed(7411)

# Functions that calculates F1 score from a prediction
calc_F1 <- function(probs, labels, thresh){
    # convert probabilities to labels
    p_labs <- as.numeric(probs > thresh)

    # Calculate true positives, false positives, and false negatives
    TP <- sum(p_labs & labels)  # the number of correct "positive" predictions
    FP <- sum(p_labs & !labels) # the number of incorrect "positive" predictions
    FN <- sum(!p_labs & labels) # the number of "negative" predictions that should be "positive"

    prec   <- TP/(TP+FP)
    recall <- TP/(TP+FN)

    return(  2 * prec * recall / (prec + recall) )
}


# read the data
data <- read_csv("data.csv",
                col_types = cols(income                           = col_double(),
                                 name_email_similarity            = col_double(),
                                 customer_age                     = col_double(),
                                 days_since_request               = col_double(),
                                 intended_balcon_amount           = col_double(),
                                 payment_type                     = col_factor(),
                                 zip_count_4w                     = col_double(),
                                 velocity_6h                      = col_double(),
                                 velocity_24h                     = col_double(),
                                 velocity_4w                      = col_double(),
                                 bank_branch_count_8w             = col_double(),
                                 date_of_birth_distinct_emails_4w = col_double(),
                                 employment_status                = col_factor(),
                                 credit_risk_score                = col_double(),
                                 email_is_free                    = col_factor(),
                                 housing_status                   = col_factor(),
                                 phone_home_valid                 = col_factor(),
                                 phone_mobile_valid               = col_factor(),
                                 bank_months_count                = col_double(),
                                 has_other_cards                  = col_factor(),
                                 proposed_credit_limit            = col_double(),
                                 foreign_request                  = col_factor(),
                                 source                           = col_factor(),
                                 session_length_in_minutes        = col_double(),
                                 device_os                        = col_factor(),
                                 keep_alive_session               = col_factor(),
                                 device_distinct_emails_8w        = col_factor(),
                                 month                            = col_factor()))

data <- data[,-1] # drop first index column

# Create a train test split that has similar class representation for both partitions
train_idx <- createDataPartition(data$fraud_bool, p=0.8, list=F, times=1)

# Train and test split
train <- data[train_idx,]
test  <- data[-train_idx,]

# One hot encoding
encoded_train <- model.matrix(~.-1, data=select(train, -c(fraud_bool)))
encoded_test  <- model.matrix(~.-1, data=select(test, -c(fraud_bool)))

# Trees
n_trees <- seq(10, 100, by=10)
N <- length(n_trees)

# features
F_n <- ncol(encoded_train)
feature_idxs <- c(2:F_n)

# threshold values
thresh_vals <-seq(0.05, 0.5, by=0.05)

# Storing the result
result <- array(0, dim = c(N, length(feature_idxs), length(thresh_vals)))

# param grid for cross validation.
param <- expand_grid(max_delta_step    = 1,
                     eta               = c(0.01, 0.1, 0.3),
                     gamma             = 0,
                     colsample_bytree  = 1,
                     subsample         = 1,
                     min_child_weight  = 1,
                     max_depth         = c(3, 6),
                     objective         = "binary:logistic",
                     eval_metric       = "aucpr")

# store cv metric
cv_metric <- vector(mode="numeric", length=nrow(param))

# Outer, over trees
for(n in seq_len(N)) {
  # Inner, over permuted features
  for(f in seq_len(length(feature_idxs))) {

    # Set up the data to be fit
    this_train <- encoded_train[, c(1:feature_idxs[f])] # take first col since output
    this_train <- xgb.DMatrix(data=this_train, label=train$fraud_bool)

    # cross validate the fit
    for(p in seq_len(nrow(param))) {
      cv_fit <- xgb.cv(params = as.list(param[p,]), data = this_train, nrounds=n_trees[n], nfold = 3, nthread=4, seed=3196)
      cv_metric[p] <- cv_fit$evaluation_log$test_aucpr_mean[n_trees[n]]
    }

    # Take the best parameter config (largest aucpr)
    best_pars <- as.list(param[which.max(cv_metric),])

    # and fit
    fit <- xgboost(data           = this_train,
                   nthread        = 4,
                   params         = best_pars,
                   nrounds        = n_trees[n])

     # predict some uniform sample of the test set
     this_test  <- encoded_test[, c(1:feature_idxs[f])]
     this_test  <- xgb.DMatrix(data=this_test, label=test$fraud_bool)

     # Calculate the F1 score
     probs <- predict(fit, newdata=this_test)
     result[n, f, ] <- sapply(thresh_vals, FUN =function(x) calc_F1(probs, test$fraud_bool, x))
    }
}
# save the result
saveRDS(result, "XGB_F1_scores_cv.rds")
