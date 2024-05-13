library(caret)
library(dbarts)
library(tidyverse)
library(HDInterval)

# Functions that calculates F1 score from a prediction
calc_F1 <- function(hdis, labels, thresh){
    # If lower hdi is larger than threshold, set positive
    p_labs <- hdis[1,] > thresh

    # Calculate true positives, false positives, and false negatives
    TP <- sum(p_labs & labels)  # the number of correct "positive" predictions
    FP <- sum(p_labs & !labels) # the number of incorrect "positive" predictions
    FN <- sum(!p_labs & labels) # the number of "negative" predictions that should be "positive"

    prec <- TP/(TP+FP)
    recall <- TP/(TP+FN)

    return(  2 * prec * recall / (prec + recall) )
}

# set a seed for reproducibility
set.seed(7411)

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

# One hot encoding
encoded_train <- data.frame(model.matrix(~.-1, data=data[train_idx,]))
encoded_test  <- data.frame(model.matrix(~.-1, data=data[-train_idx,]))

# Trees
n_trees <- seq(10, 100, by=10)
N <- length(n_trees)

# features
F_n <- ncol(encoded_train)
feature_idxs <- c(3:F_n) #  start from 3, means start with the first two features

# threshold values
thresh_vals <- seq(0.05, 0.5, by=0.05)

# Storing the result
result <- array(0, dim = c(N, length(feature_idxs), length(thresh_vals)))

# Formula
formula <- as.formula(paste("fraud_bool ~ ."))

# Outer, over trees
for(n in seq_len(N)){
  # Inner, over features
  for(f in seq_len(length(feature_idxs))){

    # and fit
    fit <- bart2(formula          = formula,
                 data             = encoded_train[, c(1:feature_idxs[f])],
                 keepTrees        = TRUE,
                 verbose          = T,
                 n.threads        = 4L,
                 n.trees          = n_trees[n],
                 keepTrainingFits = F,    # drops training f(x), neccessary with loads of data
                 printEvery       = 100L)

    # Calculate the F1 score
    probs <- predict(fit, newdata=encoded_test[, c(2:feature_idxs[f])] , mc.cores=4, type="ev")
    hdis  <- apply(probs, FUN=hdi, MARGIN=2)
    result[n, f, ] <- sapply(thresh_vals, FUN =function(x) calc_F1(hdis, encoded_test[, 1], x))
    }
}

# save the result
saveRDS(result, "bart_F1_scores.rds")
