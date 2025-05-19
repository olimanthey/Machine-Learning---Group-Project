library(randomForest)
library(ggplot2)
library(dplyr)
library(caret)
library(pdp)
library(tidyverse)
library(nnet)
library(e1071)
library(yardstick)

marine_db <- read.csv("marine_engine_data.csv")

### Preprocessing
# Convert character columns to factors
for (i in 1:ncol(marine_db)) {
  if (is.character(marine_db[[i]])) {
    marine_db[[i]] <- as.factor(marine_db[[i]])
  }
}

# Reorder factor levels by decreasing frequency
for (i in 1:ncol(marine_db)) {
  if (is.factor(marine_db[[i]])) {
    marine_db[[i]] <- reorder(marine_db[[i]], marine_db[[i]], FUN = length)
  }
}

# Delete unwanted columns
marine_db_filtered <- marine_db %>%
  select(-timestamp, -engine_id, -failure_mode, -engine_type, -manufacturer, -fuel_type, 
         -fuel_consumption_per_hour, -exhaust_temp, -oil_pressure)

# Confirm target is a factor
marine_db_filtered$maintenance_status <- as.factor(marine_db_filtered$maintenance_status)

# Data Splitting
set.seed(123)
trainIndex <- createDataPartition(marine_db_filtered$maintenance_status, p = 0.8, list = FALSE)
marine_tr <- marine_db_filtered[trainIndex, ]
marine_te  <- marine_db_filtered[-trainIndex, ]

################## MLR

# 10-Fold Cross-Validation Setup
ctrl <- trainControl(method = "cv", number = 10)

# Train Multinomial Logistic Regression
set.seed(123)
baseline_model <- train(
  maintenance_status ~ ., 
  data = marine_tr,
  method = "multinom",
  trControl = ctrl,
  trace = FALSE
)

# Print model CV performance
print(baseline_model)

# Predict on test set
baseline_preds_tr <- predict(baseline_model, newdata = marine_tr)
baseline_preds_te <- predict(baseline_model, newdata = marine_te)

# Evaluate
confusionMatrix(baseline_preds_tr, marine_tr$maintenance_status)
confusionMatrix(baseline_preds_te, marine_te$maintenance_status)

################## RF

# Train OPtimal Random Forest model
set.seed(123)
rf_model <- randomForest(
  maintenance_status ~ ., 
  data = marine_tr,
  mtry = 2, 
  ntree = 300,
  nodesize = 10,
  maxnodes = 10,
  importance = TRUE
)

# Output model summary
print(rf_model)

# Predict on training & test set
rf_preds_tr <- predict(rf_model, newdata = marine_tr)
rf_preds_te <- predict(rf_model, newdata = marine_te)

# Confusion matrix and stats
confusionMatrix(rf_preds_tr, marine_tr$maintenance_status)
confusionMatrix(rf_preds_te, marine_te$maintenance_status)

################## SVM

# Scale features to normalize them
train_x <- scale(marine_tr[, 0:7])
train_y <- marine_tr$maintenance_status
test_x <- scale(marine_te[, 0:7])
test_y <- marine_te$maintenance_status

# Compute the SVM model
set.seed(123)
svm_model_3 <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost = 1,
  sigma = 0.001
)

# Predict
svm_pred_tr <- predict(svm_model_3, newdata = train_x)
svm_pred_te <- predict(svm_model_3, newdata = test_x)

# Evaluate
confusionMatrix(svm_pred_tr, train_y)
confusionMatrix(svm_pred_te, test_y)

################## COMPUTE F1 METRIC ###################
# MLR
results_mlr <- data.frame(
  truth = marine_te$maintenance_status,
  prediction = baseline_preds_te
)

f1_score_mlr <- f_meas(results_mlr, truth = truth, estimate = prediction, estimator = "macro")

# RF
results_rf <- data.frame(
  truth = marine_te$maintenance_status,
  prediction = rf_preds_te
)

f1_score_rf <- f_meas(results_rf, truth = truth, estimate = prediction, estimator = "macro")

# SVM
results_svm <- data.frame(
  truth = marine_te$maintenance_status,
  prediction = svm_pred_te
)

f1_score_svm <- f_meas(results_svm, truth = truth, estimate = prediction, estimator = "macro")

# F1-Score per model
f1_score_mlr # 0.33
f1_score_rf # 0.285
f1_score_svm # 0.33
  