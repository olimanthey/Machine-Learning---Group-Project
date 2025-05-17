library(randomForest)
library(ggplot2)
library(dplyr)
library(caret)
library(pdp)
library(tidyverse)
library(nnet)
library(e1071)

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
  mtry = best_mtry, 
  ntree = best_ntree,
  nodesize = best_nodesize,
  maxnodes = best_maxnodes,
  importance = TRUE
)

# Output model summary
print(rf_model)

# Predict on training & test set
rf_preds_tr <- predict(rf_model, newdata = marine_tr)
rf_preds_te <- predict(rf_model, newdata = marine_te)

# Confusion matrix and stats
confusionMatrix(rf_preds_tr, marine_tr$maintenance_status)

################## SVM

features <- marine_db %>%
  select(engine_temp, coolant_temp, engine_load, rpm,
         vibration_level, fuel_consumption, running_period)

labels <- marine_db$maintenance_status

# Encode labels as factor
labels <- as.factor(labels)

# Scale features to normalize them
scaled_features <- scale(features)

# Train/test split
set.seed(242)
train_index <- createDataPartition(labels, p = 0.8, list = FALSE)
train_x <- scaled_features[train_index, ]
test_x  <- scaled_features[-train_index, ]
train_y <- labels[train_index]
test_y  <- labels[-train_index]

set.seed(242)

svm_best_model <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost= 0.1,
  gamma = 0.35,
  probability = TRUE
)

# Predict
train_pred <- predict(svm_best_model, newdata = train_x)
test_pred <- predict(svm_best_model, newdata = test_x)

# Evaluate
confusionMatrix(train_pred, train_y)
confusionMatrix(test_pred, test_y)
