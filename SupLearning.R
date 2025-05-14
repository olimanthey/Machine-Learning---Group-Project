reticulate::use_condaenv("MLBA")
library(randomForest)
library(tidyverse)
library(dplyr)
library(caret)
library(nnet)
library(ggplot2)
library(xgboost)

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
marine_db <- marine_db %>%
  select(-timestamp, -engine_id)

# Delete columns that where not meaningful after EDA
marine_db <- marine_db %>%
  select(-failure_mode, -engine_type, -manufacturer, -fuel_type, 
         -fuel_consumption_per_hour, -exhaust_temp, -oil_pressure)

################# Baseline Model

# Confirm target is a factor
marine_db$maintenance_status <- as.factor(marine_db$maintenance_status)

# Train/Test Split
set.seed(123)
trainIndex <- createDataPartition(marine_db$maintenance_status, p = 0.8, list = FALSE)
marine_tr <- marine_db[trainIndex, ]
marine_te  <- marine_db[-trainIndex, ]

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
# No overfitting

# The baseline multinomial logistic regression model, applied to the marine engine dataset with 
# `maintenance_status` as the target, showed poor predictive performance. Despite balanced class 
# distribution, the model achieved only \~34% accuracy with a near-zero Kappa (0.0089), 
# indicating predictions barely better than random guessing. Sensitivity was especially low for 
# the "Normal" (19.7%) and "Critical" (34.5%) classes, and overall balanced accuracy hovered 
# around 50% across all classes. The confusion matrix revealed significant class confusion, 
# with the model frequently misclassifying all three categories. These results suggest that the 
# linear assumptions of logistic regression are insufficient to capture the underlying patterns
# in the data, motivating the use of more flexible models like Random Forest.

############################ Random Forest

# Define tuning grid manually
mtry_vals <- c(2, 4, 6)
ntree_vals <- c(100, 300, 500, 1000)

# Initialize results list
results <- data.frame()

# Loop through combinations
set.seed(123)
for (nt in ntree_vals) {
  for (m in mtry_vals) {
    rf_model <- randomForest(
      maintenance_status ~ ., 
      data = marine_tr, 
      mtry = m, 
      ntree = nt,
      trControl = ctrl,
      importance = TRUE
    )
    
    preds <- predict(rf_model, newdata = marine_te)
    acc <- mean(preds == marine_te$maintenance_status)
    
    results <- rbind(results, data.frame(mtry = m, ntree = nt, Accuracy = acc))
  }
}

ggplot(results, aes(x = ntree, y = Accuracy, color = as.factor(mtry), group = mtry)) +
  geom_line() +
  geom_point(size = 2) +
  labs(title = "Accuracy vs. Number of Trees by mtry",
       x = "Number of Trees (ntree)",
       y = "Accuracy",
       color = "mtry") +
  theme_minimal()

#################

# Train Random Forest model
set.seed(123)
rf_model <- randomForest(
  maintenance_status ~ ., 
  data = marine_tr,
  mtry = 6, 
  ntree = 500,
  trControl = ctrl,
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
# High overfitting

# Extract variable importance
varImpPlot(rf_model)
importance(rf_model)

# Resolve Overfitting
# Keep variables with >0 importance score
marine_tr_reduced <- marine_tr %>%
  select(maintenance_status, rpm, engine_temp)

set.seed(123)
rf_pruned <- randomForest(
  maintenance_status ~ ., 
  data = marine_tr_reduced,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

# Evaluate
train_preds <- predict(rf_pruned, newdata = marine_tr_reduced)
test_preds <- predict(rf_pruned, newdata = marine_te)

confusionMatrix(train_preds, marine_tr_reduced$maintenance_status)
confusionMatrix(test_preds, marine_te$maintenance_status)



# The Random Forest model performed poorly on the marine engine dataset, yielding an overall 
# accuracy of 32.4%, with a Kappa of –0.0142, indicating worse-than-random agreement. 
# Sensitivity across classes remained low: 27.1% for "Normal", 34.2% for "Critical", and 35.9% 
# for "Requires Maintenance", with similar patterns in precision and specificity. 
# The balanced accuracy hovered near 50% for all classes, suggesting the model was unable to 
# effectively distinguish between the three maintenance categories. Despite Random Forest’s 
# ability to capture non-linear relationships, its failure here indicates limited predictive 
# signal in the available features, reinforcing the need for either more informative variables 
# or more powerful models like XGBoost to capture subtle patterns in the data.

##################### XGBoost

# Tuning grid
ctrl <- trainControl(
  method = "cv",
  number = 10,
  search = "random"
)

set.seed(123)
xgb_random <- train(
  maintenance_status ~ ., 
  data = marine_tr,
  method = "xgbTree",
  trControl = ctrl,
  tuneLength = 50
)

xgb_grid <- xgb_random$bestTune

# Train XGBoost Model with caret
set.seed(123)
xgb_model <- train(
  maintenance_status ~ ., 
  data = marine_tr,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid
)

# Evaluate and Plot
print(xgb_model)

# Evaluate on Test Set
xgb_preds_tr <- predict(xgb_model, newdata = marine_tr)
xgb_preds_te <- predict(xgb_model, newdata = marine_te)

confusionMatrix(xgb_preds_tr, marine_tr$maintenance_status)
confusionMatrix(xgb_preds_te, marine_te$maintenance_status)
# Low overfittings


