library(readr)
library(dplyr)
library(e1071)
library(caret)
library(pROC)

# Open the marine engine dataset
marine_db <- read_csv("marine_engine_data.csv")

# PREPARE THE DATASET 
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


########## SUPPORT VECTOR MACHINE ############

# Prepare data
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

# Train SVM
svm_model_1 <- svm(train_x, y = train_y, kernel = "radial", cost = 1, gamma = 0.1)

# Predict
train_pred_1 <- predict(svm_model_1, newdata = train_x)
test_pred_1 <- predict(svm_model_1, newdata = test_x)

# Evaluate
confusionMatrix(train_pred_1, train_y)
confusionMatrix(test_pred_1, test_y)

########## HYPERPARAMETERS TUNING #############

# Use of cross-validation CV to select the best combo between cost and gamma
ctrl <- trainControl(method = "cv", number = 10)

# Grid of hyperparameters
svm_grid <- expand.grid(
  C = c(0.1, 1, 10, 100, 200, 300, 400, 500, 1000),
  sigma = c(0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 1)
)

# Train the SVM model (run only if necessaty, takes 2-3h)
svm_model_cv <- train(
  x = train_x,
  y = train_y,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = svm_grid,
  preProcess = NULL
)

svm_model_cv # sigma = 0.35 C = 0.1
plot(svm_model_cv)
svm_model_cv$bestTune # sigma = 0.35, C = 0.1

######## Test the model with the new best parameters
set.seed(242)
svm_model_2 <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost = 0.1,
  sigma = 0.35
)

# Predict
train_pred_2 <- predict(svm_model_2, newdata = train_x)
test_pred_2 <- predict(svm_model_2, newdata = test_x)

# Evaluate
confusionMatrix(train_pred_2, train_y)
confusionMatrix(test_pred_2, test_y)


####### Test a third mode with the second best parameteres
set.seed(242)
svm_model_3 <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost = 1,
  sigma = 0.01
)

# Predict
train_pred_3 <- predict(svm_model_3, newdata = train_x)
test_pred_3 <- predict(svm_model_3, newdata = test_x)

# Evaluate
confusionMatrix(train_pred_3, train_y)
confusionMatrix(test_pred_3, test_y)

########### ROC Curve ##################

# Train the best model with probabilities
set.seed(242)

svm_best_model <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost= 0.1,
  gamma = 0.35,
  probability = TRUE
)

# Predict probabilities on the test set
svm_prob <- predict(svm_best_model, test_x, probability = TRUE)
svm_prob <- attr(svm_prob, "probabilities")

# One - vs- Rest ROC curves
roc_normal <- roc(test_y == "Normal", svm_prob[, "Normal"])
plot(roc_normal, col = "blue", main = "One vs Rests ROC Curves")

# Add other class
roc_critical <- roc(test_y == "Critical", svm_prob[, "Critical"])
plot(roc_critical, col = "red", add = TRUE)

roc_requires <- roc(test_y == "Requires Maintenance", svm_prob[, "Requires Maintenance"])
plot(roc_requires, col = "green", add = TRUE)

legend("bottomright", legend = c("Normal", "Critical", "Requires Maintenance"),
       col = c("blue", "red", "green"), lty = 1)
