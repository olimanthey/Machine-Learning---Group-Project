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


# Delete unwanted columns
svm_db <- marine_db %>%
  select(-timestamp, -engine_id)

# Delete columns that where not meaningful after EDA
svm_db <- svm_db %>%
  select(-failure_mode, -engine_type, -manufacturer, -fuel_type, 
         -fuel_consumption_per_hour, -exhaust_temp, -oil_pressure)

# Define maintenance status as factor
svm_db$maintenance_status <- as.factor(svm_db$maintenance_status)


# Train/Test Split
set.seed(123)
trainIndex <- createDataPartition(svm_db$maintenance_status, p = 0.8, list = FALSE)
marine_tr <- svm_db[trainIndex, ]
marine_te  <- svm_db[-trainIndex, ]

# Scale features to normalize them
train_x <- scale(marine_tr[, 0:7])
train_y <- marine_tr$maintenance_status
test_x <- scale(marine_te[, 0:7])
test_y <- marine_te$maintenance_status


################### SVM MODEL 1 ##########################
# Train SVM
set.seed(123)
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

svm_model_cv # sigma = 0.01 C = 10
plot(svm_model_cv)
svm_model_cv$bestTune # sigma = 0.01, C = 10

######## Test the model with the new best parameters
set.seed(123)
svm_model_2 <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost = 10,
  sigma = 0.01
)

# Predict
train_pred_2 <- predict(svm_model_2, newdata = train_x)
test_pred_2 <- predict(svm_model_2, newdata = test_x)

# Evaluate
confusionMatrix(train_pred_2, train_y)
confusionMatrix(test_pred_2, test_y)


####### Test a third mode with the second best parameteres
set.seed(123)
svm_model_3 <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost = 1,
  sigma = 0.001
)

# Predict
train_pred_3 <- predict(svm_model_3, newdata = train_x)
test_pred_3 <- predict(svm_model_3, newdata = test_x)

# Evaluate
confusionMatrix(train_pred_3, train_y)
confusionMatrix(test_pred_3, test_y)

####### Test a fourth mode with the second best parameteres
set.seed(123)
svm_model_4 <- svm(
  train_x,
  train_y,
  kernel = "radial",
  cost = 0.1,
  sigma = 0.35
)

# Predict
train_pred_4 <- predict(svm_model_4, newdata = train_x)
test_pred_4 <- predict(svm_model_4, newdata = test_x)

# Evaluate
confusionMatrix(train_pred_4, train_y)
confusionMatrix(test_pred_4, test_y)

########### VARIABLE IMPORTANCE ##############
# Test the variable importance with the model 3 (C = 1, Sigma = 0.001)
library(iml)

# Combine scaled features and labels into one dataframe
train_data <- as.data.frame(train_x)
train_data$maintenance_status <- train_y

# Define prediction wrapper (iml expects probability output)
predictor <- Predictor$new(
  model = svm_model_3,
  data = train_data[, -ncol(train_data)],
  y = train_data$maintenance_status,
  type = "response"
)

# Compute feature importance
imp <- FeatureImp$new(predictor, loss = "ce")  # ce = cross-entropy

# Plot importance
plot(imp) + ggtitle("Variable Importance (SVM)") + theme_minimal() + theme(plot.title = element_text(hjust = 0.5))



