reticulate::use_condaenv("MLBA")
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
xgb_db <- marine_db %>%
  select(-timestamp, -engine_id)

# Delete columns that where not meaningful after EDA
xgb_db <- xgb_db %>%
  select(-failure_mode, -engine_type, -manufacturer, -fuel_type, 
         -fuel_consumption_per_hour, -exhaust_temp, -oil_pressure)

################# Baseline Model

# Confirm target is a factor
xgb_db$maintenance_status <- as.factor(xgb_db$maintenance_status)

# Train/Test Split
set.seed(123)
trainIndex <- createDataPartition(xgb_db$maintenance_status, p = 0.8, list = FALSE)
marine_tr <- xgb_db[trainIndex, ]
marine_te  <- xgb_db[-trainIndex, ]

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

# Best tune
xgb_grid <- xgb_random$bestTune

#### Store best tune values
# nround = 649
# max_depth = 1
# eta = 0.06754678
# gamma = 4.025733
# colsample_bytree = 0.5124282
# min_child_weight = 4
# subsample = 0.8268954

# Keep the value for further use
xgb_grid_value <- expand.grid(
  nrounds = 649,
  max_depth = 1,
  eta = 0.06754678,
  gamma = 4.025733,
  colsample_bytree = 0.5124282,
  min_child_weight = 4,
  subsample = 0.8268954
)


# Train XGBoost Model with caret
set.seed(123)
xgb_model <- train(
  maintenance_status ~ ., 
  data = marine_tr,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid_value
)

# Evaluate and Plot
print(xgb_model)

# Evaluate on Test Set
xgb_preds_tr <- predict(xgb_model, newdata = marine_tr)
xgb_preds_te <- predict(xgb_model, newdata = marine_te)

confusionMatrix(xgb_preds_tr, marine_tr$maintenance_status)
confusionMatrix(xgb_preds_te, marine_te$maintenance_status)

# Low overfittings