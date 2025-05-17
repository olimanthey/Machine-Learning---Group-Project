reticulate::use_condaenv("MLBA")
library(tidyverse)
library(dplyr)
library(caret)
library(nnet)
library(ggplot2)
library(reshape2)

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
log_reg_db <- marine_db %>%
  select(-timestamp, -engine_id)

# Delete columns that where not meaningful after EDA
log_reg_db <- log_reg_db %>%
  select(-failure_mode, -engine_type, -manufacturer, -fuel_type, 
         -fuel_consumption_per_hour, -exhaust_temp, -oil_pressure)

################# Baseline Model

# Confirm target is a factor
log_reg_db$maintenance_status <- as.factor(log_reg_db$maintenance_status)

# Train/Test Split
set.seed(123)
trainIndex <- createDataPartition(log_reg_db$maintenance_status, p = 0.8, list = FALSE)
marine_tr <- log_reg_db[trainIndex, ]
marine_te  <- log_reg_db[-trainIndex, ]

# 10-Fold Cross-Validation Setup
ctrl <- trainControl(method = "cv", number = 10)

################ Train Multinomial Logistic Regression
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

################ LASSO & RIDGE regression

set.seed(123)
baseline_model_2 <- train(
  maintenance_status ~.,
  data = marine_tr,
  method= "glmnet", # add L1 (LASSO) and L2 (RIDGE)
  trControl = ctrl,
  preProcess = c("center", "scale"), # Features scaling
  trace = FALSE
)

# Predict on test set
baseline_preds_tr_2 <- predict(baseline_model_2, newdata = marine_tr)
baseline_preds_te_2 <- predict(baseline_model_2, newdata = marine_te)

# Evaluate
confusionMatrix(baseline_preds_tr_2, marine_tr$maintenance_status)
confusionMatrix(baseline_preds_te_2, marine_te$maintenance_status)



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

################### VARIABLES IMPORTANCE PLOT ############
importance <- varImp(baseline_model_2)
plot(importance, top = 7)

# Extract the model
coef_matrix_log <- coef(baseline_model_2$finalModel)
coef_dense <- as.matrix(coef_matrix_log)

# Convert to long format
coef_df <- melt(coef_dense)
colnames(coef_df) <- c("Class", "Feature", "Coefficient")

# Plot the graph
ggplot(coef_df, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient, fill = Class)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Feature Coefficients by Class (MLR)",
       x = "Feature", y = "Coefficient") +
  theme_minimal()
