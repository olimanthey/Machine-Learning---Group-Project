library(randomForest)
library(ggplot2)
library(dplyr)
library(caret)
library(pdp)

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

################# Baseline model

set.seed(678)

baseline_model <- randomForest(maintenance_status ~ ., data=marine_tr, importance=TRUE)

baseline_model

# Predict on training & test set
baseline_rf_preds_tr <- predict(baseline_model, newdata = marine_tr)
baseline_rf_preds_te <- predict(baseline_model, newdata = marine_te)

# Confusion matrix and stats
confusionMatrix(rf_preds_tr, marine_tr$maintenance_status)
confusionMatrix(rf_preds_te, marine_te$maintenance_status)

################# Hyperparameters tuning

# Trouver manuellement le nbr d'arbres pour notre modèle
set.seed(123)
# Set up cross-validation
ctrl <- trainControl(method = "cv", number = 10)  # 5-fold CV

# Define grid of mtry and ntree
mtry_vals <- 1:(ncol(marine_tr) - 1)
ntree_vals <- c(100, 300, 500, 800, 1000)

# Use expand.grid to create all combinations
grid <- expand.grid(mtry = mtry_vals, ntree = ntree_vals)

# Custom training loop for grid search (caret doesn't natively tune ntree via tuneGrid)
results_grid <- data.frame()

for (i in 1:nrow(grid)) {
  m <- grid$mtry[i]
  nt <- grid$ntree[i]
  
  model <- train(
    maintenance_status ~ .,
    data = marine_tr,
    method = "rf",
    trControl = ctrl,
    tuneGrid = data.frame(mtry = m),
    ntree = nt
  )
  
  acc <- max(model$results$Accuracy)
  results_grid <- rbind(results_grid, data.frame(mtry = m, ntree = nt, Accuracy = acc))
}

ggplot(results_grid, aes(x = ntree, y = Accuracy, color = factor(mtry))) +
  geom_line() + geom_point() +
  labs(title = "Joint Tuning of mtry and ntree", x = "ntree", y = "Accuracy", color = "mtry") +
  theme_minimal()
ggsave("RF_pictures/mtry_ntree_lineplot.png")

best_ntree <- results_grid$ntree[which.max(results_grid$Accuracy)]
best_mtry <- results_grid$mtry[which.max(results_grid$Accuracy)]

# Confirmation du résultat en regardant les erreurs
set.seed(42)

model <- randomForest(maintenance_status ~ ., data=marine_tr, ntree = 1000, importance=TRUE)

model

## Now check to see if the random forest is actually big enough...
## Up to a point, the more trees in the forest, the better. You can tell when
## you've made enough when the OOB no longer improves.
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=4),
  Type=rep(c("OOB", "Normal", "Critical", "Requires Maintenance"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"Normal"], 
          model$err.rate[,"Critical"],
          model$err.rate[,"Requires Maintenance"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))
ggsave("RF_pictures/oob_error_rate_1000_trees.pdf")

## Green line = The error rate when classifying "Normal" maintenance status
##
## Blue line = The overall OOB error rate.
##
## Red line = The error rate when classifying "Critical" maintenance status
##
## Purple line = The error rate when classifying "Requires Maintenance" maintenance status
# --> 300 trees optimal

## nodesize & maxnodes optimal

results_final <- data.frame()

nodesize_vals <- c(1, 3, 5, 10, 15, 20)
maxnodes_vals <- c(5, 10, 20, 30, 50)

for (ns in nodesize_vals) {
  for (mx in maxnodes_vals) {
    model <- train(
      maintenance_status ~ .,
      data = marine_tr,
      method = "rf",
      trControl = ctrl,
      tuneGrid = data.frame(mtry = best_mtry),  # Fix mtry
      ntree = best_ntree,                       # Fix ntree
      nodesize = ns,
      maxnodes = mx
    )
    
    acc <- max(model$results$Accuracy)
    results_final <- rbind(results_final, data.frame(nodesize = ns, maxnodes = mx, Accuracy = acc))
  }
}

# 📈 Heatmap
ggplot(results_final, aes(x = factor(nodesize), y = factor(maxnodes), fill = Accuracy)) +
  geom_tile() +
  geom_text(aes(label = round(Accuracy, 3)), color = "black") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Tuning de nodesize et maxnodes",
    x = "nodesize",
    y = "maxnodes"
  ) +
  theme_minimal()
ggsave("RF_pictures/nodesize_maxnodes_tuning.png")

best_nodesize <- results_final$nodesize[which.max(results_final$Accuracy)]
best_maxnodes <- results_final$maxnodes[which.max(results_final$Accuracy)]

################ Modèle optimal

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
confusionMatrix(rf_preds_te, marine_te$maintenance_status)
# Low overfitting

################# Variable Importance

# Extract variable importance
varImpPlot(rf_model)
importance(rf_model)

# Set layout for 2 plots side by side
par(mfrow = c(1, 2))

# PDP for engine_load
pdp_engine <- partial(
  object = rf_model,
  pred.var = "engine_load",
  train = marine_tr,
  which.class = "Requires Maintenance",
  prob = TRUE
)
plot(pdp_engine,
     type = "l",
     main = "PDP: engine_load",
     xlab = "Engine Load",
     ylab = "Predicted Probability")

# PDP for coolant_temp
pdp_coolant <- partial(
  object = rf_model,
  pred.var = "coolant_temp",
  train = marine_tr,
  which.class = "Requires Maintenance",
  prob = TRUE
)

plot(pdp_coolant,
     type = "l",
     main = "PDP: coolant_temp",
     xlab = "Coolant Temperature",
     ylab = "Predicted Probability")
