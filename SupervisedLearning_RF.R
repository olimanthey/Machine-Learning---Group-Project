library(randomForest)
library(ggplot2)
library(dplyr)

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

################# 

# Trouver manuellement le nbr d'arbres pour notre modèle
set.seed(123)
trainIndex <- createDataPartition(marine_db_filtered$maintenance_status, p = 0.8, list = FALSE)
marine_tr <- marine_db_filtered[trainIndex, ]
marine_te  <- marine_db_filtered[-trainIndex, ]

ntrees <- c(100, 300, 500, 800, 1000)
results_ntree <- data.frame()

for (nt in ntrees) {
  model <- randomForest(maintenance_status ~ ., data = marine_tr, ntree = nt)
  acc <- mean(predict(model, marine_te) == marine_te$maintenance_status)
  results_ntree <- rbind(results_ntree, data.frame(ntree = nt, Accuracy = acc))
}

# 📈 Graph
ggplot(results_ntree, aes(x = ntree, y = Accuracy)) +
  geom_line() + geom_point() +
  labs(title = "Tuning de ntree", x = "Nombre d'arbres", y = "Accuracy") +
  theme_minimal()

best_ntree <- results_ntree$ntree[which.max(results_ntree$Accuracy)]

# Confirmation du résultat en regardant les erreurs
set.seed(42)

model <- randomForest(maintenance_status ~ ., data=marine_tr, ntrees = 1000, importance=TRUE)

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
# ggsave("oob_error_rate_500_trees.pdf")

## Green line = The error rate when classifying "Normal" maintenance status
##
## Blue line = The overall OOB error rate.
##
## Red line = The error rate when classifying "Critical" maintenance status
##
## Purple line = The error rate when classifying "Requires Maintenance" maintenance status
# --> 300 trees optimal
best_ntree <- 300

################# mtrys optimal

mtrys <- 1:(ncol(marine_tr) - 1)
results_mtry <- data.frame()

for (m in mtrys) {
  model <- randomForest(maintenance_status ~ ., data = marine_tr, mtry = m, ntree = best_ntree)
  acc <- mean(predict(model, marine_te) == marine_te$maintenance_status)
  results_mtry <- rbind(results_mtry, data.frame(mtry = m, Accuracy = acc))
}

# 📈 Graph
ggplot(results_mtry, aes(x = mtry, y = Accuracy)) +
  geom_line() + geom_point() +
  labs(title = "Tuning de mtry", x = "mtry", y = "Accuracy") +
  theme_minimal()

best_mtry <- results_mtry$mtry[which.max(results_mtry$Accuracy)]

################# nodesize & maxnodes optimal

nodesize_vals <- c(1, 3, 5, 10, 20)
maxnodes_vals <- c(10, 20, 30, 50, 100)
results_final <- data.frame()

for (ns in nodesize_vals) {
  for (mx in maxnodes_vals) {
    model <- randomForest(
      maintenance_status ~ ., data = marine_tr,
      ntree = best_ntree, mtry = best_mtry,
      nodesize = ns, maxnodes = mx
    )
    acc <- mean(predict(model, marine_te) == marine_te$maintenance_status)
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

# Extract variable importance
varImpPlot(rf_model)
importance(rf_model)
