library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(factoextra)
library(tidyr)
library(patchwork)

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

# Convert to Date class
marine_db$timestamp <- as.Date(marine_db$timestamp)


# K MEAN CLUSTERING ANALYSIS
# Preprocessing
clustering_vars <- marine_db %>%
  select(fuel_consumption, rpm, running_period, engine_load, coolant_temp, engine_temp, vibration_level) %>%
  scale()

# Determining optimal number of cluster
fviz_nbclust(clustering_vars, kmeans, method = "wss")  # Elbow method
fviz_nbclust(clustering_vars, kmeans, method = "silhouette")

# K-mean clustering
set.seed(123)
kmeans_result <- kmeans(clustering_vars, centers = 3, nstart = 25)
marine_db$cluster <- as.factor(kmeans_result$cluster)

# Compare cluster with maintenance label
table(marine_db$maintenance_status, marine_db$cluster)

# Visualization
fviz_cluster(kmeans_result, data = clustering_vars, geom = "point",
             ellipse.type = "convex", main = "K-means Clustering of Engine Data")

# Perform PCA
pca_result <- prcomp(clustering_vars, scale. = TRUE)

# Visualize PCA with clusters
fviz_pca_biplot(pca_result,
                geom.ind = "point",
                col.ind = marine_db$cluster, # color by cluster
                col.var = "black",
                palette = c("#F8766D", "#00BA38", "#619CFF", "#F00246", "#619"),
                addEllipses = TRUE,
                legend.title = "Cluster") +
  ggtitle("PCA Projection of Engine Data with K-means Clusters")

summary(pca_result)

# Facet boxplot with interest variables
# Purpose: define which cluster is assign to which feature
# List of variables of interest
vars_of_interest <- c("fuel_consumption", "rpm", "running_period", "engine_load", "coolant_temp", "engine_temp", "vibration_level")

# Prepare data in long format for faceted boxplots
interest_var_clust_plot <- marine_db %>%
  select(cluster, all_of(vars_of_interest)) %>%
  pivot_longer(cols = -cluster, names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = cluster, y = value, fill = cluster)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Distribution of Key Features by Cluster", x = "Cluster", y = "Value") +
  theme_minimal() +
  theme(legend.position = "none")

# Plot the graph
interest_var_clust_plot

# Cross-tabulate each categorical variables with clusters
table(marine_db$engine_type, marine_db$cluster)
table(marine_db$fuel_type, marine_db$cluster)
table(marine_db$manufacturer, marine_db$cluster)
table(marine_db$failure_mode, marine_db$cluster)

# Chi-Squared test of independence
## To assess if the relationship is statistically significant
chisq.test(table(marine_db$engine_type, marine_db$cluster))
chisq.test(table(marine_db$fuel_type, marine_db$cluster))
chisq.test(table(marine_db$manufacturer, marine_db$cluster))
chisq.test(table(marine_db$failure_mode, marine_db$cluster))
chisq.test(table(marine_db$maintenance_status, marine_db$cluster))

# Plot visualization
# Define a reusable theme
box_theme_clust <- theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "right"
  )

# Engine type
p14 <- ggplot(marine_db, aes(x = cluster, fill = engine_type)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Engine Type Distribution by Cluster") +
  scale_fill_manual(values = c("#F8766D", "#00BA38", "#619CFF", "#F00246")) +
  box_theme_clust

# Fuel type
p24 <- ggplot(marine_db, aes(x = cluster, fill = fuel_type)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Fuel Type Distribution by Cluster") +
  scale_fill_manual(values = c("#F8766D", "#00BA38")) +
  box_theme_clust

# Manufacturer
p34 <- ggplot(marine_db, aes(x = cluster, fill = manufacturer)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Manufacturer Distribution by Cluster") +
  scale_fill_manual(values = c("#F8766D", "#00BA38", "#619CFF", "#F00246", "#000CFF", "#619")) +
  box_theme_clust

# Failure mode
p44 <- ggplot(marine_db, aes(x = cluster, fill = failure_mode)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion", title = "Failure Mode Distribution by Cluster") +
  scale_fill_manual(values = c("#F8766D", "#00BA38", "#619CFF", "#F00246")) +
  box_theme_clust

# Merge all the plots
(p14 | p24) / (p34 | p44)
