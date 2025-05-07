library(readr)
library(liver)
library(summarytools)
library(dplyr)
library(psych)
library(ggplot2)
library(ggmosaic)
library(GGally)

# Open the marine engine dataset
marine_db <- read_csv("marine_engine_data.csv")

str(marine_db)

##################################
## Preprocessing
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

str(marine_db)

# Check for missing values
sum(is.na(marine_db)) # 0

# Check for duplicate values 
sum(duplicated(marine_db)) # 0

# The data has no missing values and no duplicates. The preprocessing process consited in factorizing categorical values.

##########################################
## Univariate: Exploration of each variables
db_sum <- dfSummary(marine_db, max.distinct.values = 5)
db_sum %>% view()

## More statistics on numerical variables
describe(marine_db, omit=TRUE, skew=FALSE, IQR = TRUE)

## More summary using the R base function
summary(marine_db)



