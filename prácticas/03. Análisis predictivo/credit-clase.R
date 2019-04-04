library(tidyverse)
library(caret)
library(funModeling)

# Usando "German credit card data"
# http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
# Variable de clasificación: Class
data(GermanCredit)
data <- as_tibble(GermanCredit)
glimpse(data)

df_status(data)
ggplot(data) + geom_histogram(aes(x = Class, fill = Class), stat = 'count')

## Crear modelo de predicción usando rpart
# Particiones entrenamiento / test
trainIndex <- createDataPartition(data$Class, p = .75, list = FALSE, times = 1)
train <- data[ trainIndex, ] 
val   <- data[-trainIndex, ]

# Entrenar modelo
rpartCtrl <- trainControl(
  verboseIter = F, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary)
rpartParametersGrid <- expand.grid(
  .cp = c(0.001, 0.01, 0.1, 0.5))
rpartModel <- train(
  Class ~ ., 
  data = train, 
  method = "rpart", 
  metric = "ROC", 
  trControl = rpartCtrl, 
  tuneGrid = rpartParametersGrid)
print(rpartModel)


