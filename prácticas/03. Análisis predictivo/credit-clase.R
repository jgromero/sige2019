library(tidyverse)
library(caret)
library(funModeling)
library(pROC)
library(partykit)
library(rattle)

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

# Validacion
prediction     <- predict(rpartModel, val, type = "raw")
predictionProb <- predict(rpartModel, val, type = "prob")

auc <- roc(val$Class, predictionProb[["Good"]], levels = unique(val[["Class"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('Validation AUC:', round(auc$auc[[1]], 2)))

# Obtener valores de accuracy, precision, recall, f-score (manualmente)
results <- cbind(val, prediction)
results <- results %>%
  mutate(contingency = as.factor(
    case_when(
      Class == 'Good' & prediction == 'Good' ~ 'TP',
      Class == 'Bad'  & prediction == 'Good' ~ 'FP',
      Class == 'Bad'  & prediction == 'Bad'  ~ 'TN',
      Class == 'Good' & prediction == 'Bad'  ~ 'FN'))) 
TP <- length(which(results$contingency == 'TP'))
TN <- length(which(results$contingency == 'TN'))
FP <- length(which(results$contingency == 'FP'))
FN <- length(which(results$contingency == 'FN'))
n  <- length(results$contingency)

table(results$contingency) # comprobar recuento de TP, TN, FP, FN

accuracy <- (TP + TN) / n
error <- (FP + FN) / n

precision   <- TP / (TP + FP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
f_measure   <- (2 * TP) / (2 * TP + FP + FN)

# Obtener valores de accuracy, precision, recall, f-score (usando confusionMatrix)
cm_val <- confusionMatrix(prediction, val[["Class"]], positive = "Good")
cm_val$table[c(2,1), c(2,1)] # invertir filas y columnas para ver primero la clase "Good"





