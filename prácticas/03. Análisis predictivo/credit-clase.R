library(caret)
library(tidyverse)
library(funModeling)
library(pROC)
library(partykit)
library(rattle)
library(randomForest)
library(xgboost)

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

# Otro modelo utilizando rpart con cross-validation
rpartCtrl_2 <- trainControl(
  verboseIter = F, 
  classProbs = TRUE, 
  method = "repeatedcv",
  number = 10,
  repeats = 1,
  summaryFunction = twoClassSummary)
rpartModel_2 <- train(Class ~ ., data = train, method = "rpart", metric = "ROC", trControl = rpartCtrl_2, tuneGrid = rpartParametersGrid)
print(rpartModel_2)
varImp(rpartModel_2)
dotPlot(varImp(rpartModel_2))

plot(rpartModel_2)
plot(rpartModel_2$finalModel)
text(rpartModel_2$finalModel)

partyModel_2 <- as.party(rpartModel_2$finalModel)
plot(partyModel_2, type = 'simple')

fancyRpartPlot(rpartModel_2$finalModel)

predictionProb <- predict(rpartModel_2, val, type = "prob")

#' roc_res <- my_roc(data = validation, predict(rfModel, validation, type = "prob"), "Class", "Good")
my_roc <- function(data, predictionProb, target_var, positive_class) {
  auc <- roc(data[[target_var]], predictionProb[[positive_class]], levels = unique(data[[target_var]]))
  roc <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('AUC:', round(auc$auc[[1]], 2)))
  return(list("auc" = auc, "roc" = roc))
}

my_roc(val, predictionProb, "Class", "Good")

# Modelo básico, ajuste de manual de hiperparámetros (.mtry)
rfCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, summaryFunction = twoClassSummary)
rfParametersGrid <- expand.grid(.mtry = c(sqrt(ncol(train))))
rfModel <- train(Class ~ ., data = train, method = "rf", metric = "ROC", trControl = rfCtrl, tuneGrid = rfParametersGrid)
print(rfModel)
varImp(rfModel$finalModel)
varImpPlot(rfModel$finalModel)
my_roc(val, predict(rfModel, val, type = "prob"), "Class", "Good")

# Modelo básico, ajuste manual de hiperparámetros (.mtry) utilizando un intervalo
rfCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, summaryFunction = twoClassSummary)
rfParametersGrid <- expand.grid(.mtry = c(1:5))
rfModel <- train(Class ~ ., data = train, method = "rf", metric = "ROC", trControl = rfCtrl, tuneGrid = rfParametersGrid)
print(rfModel)
plot(rfModel)
plot(rfModel$finalModel)
my_roc(val, predict(rfModel, val, type = "prob"), "Class", "Good")

# Modelo básico, ajuste con búsqueda aleatoria de hiperparámetros (.mtry)
rfCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, search = "random", summaryFunction = twoClassSummary)
rfModel <- train(Class ~ ., data = train, method = "rf", metric = "ROC", trControl = rfCtrl, tuneLength = 15)
print(rfModel)
plot(rfModel)
my_roc(val, predict(rfModel, val, type = "prob"), "Class", "Good")

# Ajuste con tuneRF (.mtry) (Class es la columna 10)
bestmtry <- tuneRF(val[,-10], val[[10]], stepFactor=0.75, improve=1e-5, ntree=500)
print(bestmtry)









