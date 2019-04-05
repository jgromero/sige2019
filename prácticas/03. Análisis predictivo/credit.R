## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

library(caret)
library(tidyverse)
library(funModeling)
library(pROC)
library(partykit)
library(rattle)
library(randomForest)
library(xgboost)

## -------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------
## Funciones ##

#' Cálculo de valores ROC
#' @param data Datos originales
#' @param predictionProb Predicciones
#' @param target_var Variable objetivo de predicción
#' @param positive_class Clase positiva de la predicción
#' 
#' @return Lista con valores de resultado \code{$auc}, \code{$roc}
#' 
#' @examples 
#' rfModel <- train(Class ~ ., data = train, method = "rf", metric = "ROC", trControl = rfCtrl, tuneGrid = rfParametersGrid)
#' roc_res <- my_roc(data = validation, predict(rfModel, validation, type = "prob"), "Class", "Good")
my_roc <- function(data, predictionProb, target_var, positive_class) {
  auc <- roc(data[[target_var]], predictionProb[[positive_class]], levels = unique(data[[target_var]]))
  roc <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('AUC:', round(auc$auc[[1]], 2)))
  return(list("auc" = auc, "roc" = roc))
}
## -------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------

# Usando "German credit card data"
# http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
# Variable de clasificación: Class
data(GermanCredit)
data <- as_tibble(GermanCredit)
glimpse(data)
df_status(data)
ggplot(data) + geom_histogram(aes(x = Class, fill = Class), stat = 'count')

set.seed(0)

## -------------------------------------------------------------------------------------

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

# Obtener valores de accuracy, precision, recall, f-score (usando confusionMatrix)
cm_val <- confusionMatrix(prediction, val[["Class"]], positive = "Good")
cm_val$table[c(2,1), c(2,1)] # invertir filas y columnas para ver primero la clase "Good"

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

## -------------------------------------------------------------------------------------

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

roc_1 <- my_roc(val, predictionProb, "Class", "Good")

## -------------------------------------------------------------------------------------

## Crear modelo de predicción usando rf
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

# Ajuste manual del parámetro número de árboles (ntrees), usando hiperparámetro anterior(.mtry)
rfCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, summaryFunction = twoClassSummary)
rfParametersGrid <- expand.grid(.mtry = bestmtry[,1])

modellist <- list()
for (ntrees in c(100, 150, 200, 250)) {
  rfModel <- train(Class ~ ., data = train, method = "rf", metric= "ROC", tuneGrid = rfParametersGrid, trControl = rfCtrl, ntree = ntrees)
  key <- toString(ntrees)
  modellist[[key]] <- rfModel
}

results <- resamples(modellist)
summary(results)
dotplot(results)
bwplot(diff(results), metric = "ROC")

my_roc(val, predict(modellist[[3]], val, type = "prob"), "Class", "Good")

## -------------------------------------------------------------------------------------

## Crear modelo de predicción usando SVM
svmCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, summaryFunction = twoClassSummary)
svmModel <- train(Class ~ ., data = train, method = "svmRadial", metric = "ROC", trControl = svmCtrl, tuneLength = 10)
print(svmModel)
plot(svmModel)
my_roc(val, predict(svmModel, val, type = "prob"), "Class", "Good")

## -------------------------------------------------------------------------------------

## Crear modelo de predicción usando RNA
nnCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, summaryFunction = twoClassSummary)
nnParametersGrid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
nnModel <- train(Class ~ ., data = train, method = "nnet", metric = "ROC", tuneGrid = nnParametersGrid, trControl = nnCtrl, trace = FALSE, maxit = 1000) 
print(nnModel)
plot(nnModel)
my_roc(val, predict(nnModel, val, type = "prob"), "Class", "Good")

## -------------------------------------------------------------------------------------

## Ensembles
## Más: https://topepo.github.io/caret/model-training-and-tuning.html
library(caretEnsemble)

# Conjunto de modelos
listCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", number = 10, repeats = 1, summaryFunction = twoClassSummary)
model_list <- caretList(Class ~ ., data = train, trControl = listCtrl, methodList=c("rpart", "rf", "svmRadial", "nnet"))
predictions_list <- predict(model_list, newdata = val)
head(predictions_list)
print(model_list)
plot(model_list$rpart)
my_roc(val, predict(model_list$rpart, val, type = "prob"), "Class", "Good")

# Conjunto de modelos con grids de parámetros independientes
model_list <- caretList(Class~., data = train,
  trControl = listCtrl,
  metric= "ROC",
  tuneList=list(
    rpart = caretModelSpec(method="rpart",     tuneGrid = expand.grid(.cp = c(0.001, 0.01, 0.1, 0.5))),
    rf    = caretModelSpec(method="rf",        tuneGrid = expand.grid(.mtry = c(1:5))),
    svm   = caretModelSpec(method="svmRadial", tuneLength = 10), 
    nnet  = caretModelSpec(method="nnet",      tuneGrid = expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7)), trace = FALSE)
  )
)
predictions_list <- predict(model_list, newdata = val)
head(predictions_list)

rpart_roc_res <- my_roc(val, predict(model_list$rpart, val, type = "prob"), "Class", "Good")
rf_roc_res <- my_roc(val, predict(model_list$rf, val, type = "prob"), "Class", "Good")
svm_roc_res <- my_roc(val, predict(model_list$svm, val, type = "prob"), "Class", "Good")
nnet_roc_res <- my_roc(val, predict(model_list$nnet, val, type = "prob"), "Class", "Good")

plot.roc(rpart_roc_res$auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('AUC:', round(auc$auc[[1]], 2)))
plot.roc(rf_roc_res$auc, add = TRUE, col = "red", ylim=c(0,1), type = "S" , print.thres = T, main=paste('AUC:', round(auc$auc[[1]], 2)))
lines(svm_roc_res$auc, col = "blue")
lines(nnet_roc_res$auc, col = "green")

# Generar ensemble con pesos (por defecto)
xyplot(resamples(c(model_list[1], model_list[2])))
xyplot(resamples(c(model_list[2], model_list[4])))

model_list_selected <- caretList(Class~., data = train,
  trControl = listCtrl,
  metric= "ROC",
  tuneList=list(
    rf    = caretModelSpec(method="rf",   tuneGrid = expand.grid(.mtry = 5)),
    nnet  = caretModelSpec(method="nnet", tuneGrid = expand.grid(.decay = 0.5, .size = 5), trace = FALSE)
  )
)
greedy_ensemble <- caretEnsemble(
  model_list_selected , 
  metric="ROC",
  trControl=trainControl(
    number=2,
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))       
summary(greedy_ensemble)
ensemble_pred <- data.frame(Good = predict(greedy_ensemble, val, type = "prob"), Bad = 1-predict(greedy_ensemble, val, type = "prob"))
my_roc(val, ensemble_pred, "Class", "Good")

# Generar ensemble con método de ensemblado seleccionado
custom_ensemble <- caretStack(
  model_list_selected, 
  method="gbm",
  metric="ROC",
  trControl=trainControl(
    number=2,
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))  
summary(custom_ensemble)
ensemble_pred_2 <- data.frame(Good = predict(custom_ensemble, val, type = "prob"), Bad = 1-predict(custom_ensemble, val, type = "prob"))
my_roc(val, ensemble_pred_2, "Class", "Good")

## -------------------------------------------------------------------------------------

## Boosting
xgbCtrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 1,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
xgbGrid <- expand.grid( # https://xgboost.readthedocs.io/en/latest/parameter.html
  nrounds = 200,
  max_depth = c(6, 8, 10),
  eta = c(0.001, 0.003, 0.01),
  gamma = 1,
  colsample_bytree = 0.5,
  min_child_weight = 6,
  subsample = 0.5
)
xgbModel <- train(
  Class ~ ., 
  data = train, 
  method = "xgbTree", 
  metric = "ROC", 
  trControl = xgbCtrl,
  tuneGrid = xgbGrid
)
print(xgbModel)
plot(xgbModel)
my_roc(val, predict(xgbModel, val, type = "prob"), "Class", "Good")

imp <- xgb.importance(colnames(train), xgbModel$finalModel)
xgb.plot.importance(imp)


