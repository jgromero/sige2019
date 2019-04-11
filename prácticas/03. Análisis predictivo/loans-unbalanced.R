## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

library(caret)
library(tidyverse)
library(funModeling)
library(pROC)
library(DMwR)

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

#' Creación de modelo RandomForest
#' @param train Datos de entrenamiento
#' @param rfCtrl Estructura trainControl
#' @param rfParametersGrid Estructura tuneGrid
#' 
#' @return Modelo entrenado
trainRF <- function(train_data, rfCtrl = NULL, rfParametersGrid = NULL) {
  if(is.null(rfCtrl)) {
    rfCtrl <- trainControl(
      verboseIter = F, 
      classProbs = TRUE, 
      method = "repeatedcv", 
      number = 10, 
      repeats = 1, 
      summaryFunction = twoClassSummary)    
  }
  if(is.null(rfParametersGrid)) {
    rfParametersGrid <- expand.grid(
      .mtry = c(sqrt(ncol(train)))) 
  }
  
  rfModel <- train(
    loan_status ~ ., 
    data = train_data, 
    method = "rf", 
    metric = "ROC", 
    trControl = rfCtrl, 
    tuneGrid = rfParametersGrid)

  return(rfModel)
}
## -------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------

# Usando "LoanStats_2017Q4-PreProc.csv"
# Variable de clasificación: loan_status
data_raw <- read_csv('LoanStats_2017Q4-PreProc.csv')
data <- data_raw %>%
  na.exclude() %>%
  mutate(loan_status = as.factor(loan_status))

## -------------------------------------------------------------------------------------

## Estudiar equilibrio de clases
table(data$loan_status)
ggplot(data) + 
  geom_histogram(aes(x = loan_status, fill = loan_status), stat = 'count')

## -------------------------------------------------------------------------------------
set.seed(0)
## Crear modelo de predicción usando rf [Partición Aleatoria]
trainIndex <- createDataPartition(data$loan_status, p = .75, list = FALSE, times = 1)
train <- data[ trainIndex, ] 
val   <- data[-trainIndex, ]
table(train$loan_status)
table(val$loan_status)

rfModel <- trainRF(train)
saveRDS(rfModel, file = "model1.rds")
rfModel <- readRDS("model1.rds")
orig_fit <- rfModel
print(rfModel)

rfModel$xlevels[["last_pymnt_d"]] <- union(rfModel$xlevels[["last_pymnt_d"]], levels(as.factor(val$last_pymnt_d)))
rfModel$xlevels[["last_credit_pull_d"]] <- union(rfModel$xlevels[["last_credit_pull_d"]], levels(as.factor(val$last_credit_pull_d)))

prediction_p <- predict(rfModel, val, type = "prob")
prediction_r <- predict(rfModel, val, type = "raw")
result <- my_roc(val, prediction_p, "loan_status", "Paid")

plotdata <- val %>%
  select(loan_status) %>%
  bind_cols(prediction_p) %>%
  bind_cols(Prediction = prediction_r)
table(plotdata$loan_status, plotdata$Prediction)  # columnas son predicciones
ggplot(plotdata) + 
    geom_bar(aes(x = loan_status, fill = Prediction), position = position_fill())

## -------------------------------------------------------------------------------------
## Crear modelo de predicción usando rf [downsampling]
predictors <- select(data, -loan_status)
data_down <- downSample(x = predictors, y = data$loan_status, yname = 'loan_status')
table(data_down$loan_status)

trainIndex <- createDataPartition(data_down$loan_status, p = .75, list = FALSE, times = 1)
train <- data_down[ trainIndex, ] 
val   <- data_down[-trainIndex, ]
table(train$loan_status)
table(val$loan_status)

rfModel <- trainRF(train)
saveRDS(rfModel, file = "model2.rds")
rfModel <- readRDS("model2.rds")
print(rfModel)
prediction_p <- predict(rfModel, val, type = "prob")
prediction_r <- predict(rfModel, val, type = "raw")
result <- my_roc(val, prediction_p, "loan_status", "Paid")

plotdata <- val %>%
  select(loan_status) %>%
  bind_cols(prediction_p) %>%
  bind_cols(Prediction = prediction_r)
table(plotdata$loan_status, plotdata$Prediction)
ggplot(plotdata) + 
  geom_bar(aes(x = loan_status, fill = Prediction), position = position_fill())

# --> validacion real con un subconjunto del conjunto original
test <- data %>%
  sample_n(20000) 
prediction_p <- predict(rfModel, test, type = "prob")
prediction_r <- predict(rfModel, test, type = "raw")
result <- my_roc(test, prediction_p, "loan_status", "Paid")
plotdata <- test %>%
  select(loan_status) %>%
  bind_cols(prediction_p) %>%
  bind_cols(Prediction = prediction_r)
table(plotdata$loan_status, plotdata$Prediction)
ggplot(plotdata) + 
  geom_bar(aes(x = loan_status, fill = Prediction), position = position_fill())
ggplot(plotdata) + 
  geom_bar(aes(x = loan_status, fill = Prediction), position = position_identity())

## -------------------------------------------------------------------------------------
## Crear modelo de predicción usando rf [upsampling]
data_up <- upSample(x = predictors, y = data$loan_status, yname = 'loan_status')
table(data_up$loan_status)

trainIndex <- createDataPartition(data_up$loan_status, p = .75, list = FALSE, times = 1)
train <- data_up[ trainIndex, ] 
val   <- data_up[-trainIndex, ]
table(train$loan_status)
table(val$loan_status)

rfModel <- trainRF(train)
saveRDS(rfModel, file = "model3.rds")
rfModel <- readRDS("model3.rds")
print(rfModel)
prediction_p <- predict(rfModel, val, type = "prob")
prediction_r <- predict(rfModel, val, type = "raw")
result <- my_roc(val, prediction_p, "loan_status", "Paid")

plotdata <- val %>%
  select(loan_status) %>%
  bind_cols(prediction_p) %>%
  bind_cols(Prediction = prediction_r)
table(plotdata$loan_status, plotdata$Prediction)
ggplot(plotdata) + 
  geom_bar(aes(x = loan_status, fill = Prediction), position = position_fill())

## -------------------------------------------------------------------------------------
## Crear modelo de predicción usando rf [SMOTE]
# columnas chr a factor para SMOTE()
data_for_SMOTE <- data %>%
  mutate_if(is.character, as.factor) %>%
  as.data.frame()
data_smote <- SMOTE(loan_status ~ ., data_for_SMOTE, perc.over = 600, dataperc.under = 100)  
table(data_smote$loan_status) 

rfModel <- trainRF(train)
saveRDS(rfModel, file = "model4.rds")
rfModel <- readRDS("model4.rds")
print(rfModel)
prediction_p <- predict(rfModel, val, type = "prob")
prediction_r <- predict(rfModel, val, type = "raw")
result <- my_roc(val, prediction_p, "loan_status", "Paid")

plotdata <- val %>%
  select(loan_status) %>%
  bind_cols(prediction_p) %>%
  bind_cols(Prediction = prediction_r)
table(plotdata$loan_status, plotdata$Prediction)
ggplot(plotdata) + 
  geom_bar(aes(x = loan_status, fill = Prediction), position = position_fill())

## -------------------------------------------------------------------------------------
## Re-sampling dentro de train
trainIndex <- createDataPartition(data$loan_status, p = .75, list = FALSE, times = 1)
train <- data[ trainIndex, ] 
val   <- data[-trainIndex, ]
table(train$loan_status)

train <- sample_n(train, size = 5000)

rfCtrl <- trainControl(
  verboseIter = F, 
  classProbs = TRUE, 
  method = "repeatedcv", 
  number = 10, 
  repeats = 1, 
  summaryFunction = twoClassSummary,
  sampling = "down")
rfParametersGrid <- expand.grid(
  .mtry = c(sqrt(ncol(train)))) 

down_inside <- train(
  loan_status ~ ., 
  data = train,
  method = "rf",
  metric = "ROC",
  trControl = rfCtrl,
  tuneGrid = rfParametersGrid)
saveRDS(down_inside, file = "model-down_inside.rds")
down_inside <- readRDS("model-down_inside.rds")

# Cambiar opción ("up", "smote")
rfCtrl$sampling <- "up"
up_inside <- train(
  loan_status ~ ., 
  data = train,
  method = "rf",
  metric = "ROC",
  trControl = rfCtrl,
  tuneGrid = rfParametersGrid)
saveRDS(up_inside, file = "model-up_inside.rds")
up_inside <- readRDS("model-up_inside.rds")

rfCtrl$sampling <- "smote"
smote_inside <- train(
  loan_status ~ ., 
  data = train,
  method = "rf",
  metric = "ROC",
  trControl = rfCtrl,
  tuneGrid = rfParametersGrid)
saveRDS(smote_inside, file = "model-smote_inside.rds")
smote_inside <- readRDS("model-smote_inside.rds")

# Evaluar
inside_models <- list(down = down_inside,
                      up = up_inside,
                      SMOTE = smote_inside)

inside_resampling <- resamples(inside_models)
summary(inside_resampling, metric = "ROC")

