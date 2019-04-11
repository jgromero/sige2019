## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

library(tidyverse)
library(ggplot2)
library(datasets)
library(caret)
library(partykit)

## ---------------------------------------------------------------
## 1. Lectura de datos
data(iris)
data <- as.tibble(iris)

## ---------------------------------------------------------------
## 2. Exploración
ggplot(data) + geom_histogram(aes(x = Species, fill = Species), stat = 'count')
# ggsave("histogram.png", width = 10, height = 6, units = "cm")

ggplot(data) + 
  geom_point(aes(x = Sepal.Length, y = Sepal.Width, color = Species))

ggplot(data) + 
  geom_point(aes(x = Petal.Length, y = Petal.Width, color = Species))

## ---------------------------------------------------------------
## 3. Modelo de predicción básico
trainIndex <- createDataPartition(data$Species, p = .75, list = FALSE, times = 1)
train <- data[ trainIndex, ] 
val   <- data[-trainIndex, ]

rpartCtrl <- trainControl(
  verboseIter = FALSE, 
  classProbs = TRUE, 
  summaryFunction = mnLogLoss)
rpartParametersGrid <- expand.grid(
  .cp = c(0.05, 0.1, 0.25))
rpartModel <- train(
  Species ~ ., 
  data = train, 
  method = "rpart", 
  metric = "logLoss", 
  trControl = rpartCtrl, 
  tuneGrid = rpartParametersGrid)

print(rpartModel)

rpartModel_party <- as.party(rpartModel$finalModel)
plot(rpartModel_party)

prediction <- predict(rpartModel, val, type = "prob") 
cm_train <- confusionMatrix(prediction, val[["Species"]])

## ---------------------------------------------------------------
## 4. Modelo de predicción básico con evaluación mediante one-vs-all
rpartCtrl$summaryFunction <- multiClassSummary
rpartModel <- train(
  Species ~ ., 
  data = train, 
  method = "rpart", 
  metric = "AUC", 
  trControl = rpartCtrl, 
  tuneGrid = rpartParametersGrid)
print(rpartModel)

## ---------------------------------------------------------------
## 5. SVM con one-vs-one, one-vs-all
# Instalacion gmum.r: devtools::install_github("gmum/gmum.r", ref="dev")
# Error en OS X: https://github.com/velocyto-team/velocyto.R/issues/2#issuecomment-352584213
library(gmum.r)
train_df <- as.data.frame(train)
val_df   <- as.data.frame(val)

# One-vs-one 
sv.ovo <- SVM(x=train_df[,1:4], y=train_df[,5], class.type="one.versus.one", verbosity=0)
preds <- predict(sv.ovo, val_df[,1:4])
acc.ovo <- sum(diag(table(preds, val_df$Species)))/sum(table(preds, val_df$Species))

plot(sv.ovo)

# One-vs-all
sv.ova <- SVM(Species ~ ., data = train_df, class.type="one.versus.all", verbosity=0)
preds <- predict(sv.ova, val_df[,1:4])
acc.ova <- sum(diag(table(preds, val_df$Species)))/sum(table(preds, val_df$Species)) 

plot(sv.ova)