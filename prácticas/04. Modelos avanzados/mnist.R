## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

library(keras)

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar datos

# Cargar MNIST
mnist   <- dataset_mnist()

image(1:28, 1:28, as.matrix(mnist$train$x[1,,]), col=gray((0:255)/255))

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

# Redimensionar imagenes
x_train <- array_reshape(x_train, c(nrow(x_train), 784))  # 60.000 arrays de 784 elementos
x_test  <- array_reshape(x_test,  c(nrow(x_test),  784))  # 60.000 arrays de 784 elementos

# Reescalar valores de imagenes a [0, 255]
x_train <- x_train / 255
x_test  <- x_test  / 255

# Crear 'one-hot' encoding
y_train <- to_categorical(y_train, 10)
y_test  <- to_categorical(y_test,  10)

## -------------------------------------------------------------------------------------
## Crear modelo

# Definir arquitectura
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation  = 'softmax')

summary(model)

# Compilar modelo
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Entrenamiento
history <- model %>% 
  fit(
    x_train, y_train, 
    epochs = 20, 
    batch_size = 128, 
    validation_split = 0.2
  )

# Guardar modelo (HDF5)
model %>% save_model_hdf5("minist_rnn.h5")

# Visualizar entrenamiento
plot(history)

## -------------------------------------------------------------------------------------
## Evaluar modelo con datos de validación

# Calcular metrica sobre datos de validación
model %>% evaluate(x_test, y_test)

# Obtener predicciones de clase
predictions <- model %>% 
  predict_classes(x_test)

# Crear matriz de confusión
library(caret)
cm <- confusionMatrix(as.factor(mnist$test$y), as.factor(predictions))
cm_prop <- prop.table(cm$table)
plot(cm$table)

library(dplyr)
library(scales)
cm_tibble <- as_tibble(cm$table)
ggplot(data = cm_tibble) + 
  geom_tile(aes(x=Reference, y=Prediction, fill=n), colour = "white") +
  geom_text(aes(x=Reference, y=Prediction, label=n), colour = "white") +
  scale_fill_continuous(trans = 'reverse') 
