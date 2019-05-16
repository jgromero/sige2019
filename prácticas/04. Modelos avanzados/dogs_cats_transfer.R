## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## Ejemplo basado en 'Deep Learning with R'
## -------------------------------------------------------------------------------------

library(keras)

## -------------------------------------------------------------------------------------
## Clasificación con red ya entrenada
model_resnet50 <- application_resnet50(
  weights = "imagenet"
)

img_path <- './dragon1.jpg'
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)

x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)

preds <- model_resnet50 %>% predict(x)
imagenet_decode_predictions(preds, top = 3)[[1]]

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar imágenes
train_dir      <- './cats_and_dogs_small/train/'
validation_dir <- './cats_and_dogs_small/validation/' 
test_dir       <- './cats_and_dogs_small/test/'

train_datagen      <- image_data_generator(rescale = 1/255) 
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

train_data <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

validation_data <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

test_data <- flow_images_from_directory(
  directory = test_dir,
  generator = test_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

## -------------------------------------------------------------------------------------
## Extracción de características
# Cargar capa convolutiva de VGG16, pre-entrenada con ImageNet
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

# Crear función que extrae feature map --con dimension (4, 4, 512) para VGG16
batch_size <- 20
extract_features <- function(directory, sample_count) {
  
  # crear generador para transformación de imágenes de entrada
  datagen <- image_data_generator(rescale = 1/255)
  
  # crear arrays de salida, inicialmente a 0
  #  features: características (números de samples x (4, 4, 512) )
  #  labels: (número de samples)
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  # leer de directorio pasado como parámetro
  data_generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  # extraer batches hasta acumular el número de samples pasado como parámetro
  i <- 0
  while(TRUE) {
    batch <- generator_next(data_generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    i <- i + 1
    if (i * batch_size >= sample_count)
      break 
  }
  
  # devolver feature map y labels
  list(
    features = features,
    labels = labels
  ) 
}

# Generar conjuntos de entrenamiento, validación y test con características extraídas
train      <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test       <- extract_features(test_dir, 1000)

# Redimensionar datos de los feature maps
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features      <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features       <- reshape_features(test$features)

# Crear clasificador con datos de los feature maps (red neuronal)
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% 
  fit(
    train$features, train$labels,
    epochs = 30,
    batch_size = 20,
    validation_data = list(validation$features, validation$labels)
  )

plot(history)

model %>% save_model_hdf5("dogsVScats_feature-extraction.h5")

model %>% evaluate(test$features, test$labels)

## -------------------------------------------------------------------------------------
## Fine tuning (solo con GPU)

# 1. Crear modelo completo utilizando la capa convolutiva de VGG16 pre-entrenada y nuestra capa FC
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# 2. Congelar pesos de la capa convolutiva VGG16 y compilar
freeze_weights(conv_base)

# 3. Entrenamiento 'end-to-end' (pero solo se modifica de la capa FC)
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = validation_data,
    validation_steps = 50
  )

# 4. Descongelar capas de las red base
unfreeze_weights(conv_base, from = "block3_conv1")

# 5. Entrenar capa descongelada y FC
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_data,
    validation_steps = 50
  )

model %>% save_model_hdf5("dogsVScats_fine-tuning.h5")

model %>% evaluate_generator(test_generator, steps = 50)
