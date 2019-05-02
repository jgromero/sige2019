## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## Ejemplo basado en 'Deep Learning with R'
## Datos: https://1drv.ms/u/s!AkY5JpKZWcvgzfQF12ZdEtiAIuYu-g
## -------------------------------------------------------------------------------------

library(keras)

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar imágenes
train_dir      <- './cats_and_dogs_small/train/'
validation_dir <- './cats_and_dogs_small/validation/' 
test_dir       <- './cats_and_dogs_small/test/'

img_sample <- image_load(path = './cats_and_dogs_small/train/cats/cat.1.jpg', target_size = c(150, 150))
img_sample_array <- array_reshape(image_to_array(img_sample), c(1, 150, 150, 3))
plot(as.raster(img_sample_array[1,,,] / 255))

# https://tensorflow.rstudio.com/keras/reference/image_data_generator.html 
train_datagen      <- image_data_generator(rescale = 1/255) 
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

# https://tensorflow.rstudio.com/keras/reference/flow_images_from_directory.html
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
## Crear modelo

# Definir arquitectura
# https://tensorflow.rstudio.com/keras/articles/sequential_model.html
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

# Compilar modelo
# https://tensorflow.rstudio.com/keras/reference/compile.html
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)

# Entrenamiento
# https://tensorflow.rstudio.com/keras/reference/fit_generator.html
history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 100,
    epochs = 15,
    validation_data = validation_data,
    validation_steps = 50
  )

# Evaluar modelo
# https://tensorflow.rstudio.com/keras/reference/evaluate_generator.html
model %>% evaluate_generator(test_data, steps = 25)

# Guardar modelo (HDF5)
# https://tensorflow.rstudio.com/keras/reference/save_model_hdf5.html
model %>% save_model_hdf5("dogsVScats.h5")

# Visualizar entrenamiento
plot(history)

## -------------------------------------------------------------------------------------
## Data augmentation

# https://tensorflow.rstudio.com/keras/reference/image_data_generator.html
data_augmentation_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_augmented_data <- flow_images_from_directory(
  directory = train_dir,
  generator = data_augmentation_datagen,  # ¡usando nuevo datagen!
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "binary"        # etiquetas binarias
)

history <- model %>% 
  fit_generator(
    train_augmented_data,
    steps_per_epoch = 100,
    epochs = 15,
    validation_data = validation_data,
    validation_steps = 50
  )

model %>% save_model_hdf5("dogsVScats_augmentation.h5")

model %>% evaluate_generator(test_data, steps = 50)
