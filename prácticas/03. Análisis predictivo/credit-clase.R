library(tidyverse)
library(caret)

# Usando "German credit card data"
# http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
# Variable de clasificación: Class
data(GermanCredit)
data <- as_tibble(GermanCredit)
glimpse(data)

df_status(data)
ggplot(data) + geom_histogram(aes(x = Class, fill = Class), stat = 'count')

