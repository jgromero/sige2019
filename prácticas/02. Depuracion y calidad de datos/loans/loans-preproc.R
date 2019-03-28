## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2018-2019
## Juan Gómez Romero
## -------------------------------------------------------------------------------------

library(tidyverse)
library(funModeling)
library(ggplot2)
library(Hmisc)
library(corrplot)

set.seed(1)

## ---------------------------------------------------------------
## 1. Lectura de datos
data_raw <- read_csv('LoanStats_2017Q4.csv', na = c('NA', 'n/a', '', ' '))  # n_max = 10000
glimpse(data_raw)
status <- df_status(data_raw)

## ---------------------------------------------------------------
## 2. Eliminar columnos no útiles

# --> pero mantener 'loan_status'
status <- status %>% 
  filter(variable != 'loan_status')

# Identificar columnas con más del 90% de los valores a 0
zero_cols <- status %>%
  filter(p_zeros > 90) %>%
  select(variable)

# Identificar columnas con más del 50% de los valores a NA
na_cols <- status %>%
  filter(p_na > 50) %>%
  select(variable)

# Identificar columnas con <= 3 valores diferentes
eq_cols <- status %>%
  filter(unique <= 3) %>%
  select(variable)

# Identificar columnas >75% valores diferentes
dif_cols <- status %>%
  filter(unique > 0.75 * nrow(data_raw)) %>%
  select(variable)

# Eliminar columnas
remove_cols <- bind_rows(
  list(
    zero_cols,
    na_cols,
    eq_cols,
    dif_cols
  )
)

data <- data_raw %>%
  select(-one_of(remove_cols$variable))

glimpse(data)
df_status(data)

## ---------------------------------------------------------------
## 3. Eliminar filas no útiles

# Eliminar loan_status no interesantes
data <- data %>%
  filter(loan_status %in% c('Late (16-30 days)', 'Late (31-120 days)', 'In Grace Period', 'Charged Off', 'Current'))

## ---------------------------------------------------------------
## 4. Recodificar valores de clase objetivo 'loan_status'
ggplot(data) +
  geom_histogram(aes(x = loan_status, fill = loan_status), stat = 'count')

data <- data %>%
  mutate(loan_status = case_when(
    loan_status == 'Late (16-30 days)'  ~ 'Unpaid',
    loan_status == 'Late (31-120 days)' ~ 'Unpaid',
    loan_status == 'In Grace Period'    ~ 'Unpaid',
    loan_status == 'Charged Off'        ~ 'Unpaid',
    loan_status == 'Current'            ~ 'Paid'))

ggplot(data) +
  geom_histogram(aes(x = loan_status, fill = loan_status), stat = 'count')

glimpse(data)
df_status(data)

## ---------------------------------------------------------------
## 4. Identificar columnos con alta correlación

# Alta correlacion con la variable objetivo
# correlation_table(data, target='loan_status')   # se necesita imputación y valores numéricos
data_num <- data %>%
  na.exclude() %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)
cor_target <- correlation_table(data_num, target='loan_status')
important_vars <- cor_target %>% 
  filter(abs(loan_status) >= 0.01)

data <- data %>%
  select(one_of(important_vars$Variable))

# Alta correlacion entre sí
data_num <- data %>%
  na.exclude() %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)
rcorr_result <- rcorr(as.matrix(data_num))
cor_matrix <- as.tibble(rcorr_result$r, rownames = "variable")
corrplot(rcorr_result$r, type = "upper", order = "original", tl.col = "black", tl.srt = 45)

# correlated_vars <- c('open_rv_24m', 'acc_open_past_24mths', 'open_il_24m', 'total_rec_prncp', 'total_pymnt', 'total_pymnt_inv', 'grade')
v <- varclus(as.matrix(data_num), similarity="pearson") 
plot(v)
groups <- cutree(v$hclust, 25)
not_correlated_vars <- as.tibble(groups) %>% 
  rownames_to_column() %>% 
  group_by(value) %>% 
  sample_n(1)

data <- data %>%
  select(one_of(not_correlated_vars$rowname))

glimpse(data)
df_status(data)

write_csv(data, 'LoanStats_2017Q4-PreProc.csv')
