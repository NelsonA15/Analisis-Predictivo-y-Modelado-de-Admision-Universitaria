library(dplyr)
library(ggplot2)
library(glmnet)
library(pls)
library(reshape2)
library(car)
library(heatmaply)


source("C:\\Users\\nmari\\Documents\\CIMAT\\Computo E\\PF\\Addmition/Rutinas.R")

# -- Lectura de datos y tratamiento inicial --
datos <- read.csv("C:\\Users\\nmari\\Documents\\CIMAT\\Computo E\\PF\\Addmition/adm_data.csv")
datos <- datos[, -which(names(datos) == "Serial.No.")]
str(datos)

# Transformación logarítmica en la variable objetivo
datos$Chance.of.Admit <- log(datos$Chance.of.Admit + 1)

# ------------------------------------------
# -- Revision de columnas con NAs --
ColNAs <- colSums(is.na(datos))
ColNAs

# ------------------------------------------
# Distribucion de las variables
numericas <- datos[, sapply(datos, is.numeric)]

for (var in colnames(numericas)) {
  p <- ggplot(datos, aes(x = .data[[var]])) +
    geom_histogram(bins = 30, fill = "steelblue", color = "white") +
    theme_minimal() +
    labs(title = paste("Distribución de", var), x = var, y = "Frecuencia")
  
  print(p) 
}

# ------------------------------------------
# Revisar correlación
cor_matrix <- cor(datos[, -which(names(datos) == "Chance.of.Admit")])

cor_data <- melt(cor_matrix)
ggplot(cor_data, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  labs(title = "Matriz de correlación", x = "Variables", y = "Variables") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
heatmaply(cor_matrix, main = "Mapa de Calor de Correlaciones")

# ------------------------------------------
# VIF
full_model <- lm(Chance.of.Admit ~ ., data = datos)
vif_values <- vif(full_model)
print(vif_values)

# Visualizar
vif_df <- data.frame(Variable = names(vif_values), VIF = vif_values)
ggplot(vif_df, aes(x = reorder(Variable, -VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Factor de inflación de la varianza (VIF)",
       x = "Variable",
       y = "VIF") +
  geom_hline(yintercept = 5, color = "red", linetype = "dashed")

# ------------------------------------------
# -- Conjuntos de entrenamiento y prueba --
P <- ncol(datos) - 1 # Numero de variables
N <- nrow(datos) # Numero de observaciones
cat("Contamos con ", P, " variables predictoras y ", N, " observaciones\n")

X <- as.matrix(datos[, -which(names(datos) %in% c("Chance.of.Admit"))])
Y <- as.numeric(datos$Chance.of.Admit)

train_samples_index <- which(sample(c(TRUE, FALSE), size = N, replace = TRUE, prob = c(0.8, 0.2)))
Xtrain <- X[train_samples_index,]
Xtest <- X[-train_samples_index,]
Ytrain <- Y[train_samples_index]
Ytest <- Y[-train_samples_index]

# ------------------------------------------
# -- Modelo lineal --
lm_modelo <- lm(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = train_samples_index)
summary(lm_modelo)
preds_lm <- predict(lm_modelo, data.frame(Xtest))
Metricas_lm <- CalcularR2(preds_lm, Ytest, N, P)
cat("-- Resultados de modelo lineal -- \nR2 = ", Metricas_lm$R2, "\nR2 ajustado = ", Metricas_lm$R2ajust, "\nMSE = ", Metricas_lm$MSE, "\nMAE = ", Metricas_lm$MAE, "\n")

VisualizarPredReal(preds_lm, Ytest, main = "Gráfica predicción-observación de modelo lineal")

# -- Modelo Ridge con VC --
# Valores lambda propuestos para el modelado
lambda_vals <- 10^seq(-3, 0, length.out = 1024)

ridge_modelo <- cv.glmnet(Xtrain, Ytrain, alpha = 0, lambda = lambda_vals)
plot(ridge_modelo)
best_coefs_ridge <- coef(ridge_modelo, s = ridge_modelo$lambda.min)
preds_ridge <- predict(ridge_modelo, newx = Xtest, s = ridge_modelo$lambda.min)
Metricas_ridge <- CalcularR2(preds_ridge, Ytest, N, P)
cat("-- Resultados de modelo Ridge -- \nR2 = ", Metricas_ridge$R2, "\nR2 ajustado = ", Metricas_ridge$R2ajust, "\nMSE = ", Metricas_ridge$MSE, "\nMAE = ", Metricas_ridge$MAE, "\n")

VisualizarPredReal(preds_ridge, Ytest, main = "Gráfica predicción-observación de modelo Ridge")
# ------------------------------------------
# -- Modelo Lasso con VC --
lasso_modelo <- cv.glmnet(Xtrain, Ytrain, alpha = 1, lambda = lambda_vals)
plot(lasso_modelo)
best_coefs_lasso <- coef(lasso_modelo, s = lasso_modelo$lambda.min)
preds_lasso <- predict(lasso_modelo, newx = Xtest, s = lasso_modelo$lambda.min)
Metricas_lasso <- CalcularR2(preds_lasso, Ytest, N, P)
cat("-- Resultados de modelo Lasso -- \nR2 = ", Metricas_lasso$R2, "\nR2 ajustado = ", Metricas_lasso$R2ajust, "\nMSE = ", Metricas_lasso$MSE, "\nMAE = ", Metricas_lasso$MAE, "\n")

VisualizarPredReal(preds_lasso, Ytest, main = "Gráfica predicción-observación de modelo Lasso")
# ------------------------------------------
# -- Modelo PCR --
pcr_model <- pcr(Chance.of.Admit ~ ., data = datos, subset = train_samples_index, scale = TRUE, validation = "CV")
summary(pcr_model)
preds_pcr <- predict(pcr_model, newdata = datos[-train_samples_index,], ncomp = 4)
Metricas_pcr <- CalcularR2(preds_pcr, Ytest, N, P)
cat("-- Resultados de modelo PCR -- \nR2 = ", Metricas_pcr$R2, "\nR2 ajustado = ", Metricas_pcr$R2ajust, "\nMSE = ", Metricas_pcr$MSE, "\nMAE = ", Metricas_pcr$MAE, "\n")

VisualizarPredReal(preds_pcr, Ytest, main = "Gráfica predicción-observación de modelo PCR")
# ------------------------------------------
# -- Modelo PLS --
pls_model <- plsr(Chance.of.Admit ~ ., data = datos, subset = train_samples_index, scale = TRUE, validation = "CV")
summary(pls_model)
preds_pls <- predict(pls_model, newdata = datos[-train_samples_index,], ncomp = 3)
Metricas_pls <- CalcularR2(preds_pls, Ytest, N, P)
cat("-- Resultados de modelo PLS -- \nR2 = ", Metricas_pls$R2, "\nR2 ajustado = ", Metricas_pls$R2ajust, "\nMSE = ", Metricas_pls$MSE, "\nMAE = ", Metricas_pls$MAE, "\n")

VisualizarPredReal(preds_pls, Ytest, main = "Gráfica predicción-observación de modelo PLS")



# ------------------------------------------
# -- Dispersion de metricas para los modelos propuestos --
num_reps <- 100

Metricas_reps_lm <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_ridge <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_lasso <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_pcr <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_pls <- data.frame(matrix(nrow = num_reps, ncol = 4))


for(num in 1:num_reps){
  rep_train_samples_index <- which(sample(c(TRUE, FALSE), size = N, replace = TRUE, prob = c(0.8, 0.2)))
  X_rep_train <- X[rep_train_samples_index,]
  X_rep_test <- X[-rep_train_samples_index,]
  Y_rep_train <- Y[rep_train_samples_index]
  Y_rep_test <- Y[-rep_train_samples_index]
  
  # Modelo lineal
  lm_rep_model <- lm(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = rep_train_samples_index)
  preds_rep_lm <- predict(lm_rep_model, data.frame(X_rep_test))
  Metricas_rep_lm <- CalcularR2(preds_rep_lm, Y_rep_test, N, P)
  
  # Modelo Ridge
  ridge_rep_model<- cv.glmnet(X_rep_train, Y_rep_train, alpha = 0, lambda = lambda_vals)
  preds_rep_ridge <- predict(ridge_rep_model, newx = X_rep_test, s = ridge_rep_model$lambda.min)
  Metricas_rep_ridge <- CalcularR2(preds_rep_ridge, Y_rep_test, N, P)
  
  # Modelo Lasso
  lasso_rep_model<- cv.glmnet(X_rep_train, Y_rep_train, alpha = 1, lambda = lambda_vals)
  preds_rep_lasso <- predict(lasso_rep_model, newx = X_rep_test, s = lasso_rep_model$lambda.min)
  Metricas_rep_lasso <- CalcularR2(preds_rep_lasso, Y_rep_test, N, P)
  
  # Modelo PCR
  pcr_rep_model <- pcr(Chance.of.Admit ~ ., data = datos, subset = rep_train_samples_index, scale = TRUE, validation = "CV")
  preds_rep_pcr <- predict(pcr_rep_model, newdata = datos[-rep_train_samples_index,], ncomp = 4)
  Metricas_rep_pcr <- CalcularR2(preds_rep_pcr, Y_rep_test, N, P)
  
  # Modelo PLS
  pls_rep_model <- plsr(Chance.of.Admit ~ ., data = datos, subset = rep_train_samples_index, scale = TRUE, validation = "CV")
  preds_rep_pls <- predict(pls_rep_model, newdata = datos[-rep_train_samples_index,], ncomp = 4)
  Metricas_rep_pls <- CalcularR2(preds_rep_pls, Y_rep_test, N, P)
  
  
  Metricas_reps_lm[num, ] <- unlist(Metricas_rep_lm)
  Metricas_reps_ridge[num, ] <- unlist(Metricas_rep_ridge)
  Metricas_reps_lasso[num, ] <- unlist(Metricas_rep_lasso)
  Metricas_reps_pcr[num, ] <- unlist(Metricas_rep_pcr)
  Metricas_reps_pls[num, ] <- unlist(Metricas_rep_pls)
  
  print(num)
}
Metricas_reps_lm$fuente <- "Lineal"
Metricas_reps_ridge$fuente <- "Ridge"
Metricas_reps_lasso$fuente <- "Lasso"
Metricas_reps_pcr$fuente <- "PCR"
Metricas_reps_pls$fuente <- "PLS"

Metricas_reps_combinadas <- bind_rows(
  Metricas_reps_lm, 
  Metricas_reps_ridge, 
  Metricas_reps_lasso, 
  Metricas_reps_pcr, 
  Metricas_reps_pls
)
names(Metricas_reps_combinadas)[1:4] <- c("R2", "R2ajust", "MSE", "MAE")


ggplot(Metricas_reps_combinadas, aes(x = fuente, y = MSE, fill = fuente)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplots de MSE por modelo",
       x = "Modelo",
       y = "MSE")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = MAE, fill = fuente)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplots de MAE por modelo",
       x = "Modelo",
       y = "MAE")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = R2, fill = fuente)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplots de R2 por modelo",
       x = "Modelo",
       y = "R2")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = R2ajust, fill = fuente)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplots de R2 ajustada por modelo",
       x = "Modelo",
       y = "R2 ajustada")

