############### Load Library ###############################
rm(list = ls(all.names = TRUE)) # Clear environment
folder_current <- "D:/Trainings/R"; setwd(folder_current); set.seed(123)

library(data.table)
source("CommonUtils.R")

#par(mar=c(2,2,1,1)); 

# Constants
catColumns = 'ORIGIN'; strResponse = 'MPG'; g_folderImage = "./Images/"
############### Read data for prepration ######################################
train <- fread(input = "./data/mpg.csv")
names(train) <- gsub(pattern = " ", replacement = "_", x = names(train))
names(train) <- toupper(names(train))

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# First view
dim(train);
str(train);
head(train,2);
summary(train);

############### Prepare data ######################################

# Create one hot coding and scale numeric features
train <- scale_and_dummy_creation(train, strResponse, listFactorFeatures = catColumns, file_output = "./data/mpg_train_EncodedScaled.csv")

# independent features list
listPredictorNames <- setdiff(names(train),strResponse)

suppressPackageStartupMessages(library(caret))

# Collinearity 
lin_cor <- findLinearCombos(data.matrix(train[,listPredictorNames, with = F]))
cor_cols <- NA
if (length(lin_cor$remove) > 0)
{
  print(paste0("Correleated columns : ", paste0(names(train)[lin_cor$remove], collapse = ", "),  "\n", sep = " "))
  cor_cols <- names(train)[lin_cor$remove]
  #train[, (cor_cols) := NULL]
} # if

# VIF
fit <- lm(MPG ~ . , data = train)
summary(fit)
train$ORIGIN_JAPANESE <- NULL
fit <- lm(MPG ~ . , data = train)
vif_values <- sort(car::vif(fit) , decreasing = TRUE)

# Since we have done some changes above and hence taking list of  again
listPredictorNames <- setdiff(names(train),strResponse)

# Varinace analysis
nzv <- nearZeroVar(data.matrix(train[,listPredictorNames, with = F]), freqCut = 95/5, uniqueCut = 5, saveMetrics = T)
nzv[nzv$zeroVar == TRUE,]

#Keep trimmed data only
if(nrow(nzv[nzv$zeroVar == TRUE,]) > 0) train[,(row.names(nzv[nzv$zeroVar == TRUE,])) := NULL]; # dim(train)

head(nzv[order(nzv$percentUnique, decreasing = T),], 10) #

#Duplicate row # No Duplicate
uniqueTest <- dim(unique(train)) == dim(train)
if(uniqueTest[1] == F | uniqueTest[2] == F)
{
  rowDuplicate <- which(duplicated(train) == T)
  
  cat(paste("Dumplicate rows count:",length(rowDuplicate), ". Removing Duplicates\n"))
  train <- unique(train)
} else cat("No Duplicates\n")

detach(package:caret) 

# Normality
response_value <- train[[strResponse]]
ifelse(nrow(train) > 3 && nrow(train) < 5000, ifelse(shapiro.test(t(response_value))$p.value > 0.05,"Normality in response variable indicated\n", "Non - Normality in response variable indicated\n"), ("Normality test not done as number of records are more than 5000"))

# Distribution analysis: Let us see the variuos distributions
library(fitdistrplus)#; library(logspline)

#Now lets use descdist:
# par(mar=c(2,2,1,1)); # par(mfrow=c(1,1))
descdist(response_value, discrete = FALSE)

# Let's fit a and plot
fit_dist <- fitdist(response_value, "norm") # unif norm
par(mar=c(2,2,1,1)); 
plot(fit_dist) # par(mar=c(2,2,1,1)); 
detach(package:fitdistrplus); #detach(package:logspline) 

# Boxcox: Box cox transformation and lamda estimation 
library(MASS)

# Get the formula required by boxcox
listPredictorNames <- setdiff(colnames(train),strResponse)
formula_all_column <- as.formula(paste(paste0(strResponse, " ~") ,paste(listPredictorNames ,collapse="+")))
formula_all_column

# Build boxcox
bc <- boxcox(formula_all_column, data = train, plotit = TRUE) # yjPower for -ve values
lambda <- bc$x[which.max(bc$y)] 
lambda
lambda <- -0.5

# Transform
# http://onlinestatbook.com/2/transformations/box-cox.html
train$MPG_TRANFORMED <- ((train$MPG ^ lambda) - 1) / lambda

# View the data
head(train$MPG)
head(train$MPG_TRANFORMED)

# View the transformation
par(mfrow=c(1,2)) # Two graph in plot
hist((train$MPG)); hist((train$MPG_TRANFORMED))

par(mfrow=c(1,1))

# Get the inverse of Boxplot to get values back
inv_boxCox <- function(x, lambda) if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda)
MPG2 <- inv_boxCox(train[['MPG_TRANFORMED']], lambda); 
all.equal(train$MPG, MPG2)

############### Simple Linear Regression ######################################
train <- fread(input = "./data/mpg_train_EncodedScaled.csv")

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
dim(train); dim(test)
detach(package:caret) 


# fit linear regression
fit_lm <- lm(MPG ~ . , data = train)
summary(fit_lm); 

par(mfrow=c(2,2))#so you can see all 4 plots at once
par(mar=c(2,2,1,1)); 
plot(fit_lm)
par(mfrow=c(1,1))#to reset the graphic window to just one graph

# Predict
pred <- predict(fit_lm, test)

# Error Caluclations
MAE_lm <- ModelMetrics::mae(predicted = pred,actual = test[[strResponse]]) 
RMSE_lm <- ModelMetrics::rmse(predicted = pred,actual = test[[strResponse]])
mean(test[[strResponse]])
print(paste0("Overall MAE: ", round(MAE_lm, 2), ", RMSE: ", round(RMSE_lm, 2)))

# CW: Self predict on train and errors

# paste0("c(",paste0(test[1:10][[strResponse]], collapse = ","), ")")
# paste0("c(",paste0(pred[1:10], collapse = ","), ")")

# Plot actual and pred in lines
library(ggplot2)
gg <- GetGGPlotForActualAndPred(test, strResponse, pred)
plot(gg)
detach(package:ggplot2) 

# ResidualPlot
ResidualPlot(actual = test[[strResponse]], pred, folder_tosave = g_folderImage, algorithum_name = "lm")
############### Glmnet ######################################
train <- fread(input = "./data/mpg_train_EncodedScaled.csv")

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
dim(train); dim(test)
detach(package:caret) 

# fit glmnet (Lasso and Ridge) regression
suppressPackageStartupMessages(library(glmnet))

listPredictorNames <- setdiff(colnames(train),strResponse)

# It need matrix as input
train <- as.matrix(train)

# fit glmnet. alpha=1 is the lasso penalty, and alpha=0 the ridge penalty.
glmnet_fit <- glmnet(x = train[,listPredictorNames], y = train[,strResponse] , alpha=0.5, family = 'gaussian',  standardize = F)
summary(glmnet_fit); 

# Predict
test <- as.matrix(test)
pred <- predict(object = glmnet_fit, newx = test[,listPredictorNames], s = min(glmnet_fit$lambda)) # 'lambda.min' , type = "response"
pred <- as.vector(pred)

# Error Caluclations
MAE_glmnet <- ModelMetrics::mae(predicted = pred,actual = test[, strResponse]) 
RMSE_glmnet <- ModelMetrics::rmse(predicted = pred,actual = test[, strResponse]) #RMSE <- sqrt(sum((pred_xgb - getinfo(train_xgbMat, "label"))^2)/length(pred_xgb))
print(paste0("Overall MAE: ", round(MAE_glmnet, 2), ", RMSE: ", round(RMSE_glmnet, 2)))

# Plot actual and pred in lines
library(ggplot2)
test <- as.data.table(test)
gg <- GetGGPlotForActualAndPred(test, strResponse, pred)
plot(gg)
detach(package:ggplot2) 

# CW: Plot actual and pred in points

# ResidualPlot
ResidualPlot(actual = test[,strResponse, with = F], pred, folder_tosave = g_folderImage, algorithum_name = "glmnet")

detach(package:glmnet)

############ Generic Tree flow ###############
#on ppt

############### Xgboost ############################
train <- fread(input = "./data/mpg_train_EncodedScaled.csv", check.names = T)

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
dim(train); dim(test)
detach(package:caret) 

listPredictorNames <- setdiff(colnames(train),strResponse)

library(xgboost)

formula_all_column <- as.formula(paste(paste0(strResponse, " ~") ,paste(listPredictorNames ,collapse="+")))
formula_all_column

# converting to numeric as xgboost need numeric
train_xgbMat <- xgb.DMatrix(data = as.matrix(train[,listPredictorNames, with = F]), label =  train[[strResponse]]);
row_num_org <- nrow(train); col_num_org <- ncol(train); rm(train); gc()

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
print("Using default parameters") 
l_list_tuned_param <- list(nrounds = 100, max_depth = 6, min_child_weight = 1, subsample = 0.5, 
                           colsample_bytree = 0.5, eta = 0.3 , lambda = 0, lambda_bias = 0, gamma = 0)

# nthread is not getting saved and taking for current machine
l_list_tuned_param[['nthread']] <- parallel::detectCores()

print(paste0("Building model with xgb.train with row count ", row_num_org, ", col (Including response) ", col_num_org, " and parameters"))
print(paste0(paste0(names(l_list_tuned_param),': ', l_list_tuned_param), collapse = ", "))
fit_xgb <- xgb.train(params = l_list_tuned_param, nrounds = l_list_tuned_param[['nrounds']], data = train_xgbMat, eval.metric = "mae", objective = "reg:linear",   verbose = T, seed = 123) #prediction = F,
xgb.save(fit_xgb, './model/fit_xgb.rds'); 

print("Self Predicting with xgb.train")
pred_xgb <- predict(fit_xgb, train_xgbMat)

# Extract actual and feature names for further use. Rest can be deleted to free memory
actual <- getinfo(train_xgbMat, 'label'); feature_names <- colnames(train_xgbMat); rm(train_xgbMat);gc()

# Calculate and Save MAE and RMSE
MAE_xgboost_self <- ModelMetrics::mae(predicted = pred_xgb,actual = actual) 
RMSE_xgboost_self <- ModelMetrics::rmse(predicted = pred_xgb,actual = actual) #RMSE <- sqrt(sum((pred_xgb - getinfo(train_xgbMat, "label"))^2)/length(pred_xgb))
print(paste0("Overall MAE: ", round(MAE_xgboost_self, 2), ", RMSE: ", round(RMSE_xgboost_self, 2)))

print("xgboost: Draw and Save Residual plot")
ResidualPlot(actual = actual, pred = pred_xgb, folder_tosave = g_folderImage, algorithum_name = 'xgboost_self')

# View feature importance/influence from the learnt model
importance_matrix <- xgb.importance(model = fit_xgb, feature_names = feature_names)
fwrite(importance_matrix, './model/xgboost_importance.csv')

print(paste0("Importance features graph saved to ", g_folderImage,"xgboost_importance.pdf"))
pdf(paste0(g_folderImage,"xgboost_importance.pdf"))
xgb.plot.importance(importance_matrix = importance_matrix)
dev.off()

#View the trees from a model. You can dump the tree you learned using xgb.dump into a text file.
print("Xgboost model tree is written to ./model/xgboost_model_stats.json")
xgb.dump(model = fit_xgb, with_stats = T, fname = './model/xgboost_model_stats.json', dump_format = "json")

# Predict on test data
# converting to numeric as xgboost need numeric
test_xgbMat <- xgb.DMatrix(data = as.matrix(test[,listPredictorNames, with = F])); 
row_num_org <- nrow(test); col_num_org <- ncol(test);

print(paste0("Prediction started with row ", row_num_org, ", col ", col_num_org))
pred <- predict(fit_xgb, test_xgbMat)

# Error Caluclations
MAE_xgboost <- ModelMetrics::mae(predicted = pred,actual = test[[strResponse]]) 
RMSE_xgboost <- ModelMetrics::rmse(predicted = pred,actual = test[[strResponse]])
print(paste0("Overall MAE: ", round(MAE_xgboost, 2), ", RMSE: ", round(RMSE_xgboost, 2)))

# Plot actual and pred in lines
library(ggplot2)
gg <- GetGGPlotForActualAndPred(test, strResponse, pred)
plot(gg)
detach(package:ggplot2) 

# ResidualPlot
ResidualPlot(actual = test[[strResponse]], pred, folder_tosave = g_folderImage, algorithum_name = "xgboost")

# clean
rm(fit_xgb, importance_matrix); detach(package:xgboost)
