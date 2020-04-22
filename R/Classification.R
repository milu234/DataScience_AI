############### Load Library ###############################
{
  rm(list = ls(all.names = TRUE)) # Clear environment
  folder_current <- "D:/Trainings/R"; setwd(folder_current); set.seed(123)
  
  suppressPackageStartupMessages(library(data.table))
  source("CommonUtils.R")
  
  #par(mar=c(2,2,1,1)); 
  
  # Constants
  catColumns = 'SPECIES'; strResponse = 'SPECIES'; g_folderImage = "./Images/"
} # load

############### Read data for prepration######################################
train <- fread(input = "./data/iris.csv")
names(train) <- toupper(names(train))

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# First view
dim(train);
str(train);
head(train,2);
summary(train);

############### Prepare data ######################################
# Scale numeric features
train <- ScaleAllCommonNumericColumn(train, exclude = strResponse)$train
fwrite(train, "./data/iris_scaled.csv")

############### CW 'rpart': jump to the open source world ######################################
# install.packages('rpart')
# install.packages('rpart.plot')

############### Random Forest ######################################
# on ppt

############### Ranger (Fast implementaion of Random Forest) ######################################
train <- fread(input = "./data/iris_scaled.csv")

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
summary(train); summary(test); dim(train); dim(test)
detach(package:caret) 

# Model building and prediction by Random Forest (ranger implementation)
library(ranger)

listPredictorNames <- setdiff(colnames(train),strResponse)
formula_all_column <- as.formula(paste(paste0(strResponse, " ~") ,paste(listPredictorNames ,collapse="+")))
formula_all_column
# res <- paste(strResponse, "~")
# ind <- paste(listPredictorNames ,collapse="+")
# frm <- as.formula(paste(res, ind))
# class(frm)

# param list
print("Using default parameters")
l_list_tuned_param <- list(mtry = ceiling(sqrt(ncol(train)))	,num.trees = 500)

# nthread is not getting saved and taking for current machine
l_list_tuned_param[['num.threads']] <- parallel::detectCores()

row_num_org <- nrow(train); col_num_org <- ncol(train)
print(paste0("Model building for ranger with row ", row_num_org, ", col ", col_num_org, ", and parameters "))
print(paste0(paste0(paste0(names(l_list_tuned_param),': ', l_list_tuned_param), collapse = ", ")))

# fit
fit_ranger <- ranger(formula_all_column, train, num.trees = l_list_tuned_param[['num.trees']], mtry = l_list_tuned_param[['mtry']], num.threads = l_list_tuned_param[['num.threads']], write.forest = T, importance = "impurity", classification = T, save.memory = F, seed = 123)
saveRDS(fit_ranger, './model/ranger_model.rds')

print("Self Predicting with ranger")
pred_ranger <- predict(fit_ranger, train[,listPredictorNames, with = F]); 
pred_ranger <- pred_ranger$predictions; 

# Prediction summary
classification_summary(pred = pred_ranger, actual = train[[strResponse]])

# View feature importance/influence from the learnt model
importance_matrix <- data.table(Feature = names(fit_ranger$variable.importance), Gain = fit_ranger$variable.importance)
setorderv(x = importance_matrix, cols = 'Gain', order = -1)
print("Importance features matrix is saved to ./model/ranger_importance.csv")
fwrite(importance_matrix, './model/ranger_importance.csv')

rm(pred_ranger, importance_matrix)

# Prediction from test data
row_num_org <- nrow(test); col_num_org <- ncol(test);
print(paste0("Prediction started with row ", row_num_org, ", col ", col_num_org))
pred_ranger <- predict(fit_ranger, test[,listPredictorNames, with = F]); pred_ranger <- pred_ranger$predictions; 

# Prediction summary
classification_summary(pred = pred_ranger, actual = test[[strResponse]])

# Cleaning  
rm(train, test, pred_ranger, fit_ranger); detach(package:ranger)

############### xgboost - Logistic Regression ######################################
# Logistic Regression's concept on ppt

train <- fread(input = "./data/iris_scaled.csv")

# Since Logistic Regression need binary outcome and hence prepraing data for accordingly

# Keep only two outcome with 'setosa' and 'versicolor'
train <- train[SPECIES == 'setosa' | SPECIES == 'versicolor']

# Convert 'setosa' to 1 and 'versicolor' to 0
train[, SPECIES := ifelse(SPECIES == 'setosa', 1, 0)]

## Calculate cut off probability with response equal to 1. Will be required later
cutoff <- nrow(train[SPECIES == 1])/nrow(train)

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
summary(train); summary(test); dim(train); dim(test)
detach(package:caret) 

# Model building and prediction by XgBoost
library(xgboost)

listPredictorNames <- setdiff(colnames(train),strResponse)
formula_all_column <- as.formula(paste(paste0(strResponse, " ~") ,paste(listPredictorNames ,collapse="+")))

# converting to numeric as xgboost need numeric
train_xgbMat <- xgb.DMatrix(data = as.matrix(train[,listPredictorNames, with = F]), label =  train[[strResponse]]);
row_num_org <- nrow(train); col_num_org <- ncol(train); rm(train); gc()

print("Using default parameters")
l_list_tuned_param <- list(nrounds = 100, max_depth = 6, min_child_weight = 1, subsample = 0.5, colsample_bytree = 0.5, lambda = 0, lambda_bias = 0, gamma = 0, eta = 0.3)

# nthread is not getting saved and taking for current machine
l_list_tuned_param[['nthread']] <- parallel::detectCores()

print(paste0("Building model with xgb.train with row count ", row_num_org, ", col (Including response) ", col_num_org, " and parameters"))
print(paste0(paste0(names(l_list_tuned_param),': ', l_list_tuned_param), collapse = ", "))
fit_xgb <- xgb.train(params = l_list_tuned_param, nrounds = l_list_tuned_param[['nrounds']], data = train_xgbMat, eval.metric = "auc", objective = "reg:logistic",   verbose = T, seed = 123) #prediction = F,
xgb.save(fit_xgb, './model/fit_xgb_c.rds'); 

print("Self Predicting with xgb.train")
pred_xgb <- predict(fit_xgb, train_xgbMat) # for 'binary:logistic' - binary classification. Output is probability.
pred_xgb                                   # Use '"multi:softprob" for probability of ech class in two different column

# Since the prediction is probability and hence using the 'cutoff' to convert to 1/0
pred_xgb <- ifelse(pred_xgb >= cutoff, 1,0)
pred_xgb

# Extract actual and feature names for further use. Rest can be deleted to free memory
actual <- getinfo(train_xgbMat, 'label'); feature_names <- colnames(train_xgbMat); rm(train_xgbMat);gc()

# Prediction summary
classification_summary(pred = pred_xgb, actual = actual)

# View feature importance/influence from the learnt model
importance_matrix <- xgb.importance(model = fit_xgb, feature_names = feature_names)
fwrite(importance_matrix, './model/xgboost_importance_c.csv')

# Prediction from test data
row_num_org <- nrow(test); col_num_org <- ncol(test);
test_xgbMat <- xgb.DMatrix(data = as.matrix(test[,listPredictorNames, with = F])); 

print(paste0("Prediction started with row ", row_num_org, ", col ", col_num_org))
pred_xgb <- predict(fit_xgb, test_xgbMat)

# Since the prediction is probability and hence using the 'cutoff' to convert to 1/0
pred_xgb <- ifelse(pred_xgb >= cutoff, 1,0)

# Prediction summary
classification_summary(pred = pred_xgb, actual = test[[strResponse]])

# Cleaning  
rm(test_xgbMat, test, pred_xgb, fit_xgb, importance_matrix); detach(package:xgboost)

############### One Class SVM ###############################

train <- fread(input = "./data/iris_scaled.csv")

#Mark the data in group one class vs rest
table(train$SPECIES)
train[, SPECIES := ifelse(SPECIES == 'setosa', T, F)]

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
summary(train); summary(test); dim(train); dim(test)
detach(package:caret) 

#Get data for only one class only
train <- train[SPECIES == T]
train[, SPECIES := NULL]

library(e1071)

row_num_org <- nrow(train); col_num_org <- ncol(train);

print("Using default parameters")
l_list_tuned_param <- list(nu=0.05, kernel="radial", gamma = 1/ncol(train), cost = 1)

#description <- paste0(paste0(names(l_list_tuned_param),': ', l_list_tuned_param), collapse = ", ")
print(paste0("Building model with Svm one class with row count ", row_num_org, ", col ", col_num_org, " and parameters"))
print(paste0(paste0(names(l_list_tuned_param),': ', l_list_tuned_param), collapse = ", "))

# Fit model
fit_svmoneclass <- svm(x = train, y = NULL,nu = l_list_tuned_param[['nu']], gamma = l_list_tuned_param[['gamma']], kernel = l_list_tuned_param[['kernel']], cost = l_list_tuned_param[['cost']], type='one-classification', scale = F, fitted = F) 
saveRDS(fit_svmoneclass, './model/svmoneclass_model.rds'); 
rm(train); 

#Predict.  One class predicts FALSE for non member. 
listPredictorNames <- setdiff(names(test),strResponse)
pred_svmoneclass <- predict(fit_svmoneclass, test[,listPredictorNames, with = F])
pred_svmoneclass

# Prediction summary
classification_summary(pred = pred_svmoneclass, actual = test[[strResponse]])

rm(fit_svmoneclass, pred_svmoneclass, test, listPredictorNames); detach(package:e1071)

###############################How to know models are good enough: Bias vs Variance #####################
# On ppt