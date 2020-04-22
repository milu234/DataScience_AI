############### Load Library ###############################
{
  rm(list = ls(all.names = TRUE)) # Clear environment
  folder_current <- "D:/Courses/R_Prog/R"; setwd(folder_current); set.seed(123)
  
  suppressPackageStartupMessages(library(data.table))
  source("CommonUtils.R")
  
  #par(mar=c(2,2,1,1)); 
  
  # Constants
  catColumns = 'SPECIES'; strResponse = 'SPECIES'; g_folderImage = "./Images/"
} # load


############### Read data for prepration######################################
train <- fread(input = "./data/iris.csv")
names(train) <- toupper(names(train))
names(train)

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))
str(train)

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


suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.80 , list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
summary(train); summary(test); dim(train); dim(test)
detach(package:caret)

library(ranger)
listPredictorNames <- setdiff(colnames(train),strResponse)
formula_all_column <- as.formula(paste(paste0(strResponse, " ~"), paste(listPredictorNames,collapse = "+")))
formula_all_column

# param list
print("Using default parameters")
l_list_tuned_param <- list(mtry = ceiling(sqrt(ncol(train)))	,num.trees = 500)

# nthread is not getting saved and taking for current machine
l_list_tuned_param[['num.threads']] <- parallel::detectCores()
l_list_tuned_param[['num.threads']]


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

#########################################Random Forest Ends ##############################################
