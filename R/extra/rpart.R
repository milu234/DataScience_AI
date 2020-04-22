############### Load Library ###############################
{
  rm(list = ls(all.names = TRUE)) # Clear environment
  folder_current <- "D:/Trainings/R"; setwd(folder_current)
  
  suppressPackageStartupMessages(library(data.table))
  source("CommonUtils.R")
  
  #par(mar=c(2,2,1,1)); 
  
  # Constants
  catColumns = 'SPECIES'; strResponse = 'SPECIES'; g_folderImage = "./Images/"
} # load

############### rpart ######################
train <- fread(input = "./data/iris_scaled.csv")

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
summary(train); summary(test); dim(train); dim(test)
detach(package:caret) 

#install.packages('rpart.plot')
library(rpart); library(rpart.plot)
listPredictorNames <- setdiff(colnames(train),strResponse)
formula_all_column <- as.formula(paste(paste0(strResponse, " ~") ,paste(listPredictorNames ,collapse="+")))
fit_rpart <- rpart(formula = formula_all_column, data = train)
rpart.plot(fit_rpart)
summary(fit_rpart)
rm(fit_rpart, train, test); detach(package:rpart.plot);  detach(package:rpart)
