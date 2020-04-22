############### Load Library ###############################################
rm(list = ls(all.names = TRUE)) # Clear environment
folder_current <- "D:/Courses/R"; setwd(folder_current); set.seed(123)

library(data.table)
source("CommonUtils.R")

#par(mar=c(2,2,1,1)); 

# Constants
g_folderImage = "./Images/"
############## DATA FRAME vs MATRIX vs DATA TABLE ######################################

############## BASIC R WITH DATA TABLE ######################################
# Read data
train <- fread(input = "./data/mpg.csv")
names(train) <- toupper(names(train))
head(train,2);

# First view
dim(train)
str(train)
head(train,2)
summary(train)

# Constants
catColumns = 'ORIGIN'; strResponse = 'MPG'

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# Data Extraction.
train[1,MPG] 
train[1,'MPG', with = F]
train[,MPG]
train[,'MPG', with = F]
train[['MPG']]

#Add columns in data table
train[,MYCOLUMN:= 1]
head(train,2)

train[,('MYCOLUMN2'):= 1]
head(train,2)

#Update columns in data table
train[,('MYCOLUMN2'):= 2]
head(train,2)

#Delete columns in data table
train[,c('MYCOLUMN', 'MYCOLUMN2'):= NULL]
head(train,2)

#Search and replace using grep and gsub
train$ORIGIN_TEMP <- train$ORIGIN
grep('American', train$ORIGIN_TEMP)

train$ORIGIN_TEMP <- gsub('American', 'American_New', train$ORIGIN_TEMP)
train$ORIGIN_TEMP
train$ORIGIN_TEMP <- NULL

# use stringi for detail string changes

# Description: This provides data dictioery for given data table.
dt <- GetDataDictionary(train); dt[, c('MissingCount', 'PercentageMissing') := NULL]
fwrite(dt, "train_DataDictionary.csv", row.names = F)

############## Apply functions #####################
# List returns
list_numeric_cols <- setdiff(names(train),catColumns)
train[,list_numeric_cols] <- lapply(train[,list_numeric_cols, with = F], function(x) as.numeric(x))

# Vector return
mean_all_col <- sapply(train[,list_numeric_cols, with = F], mean)

# For each row
sum_all_row <- apply(train[,list_numeric_cols, with = F], 1, sum)

# For each col
mean_all_col_2 <- apply(train[,list_numeric_cols, with = F], 2, mean)

# equality
all.equal(mean_all_col, mean_all_col_2)

# CW. 1. Convert CYLINDERS to factor. 2. Sum 'ENGINE DISP' and HORSEPOWER
 
############## Feature Transformation #####################################
head(train)
unique(train$YEAR)

# Create new feature with year with two groups - less than 75 and greater than 75
train[, YEAR_GRP := ifelse(YEAR <= 75, "<=75", ">75")]
train$YEAR_GRP_2 <- ifelse(train$YEAR <= 75, "<=75", ">75")
head(train, 5)

# Remove temp columns
train$YEAR_GRP <- NULL; train$YEAR_GRP_2 <- NULL;

############## Handle Missing Data #######################################
# Is there any missing data anywhere
anyNA(train)

# Impute some missing data for practice
dt_with_missing_data <- copy(train)
dt_with_missing_data[, ACCELERATE := ifelse(ACCELERATE >=18.0 & ACCELERATE <= 20.0, NA, ACCELERATE)]
head(dt_with_missing_data, 20) # see the NA
anyNA(dt_with_missing_data)

# See the count rowwise
naRow <- sapply(c(1:nrow(dt_with_missing_data)), function(x) (sum(is.na(dt_with_missing_data[x,]) == T)))
naRow
sum(naRow)

# use of do.call
do.call(sum, list(sapply(c(1:nrow(dt_with_missing_data)), function(x) (sum(is.na(dt_with_missing_data[x,]) == T)))))

# CW: get missing count of each col and use do.call to get total sum

# CW: Read at home or at leasure
# Detail Analysis of Missing data
library(mice)

# Get pattern, if any
missing_pattern <- md.pattern(dt_with_missing_data)
missing_pattern

# Get complete records only
dim(dt_with_missing_data)
dt_with_missing_data <- train[complete.cases(dt_with_missing_data),]
dim(dt_with_missing_data)

# CW: Amelia -> Look at leisure

rm(dt_with_missing_data)

############## Encode Categorical features ###############################
unique(train[, catColumns, with = F])

# Create one hot coding
library(caret)
fit <- dummyVars("~. -1",data=train[, catColumns, with = F], fullRank=T)
dt <- as.data.table(predict(fit,train[, catColumns, with = F]))

# delete old category column
train[, (catColumns) := NULL]

# Combine
train <- cbind(train, dt)
rm(dt); detach(package:caret);

############## Various Table #####################################################
# Read data afresh
train <- fread(input = "./data/mpg.csv")
names(train) <- toupper(names(train))

# Change data types
train[,catColumns] <- lapply(train[,catColumns, with = F], function(x) as.factor(x))

# Base functions
table(train$ORIGIN)

# Cross tabs
table(train$ORIGIN, train$CYLINDERS)

# Utility functions
dt <- GetCountAndPercentage(train, 'ORIGIN')
dt
#Cw: Understand the output
# ORIGIN COUNT COUNT_PERCENTAGE
# 1: American   245            62.50
# 2: Japanese    79            20.15
# 3: European    68            17.35

dt <- GetGroup_Cat_Num(train, colCat = 'ORIGIN',numericColumn = 'CYLINDERS')
dt
dt <- GetCountAndPercentage_TwoFactors(train, colName = 'ORIGIN',colName2 = 'CYLINDERS')
dt

############## Basic plots ################################################
numericColumn <- 'ACCELERATE'

plot(train[[numericColumn]])
hist(train[[numericColumn]])

############## ggplot #####################################################
library(ggplot2); library(ggrepel)

# Temp feature for plots
numericColumn <- 'ACCELERATE'

# Box Plot
gg2  <- ggplot(data = train, aes(x = numericColumn, y = eval(as.name(numericColumn))))
gg2 <- gg2 + geom_boxplot(fill = "lightblue", outlier.colour = "red", outlier.shape = 1, varwidth = TRUE)
gg2 <- gg2 + labs(title=paste0("Spread of " , numericColumn), size = 10) + labs(x=numericColumn, y= paste0("Value of ", numericColumn), size = 10)
plot(gg2)

# histogram
gg0  <- ggplot(data = train, aes(x = eval(as.name(numericColumn))))
gg0 <- gg0 + geom_histogram(col="black", fill="lightblue", alpha = .2) # , stat="count" breaks=seq(mi, mx, by = step), binwidth  = bins,
gg0 <- gg0 + geom_vline(aes(xintercept=mean(eval(as.name(numericColumn)), na.rm=T)), color="green", linetype="dashed", size=2)
gg0 <- gg0 + labs(title= paste0("Distribution of " , numericColumn), size = 10) + labs(x=numericColumn, y="Count", size = 10)
plot(gg0)

#CW: Scatter plot: Let us see Mpg vs numericColumn
#CW: Draws kernel density estimate (smoothed version of the histogram). A useful alternative to the histogram 
# if for continuous data that comes from an underlying smooth distribution.

############### Descriptive Analysis ####################################
#Call all functions one by one

Desc_Numeric_Single(train, strResponse = strResponse, folderImageDescriptive = g_folderImage)
Desc_Numeric_Double(train, strResponse=strResponse, folderImageDescriptive = g_folderImage)
Desc_Categorical_Single(train, strResponse=strResponse, folderImageDescriptive = g_folderImage)
Desc_Numeric_AllatOnce(train, strResponse=strResponse, folderImageDescriptive = g_folderImage)

################ Miscellaneous ##########################################
nNumber <- 2
ifelse(nNumber < 1, 'Less than 1', 'Greater than 1')

# CW: Nest ifelse to print 'between 1 and 2' and 'Greater than 2'

############## Handle Outliers ############################################
# Single Numerics: Box Plot, Normality using 2-3 Sigma

# Multi variables:
# 1. By Cluster 2. One CLass SVM 3. By Anomaly - Isolation forest
