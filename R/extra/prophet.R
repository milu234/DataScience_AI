#https://facebook.github.io/prophet/docs/quick_start.html
suppressPackageStartupMessages(library(data.table)); suppressPackageStartupMessages(library(forecast)); 
suppressPackageStartupMessages(library(prophet)); suppressPackageStartupMessages(library(ggplot2)); 
source("CommonUtils.R")

# Constants
strResponse = 'TCM1Y'; g_folderImage = "./Images/"

# Get data
data(tcm)
train <- as.data.table(tcm)
names(train) <- toupper(names(train))

# Add date series (was already in tcm)
train$ds <- seq(as.Date("1953/04/01"), by = "month", length.out = nrow(train))

# see the data
head(train)
str(train)

# Keep only relevent data
train <- train[,c("ds", strResponse), with = F]
head(train)

# Change column names
names(train) = c("ds", "y")

# Model building
model <- prophet(train, yearly.seasonality = T, seasonality.prior.scale=5)

# Make furture DF for 1 year
future <- make_future_dataframe(model, periods = 12, include_history = T)

# predict
pred <- predict(model, future)

# Extract train and test pred
pred_train <- pred$yhat[1:nrow(train)]
pred_test <- tail(pred, 12)$yhat #pred$yhat[nrow(train)+1 : nrow(train)+12]

#See the accuracy
print(accuracy(f = pred_train, x = train$y))

# Plot actual and pred in lines
library(ggplot2)
gg <- GetGGPlotForActualAndPred(train, "y", pred_train)
plot(gg)

# ResidualPlot
ResidualPlot(actual = train$y, pred$yhat, folder_tosave = g_folderImage, algorithum_name = "prophet")

# Visualize forecast
plot(model, pred)