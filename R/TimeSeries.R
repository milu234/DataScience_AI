############### Load Library ###############################
rm(list = ls(all.names = TRUE)) # Clear environment
folder_current <- "D:/Trainings/R"; setwd(folder_current); set.seed(123)

suppressPackageStartupMessages(library(data.table)); suppressPackageStartupMessages(library(forecast)); 
suppressPackageStartupMessages(library(fpp)); suppressPackageStartupMessages(library(ggplot2)); 
source("CommonUtils.R")

#par(mar=c(2,2,1,1));

# Constants
strResponse = 'TCM1Y'; g_Season1 <- 12; g_folderImage = "./Images/"
############### Read data for prepration ######################################
# Let us take inbuilt data from package 'fpp'
data(tcm)
head(tcm, 100)

train <- as.data.table(tcm)
names(train) <- toupper(names(train))

############### Update data, Basic plot, first view  ######################################
# Various ways to plot
ts.plot(train[[strResponse]])
ts.plot(tcm[,1], tcm[,2], tcm[,3], tcm[,4])
ts.plot(tcm)

# Theoritical frequency
freq <- findfrequency(train[[strResponse]])
print(paste0("Suggested frequency is ", freq))

g_Season1 <- 12

# Conversion to msts(multi seasonal time series) object
ts_msts_raw <- msts(train[[strResponse]], seasonal.periods = g_Season1, start=1953+3/12) # , end=2017+26/365
ts.plot(ts_msts_raw)

############################# Acf and Pacf ###############################
# If there is seasonality, the ACF at the seasonal lag (e.g., 12 for monthly data) will be large and positive.
# For seasonal monthly data, a large ACF value will be seen at lag 12 and possibly also at lags 24, 36, . . .
# For seasonal quarterly data, a large ACF value will be seen at lag 4 and possibly also at lags 8, 12, . . .

#Let us take perfect sample data and see the seasonality through Acf and Pacf
data("ausbeer")
head(ausbeer) # See it is quaterly data

Acf(ausbeer) # at lag of 4 large ACF is visible
Pacf(ausbeer)

# see the frequency on data on which we are going to work
Acf(ts_msts_raw)
Pacf(ts_msts_raw)

##################### Further exploration #################################
#Just FYI: Extract date in any one of the following ways
as.Date(tcm)
as.yearmon(time(tcm))
as.yearmon(as.Date(tcm))

# Add date column in data table
train$DATE <- as.Date(tcm)
str(train)

# Detail View using ggplot
gg <- ggplot(data = train, aes(x = DATE, y = eval(as.name(strResponse)))) +  geom_line()
gg <- gg + theme(strip.text.y = element_text(hjust = 0))
gg <- gg +  ylab(strResponse)
plot(gg)

################################ Missing value imputation ################################

# Any missing value then replace "Using linear interpolation for non-seasonal series and a periodic stl 
#decomposition with seasonal series to replace missing values."
if(anyNA(ts_msts_raw))  ts_msts_raw <- na.interp(ts_msts_raw)

# replacing each NA with the most recent non-NA prior to it.
if(anyNA(ts_msts_raw)) ts_msts_raw <- na.locf(ts_msts_raw) 

#CW: See how to replace each NA with the most recent non-NA next to it.

########################Time series components: see the 3 parts ###########################
ts_decom <- decompose(ts_msts_raw, type = "mult");
par(mfrow = c(1,1)); plot(ts_decom)

# STL: "Seasonal and Trend decomposition using Loess". Loess is a method for estimating nonlinear relationships
fit <- stl(ts_msts_raw, t.window=g_Season1, s.window="periodic", robust=TRUE)
plot(fit)

# zoo: Takes irregular too
# xts: Extends zoo and have lot of function for manipulations. matrix, date in any form, keeps on ordering. See 
#the cheetsheet in document folder
# lubridate: It is more simple way to manipulate time data. See in extra forlder during leasure period.

############### Similarity and  Trend analysis ######################################
# Similarity bewteen two time series
sim <- TimeSeriesSimilarity(train[[1]], train[[2]])
sim

# Trend analysis
dt <- train[, c('DATE', strResponse), with = F]
trend <- getTimeSeriesTrend(dt, frequency = g_Season1, pdfFileName = paste0(g_folderImage, strResponse, "_timeseries_trend.pdf"),whetherRemoveOutlier = F, summarise = T, str_date_breaks = "18 months")
trend
# open './Images/TCM1Y_timeseries_trend.pdf' and see the pdf

############### Anomaly Detection ####################################
# Concept on ppt
library(AnomalyDetection); #library(scales);

dt <- train[, c('DATE', strResponse), with = F]
dt$DATE <- as.POSIXct(dt$DATE)
#Algorithum 1: Let it decide the seasonality ( AnomalyDetectionTs) and with seasonality (AnomalyDetectionVec)
tryCatch(anomaly_timeseries <- AnomalyDetectionVec(dt[[2]], max_anoms=0.05, direction='both',   period = g_Season1, plot=T, verbose = F), error = function(cond){print(cond);})
if(!(is.null(anomaly_timeseries) | nrow(anomaly_timeseries$anoms) == 0)) {
  plot(anomaly_timeseries$plot)
  anomaly_timeseries <- anomaly_timeseries[["anoms"]]; 
  anomaly_timeseries <- suppressWarnings(data.table(timestamp = dt[anomaly_timeseries$index,"DATE"], anoms = anomaly_timeseries$anoms));
  print('Anomaly records are'); print(anomaly_timeseries)
  rm(anomaly_timeseries)
}

# Algorithum 2: tsoutliers from forecast library
anomaly_timeseries <- NA
tryCatch(ts_outliers <- tsoutliers(x = ts_msts_raw), error = function(cond){print(cond);})
if(!(is.null(ts_outliers)) & length(ts_outliers$index) > 0)
{
  anomaly_timeseries <- suppressWarnings(data.table(timestamp = dt[ts_outliers$index,"DATE"], anoms = dt[ts_outliers$index,2]));
  names(anomaly_timeseries) <- c('DATE', strResponse)
  print('Anomaly records are'); print(anomaly_timeseries)
  
  # CW:  Find common Anomaly.Pause and do the CW
  # Draw the graph of Anomaly points on main TS graph
  
 # rm(anomaly_timeseries)
}

# plot anomaly
gg <- ggplot(data = dt, aes(x = DATE, y = eval(as.name(strResponse)))) +  geom_line()
gg <- gg + geom_point(data = anomaly_timeseries, aes(x = DATE, y = eval(as.name(strResponse)), color = 'red'))
gg <- gg + theme(strip.text.y = element_text(hjust = 0))
gg <- gg +  ylab(strResponse)
plot(gg)

############### ARIMA ######################################
# Concept on ppt
# Get data
data(tcm)
train <- as.data.table(tcm)
names(train) <- toupper(names(train))

# Convert to Timeseries. It can be done by 'ts' too
ts_msts_raw <- msts(train[[strResponse]], seasonal.periods = g_Season1, start=1953+3/12)

# Basic plot
#par(mar=c(2,2,1,1));
par(mfrow = c(1,1));
ts.plot(ts_msts_raw, col = c("blue"), main = "Basic Plot")

# for each season of 12 month duration
seasonplot(ts_msts_raw)

# for each 1 month duration
monthplot(ts_msts_raw,xlab="Month",xaxt="n", main="Monthly seasonal - with mean for each month")
axis(1,at=1:12,labels=month.abb,cex=0.8)

# Plot for last 4 lags. One lag means ? and second lag means ?
lag.plot(ts_msts_raw, lags = 4, do.lines = F)

# Augmented Dickey-Fuller Test for stationary (constant mean, constant variance, an autocovariance that does not depend on time)
adf.test(ts_msts_raw, alternative = "stationary") #p-value = 0.5351 hence null hypothesis "not stationary" is true

#see the 3 parts
ts_decom <- decompose(ts_msts_raw, type = "mult");
par(mfrow = c(1,1)); plot(ts_decom)

# See acf and pacf
old_layout <- par(no.readonly = F); layout(matrix(c(1,2), nrow = 1, ncol = 2,byrow = T))
Acf(ts_decom$random, na.action = na.exclude, main = 'acf')
Pacf(ts_decom$random, na.action = na.exclude, main = 'pacf')
par(old_layout)

#Auto ARIMA
fit <- auto.arima(ts_msts_raw,  trace = T, stepwise = F, approximation = F)
fit

#Residual test

# To see Uncorrelated nature
Acf(residuals(fit))

# Box-Pierce or Ljung-Box test statistic for examining the null hypothesis of independence in a given time series.
# Also as 'portmanteau' tests - H0: Residuals are independent (Uncorrelated). H1: Residuals are not independent  
Box.test(residuals(fit), lag = g_Season1, fitdf = 0, type = "Ljung"); # fitdf = p+q, provided of course that lag > fitdf

# See together actual and self predicted value
ts.plot(ts_msts_raw, fitted(fit), gpars = list(xlab = "", ylab = "", lty = c(1,2)), col = c("red","blue"), main = "")

# Pred for next 12 months
pred <- forecast(object = fit, h = 12) # forecast.Arima
pred

# See the forcasted part
plot(pred)

# Convert to data table for easy processing and saving
df_pred <- as.data.table(as.data.frame(pred))
dim(df_pred)

#See the accuracy
print(accuracy(f = pred$fitted, x = ts_msts_raw))
# Theil's U statistic is a relative accuracy measure. Less than 1 - forecasting better than guessing, 1 - 
#forecasting is as good as guessing, More than 1 - forecasting is worse than guessing

#CW: Segregate the train data in 85% and 15%

############### ARIMAX ######################################
# Get data
data(tcm)
train <- as.data.table(tcm)
names(train) <- toupper(names(train))
head(train, 20)

listPredictorNames <- setdiff(colnames(train),strResponse)
listPredictorNames

# Convert to Timeseries. It can be done by 'ts' too
ts_msts_raw <- msts(train[[strResponse]], seasonal.periods = g_Season1, start=1953+3/12)

# ARIMAX
fit <- auto.arima(ts_msts_raw,  trace = T, stepwise = F, approximation = F, xreg = train[,listPredictorNames, with = F])
fit

#Residual test

# To see Uncorrelated nature
Acf(residuals(fit))

# Box-Pierce or Ljung-Box test statistic for examining the null hypothesis of independence in a given time series.
# Also as 'portmanteau' tests.
Box.test(residuals(fit), lag = g_Season1, fitdf = 0, type = "Ljung"); # fitdf = p+q, provided of course that lag > fitdf. P > 0.05 to NUll test pass and hence independent

# See together actual and self predicted value
ts.plot(ts_msts_raw, fitted(fit), gpars = list(xlab = "", ylab = "", lty = c(1,2)), col = c("red","blue"), main = "")

# Pred for next data  - Just for practice let us take the past data
pred <- forecast(object = fit, xreg = train[,listPredictorNames, with = F])
pred
plot(pred)

df_pred <- as.data.frame(pred)
dim(df_pred)

#See the accuracy
print(accuracy(f = pred$fitted, x = ts_msts_raw)) # Observe Theil's U

############### ETS(Error,Trend,Seasonal) ######################################
# model: A three-letter code indicating the model to be estimated using the ETS classification and notation. 
#The possible inputs are "N" for none, "A" for additive, "M" for multiplicative, or "Z" for automatic selection.

data(tcm)
train <- as.data.table(tcm)
names(train) <- toupper(names(train))
listPredictorNames <- setdiff(colnames(train),strResponse)

# Convert to Timeseries. It can be done by 'ts' too
ts_msts_raw <- msts(train[[strResponse]], seasonal.periods = g_Season1, start=1953+3/12)

# Fit the model
fit <- ets(ts_msts_raw, model="ZZZ") # ANN
fit
plot(forecast(fit, h=12), ylab="")
fit$par

# prediction
pred <- forecast(fit, h=12)

#See the accuracy
print(accuracy(f = pred$fitted, x = ts_msts_raw))

############### Ensemple of ARIMA and ETS ######################################
# Fresh object
ts_msts_raw <- msts(train[[strResponse]], seasonal.periods = g_Season1, start=1953+3/12)

# Use function written in CommonUtils
fit_hybrid_ets_arima <- hybridf(ts_msts_raw, h = 12)
plot(fit_hybrid_ets_arima)

#See the accuracy
print(accuracy(f = fit_hybrid_ets_arima$fitted, x = ts_msts_raw))

#Summary: Accuracy of all 4 methods
# ARIMA
#          ME         RMSE        MAE       MPE       MAPE   ACF1 Theil's U
# Test set 0.00427109 0.4081358 0.2718243 -0.041417 4.667253 0.002645943 0.9380142
# 
# ARIMAX
#          ME           RMSE      MAE        MPE         MAPE     ACF1 Theil's U
# Test set 0.0006214053 0.1288491 0.09078346 -0.09477949 1.773088 0.01549712 0.3737926
# 
# ETS
#          ME           RMSE          MAE       MPE       MAPE    ACF1    Theil's U
# Test set 0.0009250209 0.4665865 0.2842241 0.07385401 4.651722 0.1350703 0.9548151
# 
# hybrid
#            ME          RMSE        MAE    MPE        MAPE     ACF1      Theil's U
# Test set -0.004697968 0.4305187 0.2692179 -0.1350783 4.45693 0.08640834 0.9033226

#######################CW: Try Prophet ###############################
# Concept on ppt
# https://facebook.github.io/prophet/docs/quick_start.html