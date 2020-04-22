library(Amelia)

# Get inbuilt data
data(freetrade)

# See it
head(freetrade, 10)
summary(freetrade)

# CW: See how many rows, Cols and trend for missing data

# Impute
imputed_datasets <- amelia(freetrade, m = 5, ts = "year", ords = "polity", noms = "signed", cs = "country", p2s = 2)
imputed_datasets

# See the impted data
head(imputed_datasets$imputations[[1]], 10)

# Joint plots
par(mfrow=c(1,2))
hist(freetrade$tariff, col="blue", border="white")
hist(imputed_datasets$imputations[[1]]$tariff, col="green", border="white")
