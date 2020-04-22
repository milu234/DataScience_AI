############### Load Library ###############################
rm(list = ls(all.names = TRUE)) # Clear environment
folder_current <- "D:/Trainings/R"; setwd(folder_current); set.seed(123)

suppressPackageStartupMessages(library(data.table))
source("CommonUtils.R")

#par(mar=c(2,2,1,1)); 

# Constants
catColumns = c('SPECIES'); strResponse = 'SPECIES'; g_folderImage = "./Images/"
############### Principle Component Analysis ###############################
# Read scaled data
train <- fread(input = "./data/iris_scaled.csv")

#Drop response
train[, (strResponse) := NULL]

# Do the PCA
pca <- prcomp(train, scale = F, center = F)
var_pca_cum <- round(cumsum((pca$sdev^2/sum(pca$sdev^2))*100),2)
print(paste0('Variance explained is ', paste0(var_pca_cum, collapse = ", ")))
pca_component_count <- 2;

print("Get transformed data")
dim(train)
dim(pca$rotation)

head(train)
head(pca$rotation)

# Do Matrix multiplication to get PCA data
train_pca <- as.data.table(as.matrix(train) %*% pca$rotation)
dim(train_pca)
head(train_pca)

# Store only 2 column as we need it for printing cluster
train_pca <- train_pca[,1:pca_component_count]

# Cleaning
rm(pca, var_pca_cum, train)
############### Determining number of clusters ###############################
# Read scaled data
train <- fread(input = "./data/iris_scaled.csv")

#Drop response
train[, (strResponse) := NULL]

# Get optimal count of cluster
library(cluster); library(ggplot2); library(factoextra)

listPredictorNames <- setdiff(colnames(train),strResponse)

#par(mfrow = c(1,2)); par(mfrow = c(1,1));

#png("./images/silhouette.png")
plot(fviz_nbclust(train[, listPredictorNames, with = F], kmeans, method = "silhouette", k.max = 10))
plot(fviz_nbclust(train[, listPredictorNames, with = F], kmeans, method = "wss", k.max = 10))

#CW: Save two image in png and see side by side

rm(train); detach(package:cluster); detach(package:factoextra); detach(package:ggplot2)

# Note: For huge data, above approach may not be possible and hence get count of cluster thorugh package 'fpc', 'NbClust'
############### KMeans Custer ###############################
# Read scaled data
train <- fread(input = "./data/iris_scaled.csv")

listPredictorNames <- setdiff(colnames(train),strResponse)

#Drop response
train[, (strResponse) := NULL]

library(fpc)
# Count of Cluster from 'silhouette' and 'wss' graph
countCluster <- 3

print(paste0("Suggested the count of Cluster is ", countCluster))

print(paste0("Overall: Calculating the Cluster with dimension (rxc) count ", nrow(train), ", " , ncol(train)))
fit <- clusterboot(data = train, B = 100, clustermethod = kmeansCBI, k = countCluster, count = F, seed = 123)
print(paste0("The stability of cluster is ", paste0(round(fit$bootmean,2), collapse = ", "))) # 0.6 and above means good cluster
print(paste0("The stability statistics of clusters are range: ", paste0(round(range(fit$bootmean), 2), collapse = " - "), ", mean: ", round(mean(fit$bootmean), 2)))

# Sort cluster data (ID and respective count)
cluster_id <- fit$result$partition; 

# Before sorting, the cluster  ID and respective count
table(cluster_id)

# # Sort the cluster list too
# cluster_id <- sortCluster(ClusterColumn =  cluster_id)
# 
# # Before sorting, the cluster  ID and respective count
# table(cluster_id)

# Print the Clusters
train_pca$CLUSTERID <- as.factor(cluster_id)
head(train_pca)

# CW: View cluster. Generate graph - x axis as PC1 and y axis as PC2 and cluster as color

#################################### Get Cluster's attributes ################################
print("Loading actual data for Cluster's attributes")
train <- fread(input = "./data/iris.csv")
names(train) <- toupper(names(train))

print("Converting to appropriate data type")
features <- intersect(names(train), catColumns); if(length(features) > 0) train[, features] <- lapply(train[, features, with = F], as.factor); rm(features) 

# Get Cluster id to calculate cluster's attributes
train$CLUSTERID <- as.factor(cluster_id)

print(paste0(" Get Cluster's attributes with dimension (rxc) count ", nrow(train), ", " , ncol(train)))
list_features <- setdiff(names(train), "CLUSTERID")
dt <- getClusterContentWithMaxPercentageAttr(train, count_cluster = countCluster, col_cluster = "CLUSTERID", FeaturesForCluster = list_features, HeaderPrefix = "Cluster",TopNFactorCount = 3)

file_name <- "./model/clusters_attributes.csv"
print(paste0("Cluster's Attributes file is saved to ", file_name))
fwrite(dt, file_name)
#CW: Do the Analysis

################################################# Get outliers using Cluster ###########################

# Get centers of each cluster
centers <- setattr(as.data.frame(fit$result$result["centers"],stringsAsFactors = FALSE), "class",  c("data.frame","data.table")) 
colnames(centers) <- gsub("centers.", "", names(centers))
centers <- cbind(ClusterID = c(1:countCluster), centers)
dim(centers)
head(centers)

# Get data table with centers of each cluster
dt <- data.table(ClusterID = cluster_id)
dt <- merge(x = dt, y = centers, by = "ClusterID", sort = F )
head(dt)
dim(dt)


# Get distances of each actual data from centers
distances <- sqrt(rowSums((as.matrix(train[,listPredictorNames, with = F]) - dt[, -1, with = F])^2))
head(distances)
summary(distances)
dt$distances <- distances; # ; View(head(dt)); head(dt)

# Choose the distance on Quantile or on top N count
q75 <- quantile(x = distances, probs = 0.75)[[1]];
color <- ifelse(distances >= q75,"red", "blue")
par(mar = c(2,2,1,1)); plot(distances, col = color)

# Outlier calculation. Note: It will give index and not actual data
outliers <- order(distances, decreasing = T)

# Top 5 plot
countTopN <- 5
color <- ifelse(distances >= distances[outliers[countTopN]],"red", "blue")
par(mar = c(2,2,1,1)); plot(distances, col = color)

#CW: For each of Cluster, plot clusters with outliers as red and normal as blue color

rm(train, dt); detach(package:fpc);

############### Hierarchical clustering ###############################
# Read scaled data
train <- fread(input = "./data/iris_scaled.csv")

#Drop response
train[, (strResponse) := NULL]

# Compute pairewise distance matrices between each row
dist_amoug_rows <- dist(train, method = "euclidean")

# Hierarchical clustering results
fit_hc <- hclust(dist_amoug_rows, method = "complete")

# Visualization of hclust
plot(fit_hc, labels = F, hang = -1)

# Add rectangle around 3 groups
rect.hclust(fit_hc, k = 3, border = 2:4) 

# Cut into 3 cluster/groups
hc_cluster_id <- cutree(fit_hc, k = 3)
hc_cluster_id

#CW: How to get difference of cluster_id and hc_cluster_id
#CW: Genearate cluster attribute and did you observe any differences from observation done by Kmeans

##################KNN on ppt ##############################
# CW: when ever you are free

#################Applications on ppt ######################
