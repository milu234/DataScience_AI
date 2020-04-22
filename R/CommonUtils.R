
#Put Cluster's attribute with content in specified file
#count_cluster = countCluster; col_cluster = "CLUSTERID"; FeaturesForCluster = list_independent_features; HeaderPrefix = "Cluster";TopNFactorCount = 5
getClusterContentWithMaxPercentageAttr <- function(train, count_cluster, col_cluster, FeaturesForCluster, HeaderPrefix = "Cluster",TopNFactorCount = 1, listCoreExcluded = NA)
{
  suppressPackageStartupMessages(library(sets))
  
  # Create Data table with structure - Cluster1..n
  dt_cluster <- data.table(matrix(NA, nrow = 1, ncol = count_cluster)); 
  for(i in c(1:count_cluster)) names(dt_cluster)[i] <- paste(HeaderPrefix, i, sep = "")
  dt_cluster <- dt_cluster[complete.cases(dt_cluster),];
  
  # Store size of cluster
  dtClusterSize <- GetCountAndPercentage(train, col_cluster); dtClusterSize <-dtClusterSize[, c(1:2), with = F]; 
  dtClusterSize$CLUSTERID <- as.integer(dtClusterSize$CLUSTERID); setorderv(dtClusterSize, col_cluster)
  
  FeaturesForCluster_Trimmed <- FeaturesForCluster 
  
  # Iterate for each factor and get the frequencies
  for (col_cluster_factor in FeaturesForCluster) # col_cluster_factor = "NATURE_OF_BUSINESS" #' count_cluster <- 6; x <- 1
  {
    if (is.factor(train[[col_cluster_factor]]) | is.logical(train[[col_cluster_factor]]))
    {
      tab_attr_cluster <- table(train[[col_cluster_factor]], train[[col_cluster]])
      
      #Shiv: It is for top N. Initially it was for 1 only. x = 6
      if( TopNFactorCount == 1 | is.logical(train[[col_cluster_factor]])) list_max_attr <- lapply(c(1:count_cluster), function(x) names(which.max(tab_attr_cluster[,x])))
      else list_max_attr <- lapply(c(1:count_cluster), function(x) do.call(paste, c(as.list(names(tab_attr_cluster[order(-tab_attr_cluster[,x])[1:TopNFactorCount],x])), sep = ",")))
      #else list_max_attr <- lapply(c(1:count_cluster), function(x) do.call(paste, c(as.list(names(tab_attr_cluster[order(-tab_attr_cluster[,x])[1:min(TopNFactorCount, dtClusterSize[CLUSTERS == x, COUNT])],x])), sep = ",")))
      
      dt_cluster <- rbind(dt_cluster, list_max_attr);
      
      list_max_attr  <- NULL;
    }  # is.factor
    else if (is.numeric(train[[col_cluster_factor]]) == T | is.integer(train[[col_cluster_factor]]) == T)
    {
      # Create table of col cluster and mean of feature colume
      tab_attr_cluster <- aggregate( formula = train[[col_cluster_factor]] ~ train[[col_cluster]], data = train, FUN = mean); colnames(tab_attr_cluster) <- c(col_cluster, col_cluster_factor)
      
      #Prepare for Quantile function
      Fun_ecdf <- ecdf(train[[col_cluster_factor]]); 
      
      # Add mean and quantile
      temp_ecdf <- Fun_ecdf(tab_attr_cluster[,col_cluster_factor]) * 100
      tab_attr_cluster$FeatureQuntile <- paste(round(tab_attr_cluster[,col_cluster_factor], getRoundCuttoff(tab_attr_cluster[,col_cluster_factor])), round(temp_ecdf, getRoundCuttoff(temp_ecdf)), sep = ",")
      
      tab_attr_cluster[,col_cluster_factor] <- NULL
      
      tab_attr_cluster <- as.data.table(t(tab_attr_cluster)); for(i in c(1:count_cluster)) names(tab_attr_cluster)[i] <- paste(HeaderPrefix, i, sep = "")
      dt_cluster <- rbind(dt_cluster, tab_attr_cluster[2,]);
      
      Fun_ecdf <- NULL; temp_ecdf <- NULL
    } # if num int
    else { 
      FeaturesForCluster_Trimmed <- setdiff(FeaturesForCluster_Trimmed, col_cluster_factor)
      print(paste("Not implemented for ", paste(col_cluster_factor, class(train[[col_cluster_factor]]), sep = ":"), sep = "")) 
    } # else
    
    tab_attr_cluster <- NULL;
  } # col_cluster_factor
  
  # Add feature columns
  dt_cluster <- cbind(Features = FeaturesForCluster_Trimmed, dt_cluster)
  
  # Add size of each clsuter
  # Shiv : dt$Freq is throwing error and hence changing logic
  # dt <- table(train[[col_cluster]]);  dt <- setattr(as.data.frame(dt,stringsAsFactors = FALSE), "class",  c("data.frame","data.table"));
  # dt <- data.table(t(dt$Freq)); for(i in c(1:count_cluster)) names(dt)[i] <- paste(HeaderPrefix, i, sep = "")
  # dt <- cbind(Features = "Size",dt)
  # dt_cluster <- rbind(dt_cluster, dt)
  
  # Add size of each clsuter
  dtClusterSize <- data.table(t(dtClusterSize)); 
  for(i in c(1:count_cluster)) names(dtClusterSize)[i] <- paste(HeaderPrefix, i, sep = "")
  dtClusterSize <- cbind(Features = "Size",dtClusterSize)
  dt_cluster <- rbind(dt_cluster, dtClusterSize[nrow(dtClusterSize)]); #View(dt_cluster)
  
  # Add Core column to identify the feature that were part of cluster formation
  if(!is.na(listCoreExcluded))
  {
    # Remove extra features, if any
    listCoreExcluded <- setdiff(listCoreExcluded, setdiff(listCoreExcluded,FeaturesForCluster))
    
    # Create temp data table with Core = Y/N
    dt <- data.table(Features = dt_cluster[,Features], CORE = 'Y')
    dt[Features %in% listCoreExcluded, CORE := 'N']
    size_with_Y <- as.character(nrow(dt[CORE == 'Y'])); dt[Features == 'Size', CORE := size_with_Y]
    
    # Inlcude Core from temp data table
    dt_cluster <- cbind(dt_cluster[,1], dt[,2], dt_cluster[,c(2:ncol(dt_cluster)), with = F])
    
    rm(dt)
  } # if(!is.na(listCoreExcluded))
  
  detach(package:sets);
  
  return(dt_cluster)
  
} # getClusterContentWithMaxPercentageAttr


#  get cluster DF with 3 clsuters agnes, vegan and cboot
getClusterDfForFactorVariables <- function(Data, numOfClusters = 5, missingValueinData = F)
{
  library(vegan) # for vegdist function to use Gower
  library(ClustOfVar)
  library(fpc)   # for Cluster Test
  library(cluster)
  
  #Within the R package cluster there is ?daisy which will create a dissimilarity matrix for mixed data. Then you can use ?agnes or other clustering functions.
  demo_cluster_daisy <- daisy(Data, metric = "gower") 
  demo_cluster_agnes <- agnes(demo_cluster_daisy, diss = T, metric = "euclidean")
  
  #plot(demo_cluster_agnes) # , , hang = 0.1
  cluster_agnes_count <- numOfClusters # As per Plot's suggestion
  #rect.hclust(demo_cluster_agnes, k=cluster_agnes_count, border = "red") # The dendrogram suggests five clusters. You can draw the rectangles on the dendrogram
  
  # get the cluster labels
  cluster_groups_agnes <- cutree(demo_cluster_agnes, k= cluster_agnes_count)
  
  #Above Clustering works with missing data and below clustering works with only non NA (missing) data
  if (missingValueinData == T)
  {
    clusterDF <- data.frame(cluster_agnes = cluster_groups_agnes)
    
    detach(package:vegan); detach(package:ClustOfVar); detach(package:fpc); detach(package:cluster);
    
    return(clusterDF)
    
  }
  
  # for Vegan Package, converting factor with string data to numeric only DF
  Data_demographic_Num <- convertFactorDFtoIntegerDF(Data)
  
  #Second approach - cluster the data using a hierarchical clustering algorithm (Ward's method):
  #   Create the distance matrix.
  cluster_veg_dist <- vegdist(Data_demographic_Num, method = "gower") #, na.rm = T - Let it fails if NA comes
  cluster_veg_hclust <- hclust(cluster_veg_dist, method = "ward.D2")  # Ward method is most appropriate for quantitative variables, and not binary variables.
  
  #   Plot the dendrogram.
  #plot(cluster_veg_hclust)     
  cluster_vegan_count <- numOfClusters
  #rect.hclust(cluster_veg_hclust, k=7) # The dendrogram suggests five clusters. You can draw the rectangles on the dendrogram
  
  # get the cluster labels
  cluster_vegan_groups <- cutree(cluster_veg_hclust, k = cluster_vegan_count)
  
  
  #Let's run clusterboot() to test clusters
  
  # set the desired number of clusters                               
  kbest.p <- numOfClusters       
  
  #   Run clusterboot() with hclust "clusterboot interface" ('clustermethod=hclustCBI') using Ward's method ('method="ward"') and kbest.p clusters 
  cboot.hclust <- clusterboot(Data_demographic_Num,clustermethod=hclustCBI, method="ward.D2", k=kbest.p,count = F)
  
  #   The results of the clustering are in cboot.hclust$result. The output of the hclust() function is in cboot.hclust$result$result. 
  #   cboot.hclust$result$partition returns a vector of clusterlabels. 
  cboot_groups<-cboot.hclust$result$partition  
  
  # -- results --
  
  # The vector of cluster stabilities. # Values close to 1 indicate stable clusters
  cboot.hclust$bootmean                                   
  
  # The count of how many times each cluster was # dissolved. By default clusterboot() runs 100 # bootstrap iterations. 
  # Clusters that are dissolved often are unstable. 
  cboot.hclust$bootbrd                                    
  
  #Club the finding together for all 3 clustering methods
  
#   pdf(file = "Demographic_Cluster.pdf")
#   
#   # For Agene (Grower) clustering method
#   cluster_content_MaxOnly <- getClusterContentWithMaxPercentageAttr(Data, cluster_groups_agnes)
#   
#   gg <- ggplot(data=cluster_content_MaxOnly, aes(x=ClusterID, y=attrPercentage, fill=attrName)) + guides(fill=FALSE) #Remove legend for a particular aesthetic (fill)
#   gg <- gg +  geom_bar(stat="identity",  width=.5) 
#   gg <- gg + geom_text(size=3, aes(label = cluster_content_MaxOnly$attrName), position = "stack", hjust=0, vjust=2) 
#   gg <- gg + xlab("Attributes") + ylab("% Contribution") + ggtitle("For Agene (Grower) clustering method") + theme(plot.title = element_text(lineheight=.8, face="bold"))
#   gg
#   
#   # For Vegan (Gower distance and Ward) clustering method
#   cluster_content_MaxOnly <- getClusterContentWithMaxPercentageAttr(Data, cluster_vegan_groups)
#   
#   gg <- ggplot(data=cluster_content_MaxOnly, aes(x=ClusterID, y=attrPercentage, fill=attrName)) + guides(fill=FALSE) #Remove legend for a particular aesthetic (fill)
#   gg <- gg +  geom_bar(stat="identity",  width=.5) 
#   gg <- gg + geom_text(size=3, aes(label = cluster_content_MaxOnly$attrName), position = "stack", hjust=0, vjust=2) 
#   gg <- gg + xlab("Attributes") + ylab("% Contribution") + ggtitle("For Vegan (Gower distance and Ward) clustering method") + theme(plot.title = element_text(lineheight=.8, face="bold"))
#   gg
#   
#   # For Cluster test (Ward) clustering method
#   cluster_content_MaxOnly <- getClusterContentWithMaxPercentageAttr(Data, cboot_groups)
#   
#   gg <- ggplot(data=cluster_content_MaxOnly, aes(x=ClusterID, y=attrPercentage, fill=attrName)) + guides(fill=FALSE) #Remove legend for a particular aesthetic (fill)
#   gg <- gg +  geom_bar(stat="identity",  width=.5) 
#   gg <- gg + geom_text(size=3, aes(label = cluster_content_MaxOnly$attrName), position = "stack", hjust=0, vjust=2) 
#   gg <- gg + xlab("Attributes") + ylab("% Contribution") + ggtitle("# For Cluster test (Ward) clustering method") + theme(plot.title = element_text(lineheight=.8, face="bold"))
#   gg
#   
#   dev.off()
  
  clusterDF <- data.frame(cluster_agnes = cluster_groups_agnes, cluster_vegan = cluster_vegan_groups,cluster_cboot = cboot_groups)
  
  return(clusterDF)
} # getClusterDfForFactorVariables
 
#Within the R package cluster there is ?daisy which will create a dissimilarity matrix for mixed data. Then you can use ?agnes or other clustering functions.
getClusterGroups_daisy_agnes <- function(Data, dataType)
{
  
  metricType <-ifelse(dataType == "factorOnly","gower","euclidean")
  
  demo_cluster_daisy <- daisy(Data, metric= metricType) 
  demo_cluster_agnes <- agnes(demo_cluster_daisy, diss = T, metric = "euclidean")
  #summary(demo_cluster_agnes)
  #print(demo_cluster_agnes)
  #plot(demo_cluster_agnes) # , , hang = 0.1
  
  # As per Plot's suggestion
  cluster_agnes_count <- 4
  
  #rect.hclust(demo_cluster_agnes, k=cluster_agnes_count, border = "red") # The dendrogram suggests five clusters. You can draw the rectangles on the dendrogram
  
  #Let's extract and print the clusters: A convenience function for printing out the countries in each cluster, along with the values 
  #   for red meat, fish, and fruit/vegetable consumption. 
  
  # get the cluster labels
  cluster_groups <- cutree(demo_cluster_agnes, k=cluster_agnes_count)
  
  return(cluster_groups)
}

#Description: This takes two similar mapping and provides mapping (only) A to B. Used by GetClusterLabelMappingVector function
GetClusterLabelMapping <- function(clusteringA, clusteringB) {
  library(clue)
  idsA <- unique(clusteringA)  # distinct cluster ids in a
  idsB <- unique(clusteringB)  # distinct cluster ids in b
  nA <- length(clusteringA)  # number of instances in a
  nB <- length(clusteringB)  # number of instances in b
  if (length(idsA) != length(idsB) || nA != nB) {
    stop("number of cluster or number of instances do not match")
  }
  
  nC <- length(idsA)
  tupel <- c(1:nA)
  
  # computing the distance matrix
  assignmentMatrix <- matrix(rep(-1, nC * nC), nrow = nC)
  for (i in 1:nC) {
    tupelClusterI <- tupel[clusteringA == i]
    solRowI <- sapply(1:nC, function(i, clusterIDsB, tupelA_I) {
      nA_I <- length(tupelA_I)  # number of elements in cluster I
      tupelB_I <- tupel[clusterIDsB == i]
      nB_I <- length(tupelB_I)
      nTupelIntersect <- length(intersect(tupelA_I, tupelB_I))
      return((nA_I - nTupelIntersect) + (nB_I - nTupelIntersect))
    }, clusteringB, tupelClusterI)
    assignmentMatrix[i, ] <- solRowI
  }
  
  # optimization
  result <- solve_LSAP(assignmentMatrix, maximum = FALSE)
  attr(result, "assignmentMatrix") <- assignmentMatrix
  
  detach(package:clue)
  
  return(result)
} # GetClusterLabelMapping

#Description: This takes two similar mapping and provides mapping A to B in vector. labels from cluster A will be matched on the labels from cluster B
GetClusterLabelMappingVector <- function(listClusteLabelA, listClusteLabelB)
{
  # Get matching
  matching <- GetClusterLabelMapping(listClusteLabelA, listClusteLabelB)
  
  print(matching)
  
  # map the labels from cluster A
  clusterB <- listClusteLabelB  
  tmp <- sapply(1:length(matching), function(i) { clusterB[which(listClusteLabelB == i)] <<- matching[i] })
  
  # cleaninig
  matching <- NULL; tmp <- NULL
  
  return(clusterB)
} # GetClusterLabelMappingVector

# Function for missingValue DF, missing Field's Count and percentage Missing Fields
getMissingValueDetails <- function(Data, bIDandUniqueColInvestigation = F, bRowMissingInvestigation = F)
{
  isDT <- ifelse(grepl("data.table", class(Data)[1]),T,F)
  # See if there are any missing value in any column
  #naTable <- sapply(c(1:length(Data)), function(x) length(which(is.na(Data[,x]) == T)))
  
  ifelse(isDT, naTable <- sapply(c(1:length(Data)), function(x) sum(is.na(Data[,x, with = F]))), naTable <- sapply(c(1:length(Data)), function(x) sum(is.na(Data[,x]))))
  
  # Storing class of column and their length (only for factor)
  col_class <- sapply(Data, function(x) class(x))
  #col_class_factor_len <- sapply(Data, function(x) ifelse(is.factor(x),length(levels(x)),0))
  col_class_factor_len <- sapply(Data, function(x) length(unique(x)))
  
  #Make DF with these data
  ifelse(isDT, Df <- data.table(colName = names(Data),  MissingCount = naTable, PercentageMissing = round((naTable/nrow(Data))*100,0), col_class = col_class, UniqueLength = col_class_factor_len), Df <- data.frame(colName = names(Data),  MissingCount = naTable, PercentageMissing = round((naTable/nrow(Data))*100,0), col_class = col_class, UniqueLength = col_class_factor_len))
  
  #How many fields are having missing value
  #Shiv: Chnaged the way missing field count is taken
  #missingFieldValueCount <- nrow(with(data = Df, Df[MissingCount > 0,]))
  missingFieldValueCount <- nrow(Df[MissingCount > 0,])
  
  #Overall what is % of data having at least one missing fields
  percentageMissingFields <- {1 - (nrow(Data[complete.cases(Data),])/nrow(Data)) }* 100
  
  if (bIDandUniqueColInvestigation == T)
  {
    #ID Column - Having each value unique. Applicable for Char, Integer and Factor only. But not necessarily all need to be removed by Analyst
    # Single Unique Value column. Applicable for any kind of column
    col_UniqueValueLength <- apply(Data, 2, function(col) length(unique(na.omit(col))))
    col_SingleUniqueValueColumn <- as.vector(names(which(col_UniqueValueLength == 1)))
    col_IDValueColumn <- as.vector(names(which(col_UniqueValueLength == nrow(Data))))
  } 
  else
  {
    col_SingleUniqueValueColumn <- NULL
    col_IDValueColumn <- NULL
  } # bIDandUniqueColInvestigation
  
  # Row missing % DF
  if (bRowMissingInvestigation == T)
  {
    # See if there are any missing value in any row for incomplete cases
    Data <- Data[!complete.cases(Data),]
    
    if (nrow(Data) != 0)
    {
        naRow <- sapply(c(1:nrow(Data)), function(x) (sum(is.na(Data[x,]) == T)))
      
        #Make DF with these data
        #DfRow <- data.frame(rowNum = c(1:nrow(Data)),  MissingCount = naRow, PercentageMissing = round((naRow/ncol(Data))*100,0))
        # Shiv 16 NOv 2015: Changed rowNum to take actual row numbers
        ifelse(isDT, DfRow <- data.table(rowNum = row.names(Data),  MissingCount = naRow, PercentageMissing = round((naRow/ncol(Data))*100,0)), DfRow <- data.frame(rowNum = row.names(Data),  MissingCount = naRow, PercentageMissing = round((naRow/ncol(Data))*100,0)))
    } else DfRow = NULL
    
  } #bRowMissingInvestigation
  else DfRow = NULL

  missingValue <- list(Df = Df, missingFieldValueCount = missingFieldValueCount, percentageMissingFields = percentageMissingFields, col_SingleUniqueValueColumn = col_SingleUniqueValueColumn, col_IDValueColumn = col_IDValueColumn, DfRow = DfRow)
  
  return(missingValue)
  
} # getMissingValueDetails

# Write CSV file in 5 different files - Sample with all column and for each classes - factor, Date, integer and numeric
WriteSampleCsv <- function(Data, int_NumRowsToSave = 500, bIntNumTogether = F, bSaveRowNames = F, bSaveAllFiles = F)
{
  # Sample rows
  rowsToSave <- sample(nrow(Data), int_NumRowsToSave); 
  
  # Write sample data for all classes columns
  col_factor <- vector(); col_Date <- vector(); col_integer <- vector();   col_numeric <- vector(); col_logical <- vector()
  for (col_num in c(1:ncol(Data)))
  {
    col_class <- class(Data[,col_num])
    ifelse(col_class == "factor", col_factor <- c(col_factor,col_num), ifelse(col_class == "Date",col_Date <- c(col_Date,col_num), ifelse(col_class == "integer",col_integer <- c(col_integer,col_num), ifelse(col_class == "numeric",col_numeric <- c(col_numeric,col_num),ifelse(col_class == "numeric",col_logical <- c(col_logical,col_num),"character")))))
  }
  table(sapply(Data, class))
  #head(Data[rowsToSave,col_factor])
  
  fileName <- paste("Sample_col_factor", int_NumRowsToSave, ".csv", sep = "")
  if(bSaveAllFiles) write.csv(file = fileName, Data[rowsToSave,col_factor], row.names = bSaveRowNames )
  
  fileName <- paste("Sample_col_Date", int_NumRowsToSave, ".csv", sep = "")
  if(bSaveAllFiles) write.csv(file = fileName, Data[rowsToSave,col_Date], row.names = bSaveRowNames )
  
  if (bIntNumTogether == T)
  {
    fileName <- paste("Sample_col_integer_numeric", int_NumRowsToSave, ".csv", sep = "")
    if(bSaveAllFiles) write.csv(file = fileName, Data[rowsToSave,c(col_integer,col_numeric)], row.names = bSaveRowNames )
  }
  else
  {
    fileName <- paste("Sample_col_integer", int_NumRowsToSave, ".csv", sep = "")
    if(bSaveAllFiles) write.csv(file = fileName, Data[rowsToSave,col_integer], row.names = bSaveRowNames )
    
    fileName <- paste("Sample_col_numeric", int_NumRowsToSave, ".csv", sep = "")
    if(bSaveAllFiles) write.csv(file = fileName, Data[rowsToSave,col_numeric], row.names = bSaveRowNames )
  }
  
  fileName <- paste("Sample_col_logical", int_NumRowsToSave, ".csv", sep = "")
  if(bSaveAllFiles) write.csv(file = fileName, Data[rowsToSave,col_logical], row.names = bSaveRowNames )
  
  # Write sample data for all columns
  fileName <- paste("Sample_", int_NumRowsToSave, ".csv", sep = "")
  write.csv(file = fileName, Data[rowsToSave,], row.names = bSaveRowNames )
  
} # WriteSampleCsv

#This function compare actual (observed) with predicted values
compareActualPredict <- function(actual, predFitted, marginErrorPercentage = 10)
{
  df_Actual_Pred <- data.frame( Actual = actual, Pred = predFitted); names(df_Actual_Pred) <- c("Actual", "Pred")
  df_Actual_Pred$DiffPercentage <- abs(((df_Actual_Pred$Actual - df_Actual_Pred$Pred)/df_Actual_Pred$Actual) * 100)
  df_Actual_Pred <- df_Actual_Pred[order(df_Actual_Pred$DiffPercentage, decreasing = T),]
  
  int_NumberMorethanMarginErrorPercentage <- nrow(df_Actual_Pred[df_Actual_Pred$DiffPercentage > marginErrorPercentage,])
  int_TotalPercentageMorethanMarginErrorPercentage <-  int_NumberMorethanMarginErrorPercentage / nrow(df_Actual_Pred) * 100
  
  int_NumberMarginError_25_50_75_100 <- sapply(seq(from = 0, to = 75, by = 25), function(x) nrow(df_Actual_Pred[df_Actual_Pred$DiffPercentage >= x & df_Actual_Pred$DiffPercentage <= (x + 25),]))
  int_NumberMarginError_25_50_75_100[4] <- int_NumberMarginError_25_50_75_100[4] + nrow(df_Actual_Pred) - sum(int_NumberMarginError_25_50_75_100)
  
  int_PercentageMarginError_25_50_75_100 <- (int_NumberMarginError_25_50_75_100 / nrow(df_Actual_Pred)) * 100
  #df_Actual_Pred = df_Actual_Pred,
  compareActualPredict <- list( DiffPercentage = df_Actual_Pred$DiffPercentage, int_NumberMorethanMarginErrorPercentage = int_NumberMorethanMarginErrorPercentage, int_TotalPercentageMorethanMarginErrorPercentage = int_TotalPercentageMorethanMarginErrorPercentage, int_NumberMarginError_25_50_75_100 = int_NumberMarginError_25_50_75_100, int_PercentageMarginError_25_50_75_100 = int_PercentageMarginError_25_50_75_100)
  
  return(compareActualPredict)
} # function

GetErrorAndConfusionMatrixForMultiFactorPrediction <- function(df_test, lev_response, str_Response, pred)
{
  #df <- data.frame(Seq = c(1: length(levels(df_test[,fac_response]))), Levels = levels(df_test[,fac_response]));
  df <- data.frame(Seq = c(1: length(lev_response)), Levels = lev_response);
  
  if (class(pred) != "matrix" & class(pred) != "data.frame") {
    saveData <- matrix(data = pred, nrow = nrow(df_test), ncol = 1 + length(lev_response), byrow = T)
    dim(saveData); saveData <- saveData[,-1]
  } else  saveData <- pred
  
    
    maxCol <- sapply(c(1:nrow(saveData)), function(x) which.max(saveData[x, ]))
    
    saveData <- as.data.frame(saveData)
    saveData$maxCol <- maxCol;
    saveData$Response <- sapply(c(1:nrow(saveData)), function(x) df[df$Seq == saveData[x,"maxCol"],"Levels"])
    saveData$Response <- as.factor(saveData$Response)
  
    # Calculate return values
    if( length(which(names(df_test) == str_Response)) > 0) {
      
      cm <- confusionMatrix(df_test[,str_Response], saveData$Response)
      err_mean <- 1 - mean(df_test[,str_Response] == saveData$Response)
    
  } else { cm <- NULL; err_mean <- NULL; Response <- NULL}
  
  # Save Response and Clean remaining
  Response <- saveData$Response; saveData$Response <- NULL;   saveData$maxCol <- NULL;
    
  err_list <- list(cm = cm, err_mean = err_mean, predMat = saveData, Response = Response)
  
  return(err_list)
  
} # GetErrorAndConfusionMatrixForMultiFactorPrediction

# This provides descriptive Analytics for single numeric features
#listNumericFeatures = NA; folderImageDescriptive = g_folderImageDescriptive; str_prefix = NA
Desc_Numeric_Single <- function(train, listNumericFeatures = NA, strResponse = NA,bins = 30, folderImageDescriptive = g_folderImageDescriptive, str_prefix = NA, skip_if_present = T)
{      
  library(ggplot2); library(ggrepel)
  if(is.na(listNumericFeatures)) listNumericFeatures <- GetNumericColumns(train)
  if(is.na(str_prefix)) str_prefix <- "singlenum"
  
  for(numericColumn in listNumericFeatures)  # numericColumn = listNumericFeatures[2]
  { 
    #par(mfrow = c(2,2));
    fileName <- paste0(folderImageDescriptive, str_prefix, '_', numericColumn, ".png")
    if(file.exists(fileName) & skip_if_present) {print(paste0("File exist and hence skiping - ", fileName)); next;}
    
    png(fileName)
    
    gg0  <- ggplot(data = train, aes(x = eval(as.name(numericColumn))))
    gg0 <- gg0 + geom_histogram(col="black", fill="lightblue", alpha = .2) # , stat="count" breaks=seq(mi, mx, by = step), binwidth  = bins,
    gg0 <- gg0 + geom_vline(aes(xintercept=mean(eval(as.name(numericColumn)), na.rm=T)), color="green", linetype="dashed", size=1)
    gg0 <- gg0 + labs(title= paste0("Distribution of " , numericColumn), size = 10) + labs(x=numericColumn, y="Count", size = 10)
    
    gg1  <- ggplot(data = train, aes(x = eval(as.name(numericColumn))))
    gg1 <- gg1 + geom_histogram(aes(y =..density..),  col="black", fill="lightblue", alpha = .2) # , stat="count" breaks=seq(mi, mx, by = step), binwidth  = bins,
    gg1 <- gg1 + geom_density(col="red") # , fill="lightblue", alpha = .2
    gg1 <- gg1 + geom_vline(aes(xintercept=mean(eval(as.name(numericColumn)), na.rm=T)), color="green", linetype="dashed", size=1)
    gg1 <- gg1 + labs(title=paste0(paste0(" (Skew:" , round(e1071::skewness(train[[numericColumn]], na.rm = T), 2) , ")"),", Density and Distribution of " , numericColumn), size = 10) + labs(x=numericColumn, y="Count/Density", size = 10)    
    gg1 <- gg1 + theme(axis.text.y=element_blank()) # axis.title.x=element_blank(),,      axis.ticks.x=element_blank())
    #plot(gg1); dev.off()
    
    gg2  <- ggplot(data = train, aes(x = numericColumn, y = eval(as.name(numericColumn))))
    gg2 <- gg2 + geom_boxplot(fill = "lightblue", outlier.colour = "red", outlier.shape = 1, varwidth = TRUE)
    gg2 <- gg2 + labs(title=paste0("Spread of " , numericColumn), size = 10) + labs(x=numericColumn, y= paste0("Value of ", numericColumn), size = 10)
    #plot(gg2)
    
    if(!is.na(strResponse))  #  strResponse = listNumericFeatures[2]
    {
      if(is.factor(train[[strResponse]]) | is.logical(train[[strResponse]]))
      {
        df <- GetGroup_Cat_Num(train, strResponse, numericColumn, nSelectTopN = 10)
        
        gg3  <- ggplot(data = df, aes(x = eval(as.name(strResponse)), y = eval(as.name(numericColumn))))
        gg3 <- gg3 + geom_bar(fill = "lightblue", stat = "identity")
        gg3 <- gg3 + labs(title=paste0("Average of ", numericColumn , " vs " , strResponse , " (RV)"), size = 10) + labs(x=strResponse, y= paste0("Average of ", numericColumn), size = 10)
        gg3 <- gg3 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
        gg3 <- gg3 + geom_text_repel(data = df, aes(y = eval(as.name(numericColumn)), label = eval(as.name(numericColumn))))
        #rm(df)
      }
      else if(is.numeric(train[[strResponse]]) | is.integer(train[[strResponse]]))
      {
        gg3  <- ggplot(data = train, aes(x = eval(as.name(numericColumn)), y = eval(as.name(strResponse))))
        gg3 <- gg3 + geom_point(col = "lightblue", size = 2)#  bar(, stat = "identity")
        gg3 <- gg3 + geom_smooth()
        gg3 <- gg3 + labs(title=paste0(strResponse , " (RV)", " vs " , numericColumn), size = 10) + labs(y=strResponse, x= numericColumn, size = 10)
        #plot(gg3)
      }    
      else { gg3 <- NULL; print("Not implemented")}
      
      # if(!is.na(strResponse))
    } else gg3 <- NULL
    
    suppressPackageStartupMessages(multiplot(gg1, gg2, gg0, gg3 , cols=2))
    #gridExtra::grid.arrange(gg1, gg2, gg3)          , main = paste0('Distribution of ', numericColumn, " (Skew:" , round(e1071::skewness(train[[numericColumn]]), 2) , ")"))
    #mtext("My place", side = 3, line = -21, outer = TRUE)
    #plot(gg1); #plot(gg2); plot(gg3); 
    dev.off()
    rm(gg0, gg1, gg2, gg3)
  } # for
  
  detach(package:ggrepel); detach(package:ggplot2)
  return(T)  # Desc_Numeric_Single
} # Desc_Numeric_Single

# This provides descriptive Analytics for two numeric features
Desc_Numeric_Double <- function (train, listNumericFeatures1 = NA, listNumericFeatures2 = NA, strResponse = None, filePdf = "Desc_Numeric_Double.pdf", folderImageDescriptive = g_folderImageDescriptive)
{
  library(ggplot2); library(ggrepel)
  
  if(is.na(listNumericFeatures1)) listNumericFeatures1 <- GetNumericColumns(train)
  if(is.na(listNumericFeatures2)) listNumericFeatures2 <- GetNumericColumns(train)
  
  listNumericFeatures1 <- intersect(listNumericFeatures1, listNumericFeatures2)
  
  for(i in seq_len(length(listNumericFeatures1) - 1)) # i = 1
  {   
    numericColumn1 = listNumericFeatures1[i]
    
    for(j in seq(i+1, length(listNumericFeatures1))) # j = 6
    {
      numericColumn2 = listNumericFeatures1[j]
      
      #par(mfrow = c(2,2));
      fileName <- paste0(folderImageDescriptive, "doublenum_", numericColumn1, "_with_", numericColumn2, ".png")
      if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName)); next;}
      
      png(fileName)
      
      
      # multiplot does lazy calculation and hence creating separate object
      df1 <- df2 <- df3 <- data.table()
      
      
      gg1  <- ggplot(data = train, aes(x = eval(as.name(numericColumn1)), y = eval(as.name(numericColumn2))))
      gg1 <- gg1 + geom_point(col = "lightblue", size = 2)
      gg1 <- gg1 + geom_smooth()
      gg1 <- gg1 + labs(title=paste0("Corr: ",round(cor(train[,c(numericColumn1, numericColumn2), with = F])[1,2], 2), ",", numericColumn2 , " vs " , numericColumn1), size = 10) + labs(x=numericColumn1, y= numericColumn2, size = 10)
      
      df1 <- train[, c(numericColumn1, numericColumn2), with = F]
      df1 <- suppressWarnings(melt(df1))
      gg2  <- ggplot(data = df1, aes(x = variable, y = value))
      gg2 <- gg2 + geom_boxplot(fill = "lightblue", outlier.colour = "red", outlier.shape = 1, varwidth = TRUE)
      gg2 <- gg2 + labs(title=paste0("Spread of " , numericColumn1, " and ", numericColumn2), size = 10) + labs(x ="", y= "Values", size = 10)
      #rm(df)
      
      if(!is.na(strResponse))  #  strResponse = listNumericFeatures[2]
      {
        if(is.factor(train[[strResponse]]) | is.logical(train[[strResponse]]))
        {
          df2 <- GetGroup_Cat_Num(train, strResponse,numericColumn1, nSelectTopN = 10)
          gg3  <- ggplot(data = df2, aes(x = eval(as.name(strResponse)), y = eval(as.name(numericColumn1))))
          gg3 <- gg3 + geom_bar(fill = "lightblue", stat = "identity")
          gg3 <- gg3 + labs(title=paste0("Average of ", numericColumn1 , " vs " , strResponse , " (RV)"), size = 10) + labs(x=strResponse, y= paste0("Average of ", numericColumn1), size = 10)
          gg3 <- gg3 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
          gg3 <- gg3 + geom_text_repel(data = df2, aes(y = eval(as.name(numericColumn1)), label = eval(as.name(numericColumn1))))
          #rm(df)
          
          df3 <- GetGroup_Cat_Num(train, strResponse,numericColumn2, nSelectTopN = 10)
          gg4  <- ggplot(data = df3, aes(x = eval(as.name(strResponse)), y = eval(as.name(numericColumn2))))
          gg4 <- gg4 + geom_bar(fill = "lightblue", stat = "identity")
          gg4 <- gg4 + labs(title=paste0("Average of ", numericColumn2 , " vs " , strResponse , " (RV)"), size = 10) + labs(x=strResponse, y= paste0("Average of ", numericColumn2), size = 10)
          gg4 <- gg4 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
          gg4 <- gg4 + geom_text_repel(data = df3, aes(y = eval(as.name(numericColumn2)), label = eval(as.name(numericColumn2))))
          #rm(df)
        }
        else if(is.numeric(train[[strResponse]]) | is.integer(train[[strResponse]]))
        {
          gg3  <- ggplot(data = train, aes(x = eval(as.name(strResponse)), y = eval(as.name(numericColumn1))))
          gg3 <- gg3 + geom_point(col = "lightblue", size = 2)#  bar(, stat = "identity")
          gg3 <- gg3 + geom_smooth()
          gg3 <- gg3 + labs(title=paste0(numericColumn1 , " vs " , strResponse , " (RV)"), size = 10) + labs(x=strResponse, y= numericColumn1, size = 10)
          
          gg4  <- ggplot(data = train, aes(x = eval(as.name(strResponse)), y = eval(as.name(numericColumn2))))
          gg4 <- gg4 + geom_point(col = "lightblue", size = 2)#  bar(, stat = "identity")
          gg4 <- gg4 + geom_smooth()
          gg4 <- gg4 + labs(title=paste0(numericColumn2 , " vs " , strResponse , " (RV)"), size = 10) + labs(x=strResponse, y= numericColumn2, size = 10)
        }    
        else { gg3 <- NULL; gg4 <- NULL; print("Not implemented")}
        
      } # if(!is.na(strResponse))
      else { gg3 <- NULL; gg4 <- NULL;}
      
      suppressPackageStartupMessages(multiplot(gg1, gg3, gg2, gg4 , cols=2))
      #plot(gg1); plot(gg2); plot(gg3); plot(gg4);
      
      dev.off()
      rm(gg1, gg2, gg3, gg4, df1, df2, df3)
    } # J
  } # for(i in seq_len(length(listNumericFeatures1))) # i = 1
  
  
  detach(package:ggrepel); detach(package:ggplot2)
  return(T)  
} # # Desc_Numeric_Double

# This provides descriptive Analytics for single Categorical features
# listCategoricalFeatures = NA; strResponse = NA; nMaxFacVArOnXaxis = 10; filePdf = "Desc_Categorical_Double.pdf"; folderImageDescriptive = "images/descriptive/"
Desc_Categorical_Single   <- function(train, listCategoricalFeatures = NA, strResponse = NA, nMaxFacVArOnXaxis = 10, filePdf = "Desc_Categorical_Double.pdf", folderImageDescriptive = g_folderImageDescriptive)
{
  library(ggplot2); library(ggrepel)
  
  if(is.na(listCategoricalFeatures)) listCategoricalFeatures <- GetFactorLogicalColumns(train)
  
  for(col_cluster_factor in listCategoricalFeatures) # col_cluster_factor = listCategoricalFeatures[2]
  {   
    #par(mfrow = c(1,2));
    fileName <- paste0(folderImageDescriptive, "singlefac_", col_cluster_factor, ".png")
    if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName)); next;}
    
    png(fileName)
    
    #Get table of both factors and remove row sum and col sums
    dt <- GetCountAndPercentage(train, col_cluster_factor)
    dt[,COUNT_PERCENTAGE := NULL]; setorderv(dt, "COUNT", order = -1)
    
    # See if thresold crosses
    if(nMaxFacVArOnXaxis < nrow(dt)) dt <- dt[c(1:nMaxFacVArOnXaxis)]
    
    # draw the graph
    gg1 <- ggplot(data = dt, aes(x = reorder(get(col_cluster_factor), -COUNT), y = COUNT))
    gg1 <- gg1 + geom_bar(stat = "identity", fill = "lightblue") # 
    gg1 <- gg1 + labs(x = col_cluster_factor, y = "Count", title = paste0("Count of ", col_cluster_factor, " (top ",nMaxFacVArOnXaxis,")"))
    gg1 <- gg1 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
    gg1 <- gg1 + geom_text_repel(data = dt, aes(y = COUNT, label = COUNT))
    
    if(!is.na(strResponse))  #  strResponse = listNumericFeatures[2]
    {
      if(is.factor(train[[strResponse]]) | is.logical(train[[strResponse]]))
      {
        #Get table of both factors and remove row sum and col sums
        dt <- GetCountAndPercentage_TwoFactors(train, col_cluster_factor, strResponse)
        dt[,row_Sums := NULL]; dt <- dt[!(get(col_cluster_factor) %in% "col_Sums")]
        
        # See if thresold crosses
        if(nMaxFacVArOnXaxis < nrow(dt)) dt <- dt[c(1:nMaxFacVArOnXaxis)]
        if((nMaxFacVArOnXaxis +1) < ncol(dt)) dt <- dt[, c(1:(nMaxFacVArOnXaxis +1)), with = F]
        
        # Stote the title text
        title <- paste0(col_cluster_factor, " vs ", strResponse, " (top ", nrow(dt), ")")
        
        # modify the data in long format
        dt <- melt(dt, id.vars = c(col_cluster_factor)); dt$variable <- as.factor(dt$variable)
        setorderv(dt, c(col_cluster_factor, "variable"))
        
        # Reformat the data to make huge data readable
        tempValueVar <- names(dt)[3]
        listDivisorAndSuffix <- getDivisorAndSuffix(mean(dt[[tempValueVar]]))
        if(listDivisorAndSuffix$Divisor > 1) {
          dt[, (tempValueVar) := round(get(tempValueVar) / listDivisorAndSuffix$Divisor, getRoundCuttoff(get(tempValueVar) / listDivisorAndSuffix$Divisor))] 
          tempValueVar <- names(dt)[3] <- paste0(tempValueVar, "_", listDivisorAndSuffix$Suffix)
        }
        
        #Get the cummulative position for Text
        dt[, TEMP_CUMSUM := cumsum(get(tempValueVar)), by = col_cluster_factor]
        
        # draw the graph
        gg2 <- ggplot(data = dt, aes(x = get(col_cluster_factor), y = eval(as.name(tempValueVar))))
        gg2 <- gg2 + geom_bar(data = dt, aes(fill = variable), position = "stack", stat = "identity") # 
        gg2 <- gg2 + geom_text_repel(data = dt, aes(y = TEMP_CUMSUM, label = get(tempValueVar))) # , alpha = 0.5 , position = "jitter", jitter.width = 1.3, jitter.height = 0.1
        gg2 <- gg2 + labs(x = col_cluster_factor, y = strResponse, title = title)
        gg2 <- gg2 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
        gg2 <- gg2 + geom_text_repel(data = dt, aes(y = eval(as.name(tempValueVar)), label = eval(as.name(tempValueVar))))
      }
      else if(is.numeric(train[[strResponse]]) | is.integer(train[[strResponse]])) # strResponse = "Sepal.Length"
      { 
        dt <- GetGroup_Cat_Num(train, col_cluster_factor, strResponse, nSelectTopN = 10)
        
        gg2  <- ggplot(data = dt, aes(x = eval(as.name(col_cluster_factor)), y = eval(as.name(strResponse))))
        gg2 <- gg2 + geom_bar(fill = "lightblue", stat = "identity")
        gg2 <- gg2 + labs(title=paste0("Average of ", strResponse , "(RV) vs " , col_cluster_factor), size = 10) + labs(x=col_cluster_factor, y= paste0("Average of ", strResponse), size = 10)
        gg2 <- gg2 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
        gg2 <- gg2 + geom_text_repel(data = dt, aes(y = eval(as.name(strResponse)), label = eval(as.name(strResponse))))
      }    
      else { gg2 <- NULL; print("Not implemented")}
      
    } # if(!is.na(strResponse))
    else gg2 <- NULL
    
    suppressPackageStartupMessages(multiplot(gg1, gg2 , cols=1))
    #plot(gg1); plot(gg2);
    dev.off()
    rm(gg1, gg2, dt)
  } # for
  
  detach(package:ggrepel); detach(package:ggplot2);
  return(T)
} # Desc_Categorical_Single  

# This provides descriptive Analytics for double categorical features
# listCategoricalFeatures1 = NA; listCategoricalFeatures2 = NA; nMaxFacVArOnXaxis = 10; filePdf = "Desc_Categorical_Double.pdf"; folderImageDescriptive = "images/descriptive/"
Desc_Categorical_Double  <- function(train, listCategoricalFeatures1 = NA, listCategoricalFeatures2 = NA, nMaxFacVArOnXaxis = 10, filePdf = "Desc_Categorical_Double.pdf", folderImageDescriptive = g_folderImageDescriptive)
{
  library(ggplot2); library(ggrepel)
  
  if(is.na(listCategoricalFeatures1)) listCategoricalFeatures1 <- GetFactorLogicalColumns(train)
  if(is.na(listCategoricalFeatures2)) listCategoricalFeatures2 <- GetFactorLogicalColumns(train)
  
  listCategoricalFeatures1 <- intersect(listCategoricalFeatures1, listCategoricalFeatures2)
  
  for(i in seq_len(length(listCategoricalFeatures1) - 1)) # i = 1
  {   
    col_cluster_factor = listCategoricalFeatures1[i]
    
    for(j in seq(i+1, length(listCategoricalFeatures1))) # j = 6
    {
      col_second_cluster_factor = listCategoricalFeatures1[j]
      
      fileName <- paste0(folderImageDescriptive, "doublefac_", col_second_cluster_factor, "_vs_", col_cluster_factor, ".png")
      if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName)); next;}
      
      png(fileName)
      #Get table of both factors and remove row sum and col sums
      dt <- GetCountAndPercentage_TwoFactors(train, col_cluster_factor, col_second_cluster_factor)
      dt[,row_Sums := NULL]; dt <- dt[!(get(col_cluster_factor) %in% "col_Sums")]
      
      # See if thresold crosses
      if(nMaxFacVArOnXaxis < nrow(dt)) dt <- dt[c(1:nMaxFacVArOnXaxis)]
      if((nMaxFacVArOnXaxis +1) < ncol(dt)) dt <- dt[, c(1:(nMaxFacVArOnXaxis +1)), with = F]
      
      # Stote the title text
      title <- paste0(col_cluster_factor, " vs ", col_second_cluster_factor, " (top ", nrow(dt), ")")
      
      # modify the data in long format
      dt <- melt(dt, id.vars = c(col_cluster_factor)); dt$variable <- as.factor(dt$variable)
      setorderv(dt, c(col_cluster_factor, "variable"))
      
      # Reformat the data to make huge data readable
      tempValueVar <- names(dt)[3]
      listDivisorAndSuffix <- getDivisorAndSuffix(mean(dt[[tempValueVar]]))
      if(listDivisorAndSuffix$Divisor > 1) {
        dt[, (tempValueVar) := round(get(tempValueVar) / listDivisorAndSuffix$Divisor, getRoundCuttoff(get(tempValueVar) / listDivisorAndSuffix$Divisor))]
        tempValueVar <- names(dt)[3] <- paste0(tempValueVar, "_", listDivisorAndSuffix$Suffix)
      }
      
      #Get the cummulative position for Text
      dt[, TEMP_CUMSUM := rev(cumsum(rev(get(tempValueVar)))), by = col_cluster_factor]
      #dt[, (col_cluster_factor) := as.factor(eval(as.name(col_cluster_factor)))]
      #str(dt)
      
      # draw the graph
      gg <- ggplot(data = dt, aes(x = get(col_cluster_factor), y = eval(as.name(tempValueVar))))
      gg <- gg + geom_bar(data = dt, aes(fill = variable), position = "stack", stat = "identity") #
      gg <- gg + geom_text_repel(data = dt, aes(y = TEMP_CUMSUM, label = get(tempValueVar))) # , alpha = 0.5 , position = "jitter", jitter.width = 1.3, jitter.height = 0.1
      gg <- gg + labs(x = col_cluster_factor, y = paste0(col_second_cluster_factor," (",listDivisorAndSuffix$Suffix,")"), title = title)
      gg <- gg + theme(axis.text.x = element_text(size = rel(1), angle = 45))
      plot(gg)
      #class(col_cluster_factor)
      #class()
      dev.off()
      rm(dt); gc()
    } # for for j
  } # for for i
  
  detach(package:ggrepel); detach(package:ggplot2);
  return(T)
} # Desc_Categorical_Double 

# This provides descriptive Analytics for all numeric features at once
#listNumericFeatures = NA; corThreshold = 0.5; filePdf = "Desc_Numeric_AllatOnce.pdf"; folderImageDescriptive = g_folderImageDescriptive; folderOutput=folderImageDescriptive
Desc_Numeric_AllatOnce <- function(train, listNumericFeatures = NA, strResponse = NA, corThreshold = 0.5, filePdf = "Desc_Numeric_AllatOnce.pdf", folderImageDescriptive = g_folderImageDescriptive, folderOutput=folderImageDescriptive )
{      
  library(ggplot2)
  if(is.na(listNumericFeatures)) listNumericFeatures <- GetNumericColumns(train)
  #if(!is.na(strResponse)) listNumericFeatures <- setdiff(listNumericFeatures, strResponse)
  # Generate paramerter for bucketting. Order first so that all big mean data comes first
  listMeans <- sapply(train[,listNumericFeatures, with=F], mean, na.rm = T)
  #km <- kmeans(listMeans,4)
  listNumericFeatures <- listNumericFeatures[order(listMeans, decreasing = T)]
  
  nTotalFeaturesCount = length(listNumericFeatures); bucketLength = min(10, nTotalFeaturesCount); from <- 1
  
  for(rowNum in seq(from, nTotalFeaturesCount, bucketLength))
  {
    to <- from + bucketLength  -1; if(to > nTotalFeaturesCount) to <- nTotalFeaturesCount;
    
    # start Process
    #print(paste("Range:", from, "-", to, collapse = ""))
    
    fileName <- paste0(folderImageDescriptive, "allatoncenum_", from, "-", to, ".png")
    if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName)); next;}
    
    png(fileName)   
    df1 <- suppressWarnings(melt(train[, listNumericFeatures[c(from:to)], with = F]))
    gg2  <- ggplot(data = df1, aes(x = variable, y = value))
    gg2 <- gg2 + geom_boxplot(fill = "lightblue", outlier.colour = "red", outlier.shape = 1, varwidth = TRUE)
    gg2 <- gg2 + labs(title=paste0("Spread in range " , from, " - ", to), size = 10) + labs(x ="", y= "Values", size = 10)
    gg2 <- gg2 + theme(axis.text.x = element_text(size = rel(1), angle = 45))
    plot(gg2)
    # end Process
    dev.off()
    
    if(to == nTotalFeaturesCount) break;
    from <- to + 1
  } # for rowNum
  
  detach(package:ggplot2)
  
  # Bring strResponse in front
  if(!is.na(strResponse)) listNumericFeatures <- c(strResponse, setdiff(listNumericFeatures, strResponse)) 
  
  # Correlation amoung numeric features
  fileName <- paste0(folderImageDescriptive, "AllAtOnceNum_Correlation.png")
  if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName));}
  else
  {   
    df <- round(cor(train[, listNumericFeatures, with = F]),2)
    #corrgram is throwing error for NA
    df[is.na(df)] <- 0
    library(corrgram)
    png(paste0(folderImageDescriptive, "AllAtOnceNum__Correlation.png"))
    corrgram(df, order = TRUE, lower.panel = panel.shade, upper.panel = panel.pie, text.panel = panel.txt, main = "AllAtOnceNum Correlation")
    dev.off()
    detach(package:corrgram)
    
    # Save the data
    df <- data.table(Features = row.names(df), df)
    print(paste0("Correlation file is saved to " , folderOutput ,"allatanceaum_correlation.csv"))
    fwrite(df, paste0(folderOutput , "AllAtOnceNum_Correlation.csv"))
    
    # Strong correlation
    # Set the threshold to select only highly correlated attributes
    suppressWarnings(df <- melt(df, id.vars = "Features"))
    df <- df[Features != variable & value != 1.0 & abs(value) > corThreshold]
    
    # Now remove duplicate entry like y vs x and x vs y
    df$IsDuplicate <- F
    for(rowNum in seq(1, nrow(df)-1, 1)) # rowNum = 2
    {
      dupRow <- which(df[(rowNum + 1): nrow(df)][["Features"]] == df[rowNum, variable] & df[(rowNum + 1): nrow(df)][["variable"]] == df[rowNum, Features])
      if(length(dupRow) > 0) df[dupRow + rowNum, IsDuplicate := T]
    } # for
    
    df <- df[IsDuplicate == F]; df[, IsDuplicate := NULL]
    
    print(paste0("Strong Correlation file is saved to " , folderOutput ,"AllAtOnceNum_Correlation_Strong.csv"))
    fwrite(df, paste0(folderOutput , "allatoncenum_correlation_strong.csv"))
  } # if(file.exists("AllAtOnceNum__Correlation.png"))
  
  #Skewness of features
  fileName <- paste0(folderImageDescriptive, "allstsncesum_skew.csv")
  if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName));}
  else
  {
    df <-  round(sapply(train[, listNumericFeatures, with = F], e1071::skewness, na.rm = T), 2)
    df <- data.table(Features = names(df), Skew = df)
    setorderv(df, "Skew", order = -1)
    print(paste0("Skew file is saved to " , folderOutput ,"allatoncenum_skew.csv"))
    fwrite(df, paste0(folderOutput , "allstsncesum_skew.csv"))
  } #  if(file.exists("allstsncesum_skew.csv"))
  
  return()  
} # Desc_Numeric_AllatOnce

# Smoothing with various transformation
desc_smoothing_with_transformation <- function(train, listNumericFeatures = g_listNumericFeatures, folderImageDescriptive = g_folderImageDescriptive, bLog = T, bDivision_by_one = T,bScale = T)
{
  listNumericFeatures <- intersect(names(train), listNumericFeatures); train[, listNumericFeatures] <- lapply(train[, listNumericFeatures, with = F], as.numeric)
  
  for(feature in listNumericFeatures) # feature = listNumericFeatures[1]
  {
    fileName <- paste0(folderImageDescriptive,feature,".png")
    if(file.exists(fileName)) {print(paste0("File exist and hence skiping - ", fileName)); next;}
    
    png(fileName)
    
    par(mfrow = c(2,2));
    hist(train[[feature]]);  
    if(bLog) hist(log(train[[feature]]))
    if(bDivision_by_one) hist(1/train[[feature]])
    if(bScale) hist(scale(train[[feature]]))
    dev.off()
  }# for(feature in listNumericFeatures)
  
  return()
} # desc_smoothing_with_transformation

# Export the graph to file. Input Parameter is DT, Cluster id or NA, file to save, array, plot_type = c("num_single", "int_single", "num_int_multiple", "fac_single", "fac_double")
# train = train; clusterID = NA; filename = "Descriptive.pdf", plot_type = plot_type, fac_double_fields = NA, num_int_multiple_fields = NA
exportToFile<- function(train, clusterID, filename, plot_type = NA, fac_double_fields = NA, num_int_multiple_fields = NA, strGraphFOlder = "./graphs/", strSuffix = "", nMaxFacVArOnXaxis = 10)
{
  library(ggplot2); library(ggrepel)
  pdf(file = filename)
  
  if(is.na(clusterID) == F) {
    
    print(paste("Image for Cluster started"))
    
    count_cluster <- max(clusterID)  
    for (col_cluster_factor in colnames(train)) #  col_cluster_factor <- c("gb") # "TranMonth", "deal_category",
    {
      if (is.factor(train[[col_cluster_factor]]) == T) {
        plot(table(train[[col_cluster_factor]],clusterID), col = c(1:count_cluster), main = paste(col_cluster_factor, " with Cluster", sep = ""), xlab = col_cluster_factor, ylab = "Cluster"); #c("red", "green", "blue")
      }
      else {
        plot(train[[col_cluster_factor]], col = c(1:count_cluster), main = paste(col_cluster_factor, " with Cluster", sep = ""), xlab = "", ylab = col_cluster_factor);
      }
      #points(clusterID, col = c(1 : count_cluster), pch = 8) # Good
      legend("topleft", legend = c(1:count_cluster), col = c(1:count_cluster), lty = 1, lwd = 1, bty = "n", xjust = 0, yjust = 1) # , ncol = count_cluster
    } # for
  } #  is.na(clusterID)
  
  # plot the different types
  if(is.na(plot_type[1]) == F) {
    #plot_type <- c("num_single", "int_single", "fac_single", "fac_double");   "num_single" %in% plot_type
    for (col_cluster_factor in colnames(train)) #  col_cluster_factor <- c("gender") # "gb" "TranMonth", "deal_category",
    {
      print(paste("Image for ", col_cluster_factor))
      
      if (is.factor(train[[col_cluster_factor]]) == T | is.logical(train[[col_cluster_factor]]) == T) {
        
        ifelse(is.logical(train[[col_cluster_factor]]), fac_levels <- c("T","F"), fac_levels <- levels(train[[col_cluster_factor]]));
        fac_levels_len <- length(fac_levels)
        
        # Shiv 06-Aug-2016: Chnaged logic to take top 10 factors
        dt <- GetCountAndPercentage(train = train, colName = col_cluster_factor, bConvertToLower = T)
        
        if(nMaxFacVArOnXaxis < fac_levels_len)
        {
          dt <- dt[c(1:nMaxFacVArOnXaxis), c(1,2), with = F]
          fac_levels <- dt[[col_cluster_factor]]; fac_levels_len <- nMaxFacVArOnXaxis
        } # if(nMaxFacVArOnXaxis < fac_levels_len)
        
        #dt[,1] <- as.factor(dt[,1, with = F])
        
        if( "fac_single" %in% plot_type) {
          #plot(table(train[[col_cluster_factor]]), col = c(1:fac_levels_len), main = col_cluster_factor, xlab = col_cluster_factor, ylab = "");
          
          # Reformat the data to make huge data readable
          tempValueVar <- names(dt)[2]
          listDivisorAndSuffix <- getDivisorAndSuffix(mean(dt[[tempValueVar]]))
          if(listDivisorAndSuffix$Divisor > 1) {
            dt[, (tempValueVar) := round(get(tempValueVar) / listDivisorAndSuffix$Divisor, getRoundCuttoff(get(tempValueVar) / listDivisorAndSuffix$Divisor))] 
            tempValueVar <- names(dt)[2] <- paste0(tempValueVar, "_", listDivisorAndSuffix$Suffix)
          }
          
          title <- paste0(col_cluster_factor, " (top ", nrow(dt), ")")
          gg <- ggplot(data = dt, aes(x = get(col_cluster_factor), y = get(tempValueVar)))
          gg <- gg + geom_bar(stat = "identity", fill = c(1:fac_levels_len))
          gg <- gg + geom_text_repel(aes(label = get(tempValueVar))) # , alpha = 0.9 , position = "jitter"
          gg <- gg + labs(x = col_cluster_factor, y = "Count", title = title)
          gg <- gg + theme(legend.position = "bottom",legend.title = element_blank(), axis.text.x = element_text(size = rel(1), angle = 45))
          plot(gg)    
          
        }  
        
        #legend("topright", legend = c(fac_levels), col = c(1:fac_levels_len), lty = 1, lwd = 1, bty = "n", xjust = 0, yjust = 1) # , ncol = count_cluster
        
        #plot(table(train[[col_cluster_factor]],clusterID), col = c(1:count_cluster), main = paste(col_cluster_factor, " with Cluster", sep = ""), xlab = "", ylab = col_cluster_factor); #c("red", "green", "blue")
        
        dt <- NULL
      }
      else if (is.numeric(train[[col_cluster_factor]]) == T | is.integer(train[[col_cluster_factor]]) == T) {
        boxplot(train[[col_cluster_factor]], main = col_cluster_factor, xlab = "", ylab = col_cluster_factor, staplewex = 1, horizontal = T);
        #X <- train$UserTransCount;       boxplot(train$UserTransCount, horizontal = T); text(x = boxplot.stats(X)$stats, labels = boxplot.stats(X)$stats, y = 1.25)
        # SHiv: 10 May 2016. Addded round
        text(x = round(boxplot.stats(train[[col_cluster_factor]])$stats,0), labels = round(boxplot.stats(train[[col_cluster_factor]])$stats,0), y = 1.25)
        
      }
      
    } # for col_cluster_factor
    
  } # is.na(plot_type)
  
  dev.off()
  
  # plot the different types
  if(is.na(plot_type[1]) == F & ("fac_double" %in% plot_type)) {
    PlotTwoFactorFeatures(train,fac_double_fields, strGraphFOlder, strSuffix, nMaxFacVArOnXaxis)
    
    # for (col_cluster_factor in fac_double_fields) # col_cluster_factor = "label_id"
    # {
    #   if (is.factor(train[[col_cluster_factor]]) == T | is.logical(train[[col_cluster_factor]]) == T) {
    #     
    #     ifelse(is.logical(train[[col_cluster_factor]]), fac_levels <- c("T","F"), fac_levels <- levels(train[[col_cluster_factor]]));
    #     fac_levels_len <- length(fac_levels)
    #     
    #     file <- paste(strGraphFOlder, col_cluster_factor, strSuffix, ".pdf", sep = ""); pdf(file = file) # col_cluster_factor <- "x"
    #     print(paste("2. Image for file ", file))
    #     
    #     for (col_second_cluster_factor in fac_double_fields) # col_second_cluster_factor <- "app_id"
    #     {
    #       if ((is.factor(train[[col_second_cluster_factor]]) | is.logical(train[[col_second_cluster_factor]])) & col_second_cluster_factor != col_cluster_factor)
    #       {
    #         print(paste("2. Image for file ", file, "w.r.t ", col_second_cluster_factor))
    #         
    #         # Shiv 06-Aug-2016: Chnaged logic to take top 10 factors
    #         #plot(table(train[[col_cluster_factor]], train[[col_second_cluster_factor]]), col = c(1:fac_levels_len), main = col_cluster_factor, xlab = col_cluster_factor, ylab = col_second_cluster_factor);                  
    #         
    #         #Get table of both factors and remove row sum and col sums
    #         dt <- GetCountAndPercentage_TwoFactors(train, col_cluster_factor, col_second_cluster_factor)
    #         dt[,row_Sums := NULL]; dt <- dt[!(get(col_cluster_factor) %in% "col_Sums")]
    #         
    #         # See if thresold crosses
    #         if(nMaxFacVArOnXaxis < nrow(dt)) dt <- dt[c(1:nMaxFacVArOnXaxis)]
    #         if((nMaxFacVArOnXaxis +1) < ncol(dt)) dt <- dt[, c(1:(nMaxFacVArOnXaxis +1)), with = F]
    #         
    #         # Stote the title text
    #         title <- paste0(col_cluster_factor, " vs ", col_second_cluster_factor, " (top ", nrow(dt), ")")
    #         
    #         # modify the data in long format
    #         dt <- melt(dt, id.vars = c(col_cluster_factor)); dt$variable <- as.factor(dt$variable)
    #         setorderv(dt, c(col_cluster_factor, "variable"))
    #         
    #         # Reformat the data to make huge data readable
    #         tempValueVar <- names(dt)[3]
    #         listDivisorAndSuffix <- getDivisorAndSuffix(mean(dt[[tempValueVar]]))
    #         if(listDivisorAndSuffix$Divisor > 1) {
    #           dt[, (tempValueVar) := round(get(tempValueVar) / listDivisorAndSuffix$Divisor, getRoundCuttoff(get(tempValueVar) / listDivisorAndSuffix$Divisor))] 
    #           tempValueVar <- names(dt)[3] <- paste0(tempValueVar, "_", listDivisorAndSuffix$Suffix)
    #         }
    #         
    #         #Get the cummulative position for Text
    #         dt[, TEMP_CUMSUM := cumsum(get(tempValueVar)), by = col_cluster_factor]
    # 
    #         # draw the graph
    #         gg <- ggplot(data = dt, aes(x = get(col_cluster_factor), y = get(tempValueVar)))
    #         gg <- gg + geom_bar(data = dt, aes(fill = variable), position = "stack", stat = "identity") # 
    #         gg <- gg + geom_text_repel(data = dt, aes(y = TEMP_CUMSUM, label = get(tempValueVar))) # , alpha = 0.5 , position = "jitter", jitter.width = 1.3, jitter.height = 0.1
    #         gg <- gg + labs(x = col_cluster_factor, y = col_second_cluster_factor, title = title)
    #         gg <- gg + theme(legend.position = "bottom",legend.title = element_blank(), axis.text.x = element_text(size = rel(1), angle = 45))
    #         plot(gg)    
    #         
    #         #gg <- gg + geom_tile(aes(fill = variable, height = value), colour = "grey50")
    #         
    #         dt <- NULL
    #       }
    #     } # for (col_second_cluster_factor in fac_double_fields)
    #     #legend("topright", legend = c(fac_levels), col = c(1:fac_levels_len), lty = 1, lwd = 1, bty = "n", xjust = 0, yjust = 1) # , ncol = count_cluster
    #     
    #     dev.off()          
    #   }# if is.factor
    # } # for col_cluster_factor
    
  } # is.na(plot_type)
  
  if(is.na(plot_type[1]) == F & ("num_int_multiple" %in% plot_type)) {
    
    #pdf(file = paste("./graphs/", paste(num_int_multiple_fields, collapse = "-"), ".pdf", sep = "")) # col_cluster_factor <- "x"
    
    file <- paste(num_int_multiple_fields, collapse = "-"); if(nchar(file) > 100)  file <- substr(file, 1, 100)
    file <- paste(strGraphFOlder, file , strSuffix, ".pdf", sep = ""); pdf(file = file) # col_cluster_factor <- "x"
    
    print(paste("Image for file ", file))
    
    boxplot(train[,num_int_multiple_fields, with = F] ,  main = "Un transformed plot"); # notch = T,
    boxplot(log(train[,num_int_multiple_fields, with = F]), main = "Log transformed");
    boxplot(scale(train[,num_int_multiple_fields, with = F]), main = "Scale transformed");
    
    dev.off()          
  } # is.na(plot_type)
  
  detach(package:ggrepel); detach(package:ggplot2)
  
} # exportToFile

# Export Train and Test graph side by side
exportTrainAndTestToFile<- function(train, test, filename = "./graphs/TrainTest.pdf")
{
  listCommonColumns <- intersect(names(train), names(test))
  
  pdf(file = filename)
  
  for (col_cluster_factor in listCommonColumns) 
  {
    if (is.factor(train[[col_cluster_factor]]) == T | is.logical(train[[col_cluster_factor]]) == T) {
      
      ifelse(is.logical(train[[col_cluster_factor]]), fac_levels <- c("T","F"), fac_levels <- levels(train[[col_cluster_factor]]));
      fac_levels_len <- length(fac_levels)
      
      par(mfrow = c(1,2));      
      plot(table(train[[col_cluster_factor]]), col = c(1:fac_levels_len), main = col_cluster_factor, xlab = col_cluster_factor, ylab = "");
      plot(table(test[[col_cluster_factor]]), col = c(1:fac_levels_len), main = col_cluster_factor, xlab = col_cluster_factor, ylab = "");
      legend("topright", legend = c(fac_levels), col = c(1:fac_levels_len), lty = 1, lwd = 1, bty = "n", xjust = 0, yjust = 1) # , ncol = count_cluster
    }
    else if (is.numeric(train[[col_cluster_factor]]) == T | is.integer(train[[col_cluster_factor]]) == T) {
      par(mfrow = c(1,2));      
      boxplot(train[[col_cluster_factor]], main = col_cluster_factor, xlab = "", ylab = col_cluster_factor, staplewex = 1, horizontal = T);
      text(x = round(boxplot.stats(train[[col_cluster_factor]])$stats,0), labels = round(boxplot.stats(train[[col_cluster_factor]])$stats,0), y = 1.25)
      
      boxplot(test[[col_cluster_factor]], main = col_cluster_factor, xlab = "", ylab = col_cluster_factor, staplewex = 1, horizontal = T);
      text(x = round(boxplot.stats(test[[col_cluster_factor]])$stats,0), labels = round(boxplot.stats(test[[col_cluster_factor]])$stats,0), y = 1.25)
    }
    
  } # for col_cluster_factor
  
  dev.off()
}

# This gives basic trend for time series. Input is Date, Data, file name to save graph (initial, with change point)
# frequency = g_Season1; pdfFileName = paste0(g_folderImage, strResponse, "_timeseries_trend.pdf"); whetherRemoveOutlier = F; summarise = T
getTimeSeriesTrend <- function(dt, frequency, pdfFileName,whetherRemoveOutlier = F, minor_tick_nx = 16, par_mar = c(2,2,1,1), summarise = T, str_date_breaks = "3 months") # dateSeries, dataSeries,
{
  library(trend)
  library(zoo)
  #library(Hmisc)
  library(forecast) # for bats - Time series forecast and outliers
  #library(tsoutliers)
  library(ggplot2)
  library(scales)
  
  # Create multi sesaonality time series
  zoo_data <- zoo(x = dt[[2]], order.by = dt[[1]], frequency = frequency); 
  zoo_data <- na.locf(zoo_data) # replacing each NA with the most recent non-NA prior to it.
  
  # Create single sesaonality time series
  ts_data <- as.ts(x = dt[[2]], start = dt[1,1], frequency = frequency)
  
  pdf(pdfFileName); par(mar = par_mar); #plot(zoo_data, plot.type = "multiple"); minor.tick(nx = minor_tick_nx);
  gg <- ggplot(data = dt, aes(x = as.POSIXct(dt[[1]]), y = dt[[2]])) + geom_line() + geom_smooth() + xlab("") + ylab("") + theme_bw() # lm, glm, gam, loess, rlm
  # Hide all the vertical and horizontal gridlines
  gg <- gg + theme(panel.grid.minor.x = element_blank(), panel.grid.major.x=element_blank()) + theme(panel.grid.minor.y=element_blank(), panel.grid.major.y=element_blank())
  gg <- gg + scale_x_datetime(labels = date_format("%b-%y"), breaks = date_breaks(str_date_breaks)) + theme(axis.text.x = element_text(angle = 45))
  
  if (whetherRemoveOutlier) {
    ts_outliers <- tsoutliers(ts_data);  
    print(ifelse(length(ts_outliers$index) > 0 , paste(paste("Outliers are ",paste(ts_data[ts_outliers$index], collapse = ","), sep = ":"), paste(" and replacements suggetsed are ",paste(round(ts_outliers$replacements,1), collapse = ","), sep = ":"), sep = " "), "No outlier detected"));
    
    zoo_data[ts_outliers$index] <- ts_outliers$replacements
    ts_data[ts_outliers$index] <- ts_outliers$replacements
    
    dt[ts_outliers$index,2] <- ts_outliers$replacements
    
    # Add outlier points on graph
    gg <- gg + geom_point( data = dt[ts_outliers$index,], aes(x = as.POSIXct(dt[ts_outliers$index,1]), y = dt[ts_outliers$index,2]), col = "red", pch = 13, cex = 4)
    
  } else print("No outlier detectin performed")
  
  # Now plot with or without outliers  
  plot(gg)
  
  #outlier <- tso(y = ts_data); plot(outlier.chicken)
  
  #Performes the non-parametric Cox and Stuart trend test (two-sided test). p-value < 2.2e-16. Alternative hypothesis: monotonic trend
  CoxStuart <- cs.test(ts_data); if(summarise) if(CoxStuart$p.value <= 0.05)  print(paste("CoxStuart:p is <= 0.05 hence Alternative hypothesis:", CoxStuart$alternative, sep = ""))
  
  # Performs a univariate Mann-Kendall test. p-value < 2.2e-16. Alternative hypothesis: monotonic trend
  MannKendall <- mk.test(ts_data); if(summarise) if(MannKendall$pvalg <= 0.05)  print("MannKendall:p is <= 0.05 hence Alternative hypothesis: monotonic trend")
  
  # Not feasible as multiple season are not available
  MannKendall_Seasonal <- NULL # default
  if(length(frequency) > 1)
  {
    fit = tryCatch({ fit <- csmk.test(ts_data)}, warning = function(cond) {}, error = function(cond) { print(cond); fit <- NULL;}, finally = {})
    MannKendall_Seasonal <- fit 
  } else print("Not performing test Correlated Seasonal Mann-Kendall Test (csmk.test) as minimum two season required")
  
  #partial.cor.trend.test(zoo_data, ts_tran_count_Weekday) # Error: time-series x and y must be of same length
  #partial.mk.test(zoo_data, ts_tran_count_Weekday) # Error: time-series x and y must be of same length
  
  #Pettitt's test for change-point-detection
  s.res <- pettitt.test(ts_data)# K = 61220, p-value < 2.2e-16. alternative hypothesis: true change point is present in the series. probable change point at tau 297 
  if(summarise) if(s.res$p.value <= 0.05)  print(paste("pettitt.test: p is <= 0.05 hence Alternative hypothesis:", s.res$alternative, s.res$estimate,  sep = " "))
  n <- s.res$nobs; i <- s.res$estimate
  s.1 <- mean(zoo_data[1:i]); s.2 <- mean(zoo_data[(i+1):n])
  s <- zoo(c(rep(s.1,i), rep(s.2,(n-i))), order.by = dt[[1]], frequency = frequency); tsp(s) <- tsp(zoo_data)
  #plot(zoo_data); lines(s, lty = 1, lwd = 2, col = "yellow"); minor.tick(nx = minor_tick_nx);
  
  gg <- ggplot(data = dt, aes(x = as.POSIXct(dt[[1]]), y = dt[[2]])) + geom_line() + geom_line(aes(x = as.POSIXct(dt[[1]]), y = c(rep(s.1,i), rep(s.2,(n-i)))), col = "yellow") + geom_smooth() + xlab("") + ylab("") + theme_bw()
  # Hide all the vertical and horizontal gridlines
  gg <- gg + theme(panel.grid.minor.x = element_blank(), panel.grid.major.x=element_blank()) + theme(panel.grid.minor.y=element_blank(), panel.grid.major.y=element_blank())
  gg <- gg + scale_x_datetime(labels = date_format("%b-%y"), breaks = date_breaks(str_date_breaks)) + theme(axis.text.x = element_text(angle = 45))
  plot(gg)
  
  print(paste0("File is saved at ", pdfFileName))
  dev.off()
  
  #Seasonal Sen's slope and intercept
  fit = tryCatch({ fit <- sea.sens.slope(ts_data)}, warning = function(cond) {}, error = function(cond) { print(cond); fit <- NULL;}, finally = {})
  SenSlope_Seasonal <- fit
  
  #Sen's slope and intercept
  SenSlope <- sens.slope(ts_data) # slope:  0.1123, intercept: 68.1444
  
  #Seasonal Mann-Kendall Test, p < 2.22e-16, hence monotonic trend
  fit = tryCatch({ fit <- smk.test(ts_data)}, warning = function(cond) {}, error = function(cond) { print(cond); fit <- NULL;}, finally = {})
  MannKendall_Seasonal <- fit
  
  #Wallis and Moore phase-frequency test. p-value = 0.02904, The series is significantly different from randomness 
  WallisMooreRandomness <- wm.test(ts_data); if(summarise) if(WallisMooreRandomness$p.value <= 0.05)  print(paste("WallisMooreRandomness: p is <= 0.05 hence Alternative hypothesis:", WallisMooreRandomness$alternative, sep = ""))
  
  return(list(CoxStuart = CoxStuart, MannKendall = MannKendall, MannKendall_Seasonal = MannKendall_Seasonal, test_pettitt = s.res,SenSlope = SenSlope, SenSlope_Seasonal = SenSlope_Seasonal, MannKendall_Seasonal = MannKendall_Seasonal, WallisMooreRandomness = WallisMooreRandomness))
} # getTimeSeriesTrend

#Interpreting the value of the Level of Association for factor variable using ChiSqTest and Cramer V
getCramerVDesciption <- function(train,str_Response)
{
  # response variable must be factor
  if (!(is.factor(train[[str_Response]]) | is.logical(train[[str_Response]]))) { print("Response variable in non factor or logical"); return(NULL) }
  
  col_names <- setdiff(colnames(train), str_Response)
  dt_Response <- data.table(ResponseVariable = character(), IndependentVariable = character(), ChisqP = numeric(), CramerV = numeric(), Association = character(),CramerVDesc = character())
  
  # Itertae through each column and filter fatcor column for associations
  for (col_cluster_factor in col_names) #  col_cluster_factor <- c("transaction_date") # "gb" "TranMonth", "deal_category", "transaction_date"
  {
    if (is.factor(train[[col_cluster_factor]]) | is.logical(train[[col_cluster_factor]])) {
      tbl <- table(train[[str_Response]], train[[col_cluster_factor]]); ch <- chisq.test(tbl, simulate.p.value = T); V <- sqrt(ch$statistic / sum(tbl))
      
      # Taken definition from http://groups.chass.utoronto.ca/pol242/Labs/LM-3A/LM-3A_content.htm        
      VDesc <- ifelse(V < 0.01,"No Relationship - Knowing the independent variable does not help in predicting the dependent variable." , ifelse(V >= 0.01 & V < 0.15, "Very Weak - Not generally acceptable", ifelse(V >= 0.15 & V < 0.2,"Weak - Minimally acceptable" , ifelse(V >= 0.2 & V < 0.25,"Moderate - Acceptable" , ifelse(V >= 0.25 & V < 0.30, "Moderately Strong - Desirable", ifelse(V >= 0.3 & V < 0.35, "Strong - Very Desirable", ifelse(V >= 0.35 & V < 0.40, "Very Strong - Extremely Desirable", ifelse(V >= 0.40 & V < 0.50, "Worrisomely Strong - Either an extremely good relationship or the two variables are measuring the same concept", ifelse(V >= 0.50 & V < 0.99,"Redundant - The two variables are probably measuring the same concept." ,"Perfect Relationship - If we the know the independent variable, we can perfectly predict the dependent variable")))))))))
      Association <- ifelse(V < 0.01,"No" , ifelse(V >= 0.01 & V < 0.15, "VeryWeak", ifelse(V >= 0.15 & V < 0.2,"Weak" , ifelse(V >= 0.2 & V < 0.25,"Moderate" , ifelse(V >= 0.25 & V < 0.30, "Strong", ifelse(V >= 0.3 & V < 0.35, "VeryStrong", ifelse(V >= 0.35 & V < 0.40, "ExtremelyStrong", ifelse(V >= 0.40 & V < 0.50, "WorrisomelyStrong", ifelse(V >= 0.50 & V < 0.99,"Redundant" ,"PerfectRedundant")))))))))
      
      dt_Response <- rbind(dt_Response, list(ResponseVariable = str_Response, IndependentVariable = col_cluster_factor, ChisqP = ch$p.value, CramerV = V, Association = Association, CramerVDesc = VDesc))
    } # Is factor
  } # for 
  # Sort as per usefulness of minimum p and maximum CramerV
  setorder(dt_Response, ChisqP, -CramerV)
  
  return(dt_Response)
  
} #getCramerVDesciption

# Sort the Cluster sequence with Cluster 1 as max and so on
sortCluster <- function(ClusterColumn)
{
  train <- data.table(Cluster = ClusterColumn)
  dt <- table(train$Cluster); dt <- setattr(as.data.frame(dt,stringsAsFactors = FALSE), "class",  c("data.frame","data.table"));
  names(dt) <- c('Cluster', "Count"); dt$Cluster <- as.integer(dt$Cluster); setorder(dt, -Count); max_Cluster <- max(dt$Cluster); dt$OffsetCluster <- dt$Cluster + max_Cluster; dt$NewCluster <- c(1:nrow(dt));
  
  t1 <- lapply(dt$Cluster, function(x) train[Cluster == x, Cluster := dt[dt$Cluster == x,"OffsetCluster"]])
  t1 <- lapply(dt$OffsetCluster, function(x) train[Cluster == x, Cluster := dt[dt$OffsetCluster == x,"NewCluster"]])
  
  ClusterColumn <- train$Cluster
  
  return(ClusterColumn)
} # sortCluster

#splits a block of text into individual words, stems each word, and then recombines the words together into a block of text
getStemText <- function(text, language = "porter", mc.cores = 1) {
  # stem each word in a block of text
  stem_string <- function(str, language) {
    str <- strsplit(x = str, split = " ")
    str <- SnowballC::wordStem(unlist(str), language = language)
    str <- paste(str, collapse = " ")
    return(str)
  }
  
  # stem each text block in turn
  x <- lapply(X = text, FUN = stem_string, language)
  
  # return stemed text blocks
  return(unlist(x))
} # sentences <- c('walk walks walked walking walker walkers'); getStemText(sentences, language = 'en')
#sentences <- c("assembl","assemble","assembled","assembly")

# This creates DTM. strVocabProcess will take c("None", "Prune","Hash"). colFeatureName = "product_search";strVocabProcess = "Prune"; 
#ngram_max = 1; preprocess_function = tolower; chunks_number = 10; term_count_min = 10; doc_proportion_max = 0.5; doc_proportion_min = 0.001; IsTfidf = T; hash_size = 2^18;tokenizer = word_tokenizer
getDtmUsingVocabulary <- function(train, colFeatureName,preprocess_function = tolower, tokenizer = word_tokenizer, chunks_number = 10, 
                                  ngram_max = 1, strVocabProcess = "None", term_count_min = 10, doc_proportion_max = 0.5, doc_proportion_min = 0.001,
                                  IsTfidf = T, hash_size = 2^18, IsStem = T)
{
  # do stemming
  #if(IsStem) train[[colFeatureName]] <- getStemText(train[[colFeatureName]], language = 'en');
  
  # To represent documents in vector space, first of all we have to create term -> term_id mappings. We use termin term instead of word, because 
  # actually it can be arbitrary ngram, not just single word. Having set of documents we want represent them as sparse matrix, where each row should
  # corresponds to document and each column should corresponds to term. This can be done in 2 ways: using vocabulary, or by feature hashing (hashing
  # trick).
  # Vocabulary based vectorization. Lets examine the first choice. He we collect unique terms from all documents and mark them with unique_id. 
  # vocabulary() function designed specially for this purpose.
  
  it <- itoken(train[[colFeatureName]], preprocess_function = preprocess_function, tokenizer = tokenizer, chunks_number = chunks_number)
  # using unigrams here
  vocab <- vocabulary(src = it, ngram = c(1L, ngram_max))
  #str(vocab, nchar.max = 20, width = 80, strict.width = 'wrap')
  
  # Now we can costruct DTM. Again, since all functions related to corpus construction have streaming API, we have to create iterator and provide it
  # to create_vocab_corpus function:
  
  # Take fresh iterator token as previous pointer is changed  
  it <- itoken(train[[colFeatureName]], preprocess_function = preprocess_function, tokenizer = tokenizer, chunks_number = chunks_number)
  
  #Note, that training time is quite high. We can reduce it and also significantly improve accuracy.
  #Pruning vocabulary. We will prune our vocabulary. For example we can find words "a", "the", "in" in almost all documents, but actually they don't
  #give any useful information. Usually they called stop words. But in contrast to them, corpus also contains very uncommon terms, which contained
  #only in few documents. These terms also useless, because we don't have sufficient statistics for them. Here we will filter them out:
  if(strVocabProcess == "Prune")
  { print("Pruning vocablary")
    # remove very common and uncommon words
    pruned_vocab <- prune_vocabulary(vocab, term_count_min = term_count_min, doc_proportion_max = doc_proportion_max, doc_proportion_min = doc_proportion_min)
    corpus <- create_vocab_corpus(it, vocabulary = pruned_vocab)
  } else if(strVocabProcess == "Hash") {
  print("Hashing vocablary")
  #It may be little bit worse, but DTM construction time was considerably lower. On large collections of documents this can become a serious argument.
  fh <- feature_hasher(hash_size = hash_size, ngram = c(1L, ngram_max))
  corpus <- create_hash_corpus(it, feature_hasher = fh)
  } else {
    print("Default vocablary")
    corpus <- create_vocab_corpus(it, vocabulary = vocab)
  }
  
  # get dtm
  dtm <- get_dtm(corpus)
  
  if(IsTfidf)
  {
    #TF-IDF. Also we can (and usually should!) apply TF-IDF transofrmation, which will increase weight for document-specific terms and decrease 
    #weight for widely used terms:
    dtm <- tfidf_transformer(dtm)
  } # IsTfidf
  
  return(dtm) # [1]  train: 74067 23039
  
} # getDtmUsingVocabulary

#Tag a sentence using NLP. http://www.martinschweinberger.de/blog/part-of-speech-tagging-with-r/
# Tag with examples and meaning
{
# CC = Coordinating conjunction
# CD = Cardinal number
# DT = Determiner
# EX = Existential there
# FW = Foreign word
# IN = Preposition or subordinating conjunction
# JJ = Adjective
# JJR = Adjective, comparative
# JJS = Adjective, superlative
# LS = List item marker
# MD = Modal
# NN = Noun, singular or mass
# NNS = Noun, plural
# NNP = Proper noun, singular
# NNPS = Proper noun, plural
# PDT = Predeterminer
# POS = Possessive ending
# PRP = Personal pronoun
# PRP$ = Possessive pronoun
# RB = Adverb
# RBR = Adverb, comparative
# RBS = Adverb, superlative
# RP = Particle
# SYM = Symbol
# TO = to
# UH = Interjection
# VB = Verb, base form
# VBD = Verb, past tense
# VBG = Verb, gerund or present participle
# VBN = Verb, past participle
# VBP = Verb, non¬3rd person singular present
# VBZ = Verb, 3rd person singular present
# WDT = Wh¬determiner
# WP = Wh¬pronoun
# WP$ = Possessive wh¬pronoun
# WRB = Wh¬adverb 
# 
# token  tag      wclass                                             desc
# This   DT  determiner                                       Determiner
# is  VBZ        verb     Verb, 3rd person singular present of "to be"
# the   DT  determiner                                       Determiner
# first   JJ   adjective                                        Adjective
# sentence   NN        noun                           Noun, singular or mass
# in   IN preposition         Preposition or subordinating conjunction
# the   DT  determiner                                       Determiner
# first   JJ   adjective                                        Adjective
# file   NN        noun                           Noun, singular or mass
# of   IN preposition         Preposition or subordinating conjunction
# the   DT  determiner                                       Determiner
# test   NN        noun                           Noun, singular or mass
# corpus   NN        noun                           Noun, singular or mass
# . SENT    fullstop                      Sentence ending punctuation
# This   DT  determiner                                       Determiner
# is  VBZ        verb     Verb, 3rd person singular present of "to be"
# a   DT  determiner                                       Determiner
# second   JJ   adjective                                        Adjective
# sentence   NN        noun                           Noun, singular or mass
# in   IN preposition         Preposition or subordinating conjunction
# the   DT  determiner                                       Determiner
# test   NN        noun                           Noun, singular or mass
# corpus   NN        noun                           Noun, singular or mass
# but   CC conjunction                         Coordinating conjunction
# I   PP     pronoun                                 Personal pronoun
# am  VBP        verb Verb, non-3rd person singular present of "to be"
# too   RB      adverb                                           Adverb
# lazy   JJ   adjective                                        Adjective
# to   TO          to                                               to
# write   VV        verb                                  Verb, base form
# much   RB      adverb                                           Adverb
# more  RBR      adverb                              Adverb, comparative
# so   RB      adverb                                           Adverb
# this   DT  determiner                                       Determiner
# has  VHZ        verb   Verb, 3rd person singular present of "to have"
# to   TO          to                                               to
# suffice   VV        verb                                  Verb, base form
# . SENT    fullstop                      Sentence ending punctuation
# well   RB      adverb                                           Adverb
# ,    ,       comma                                            Comma
# one   CD   number                                  Cardinal number
# more  JJRadjective                           Adjective, comparative
# sentence   NN     noun                           Noun, singular or mass
# should   MD    modal                                            Modal
# do   VV     verb                                  Verb, base form
# . SENT fullstop                      Sentence ending punctuation
# 

}

getWordTags <- function(s)
{
require("NLP")

## Need sentence and word token annotations.
sent_token_annotator <- Maxent_Sent_Token_Annotator(); word_token_annotator <- Maxent_Word_Token_Annotator()
a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))

a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
## Variant with POS tag probabilities as (additional) features.

## Determine the distribution of POS tags for word tokens.
a3w <- subset(a3, type == "word"); tags <- sapply(a3w$features, `[[`, "POS")
## Extract token/POS pairs (all of them): easy.
dt <- data.table(word = s[a3w], tags = tags)

## Extract pairs of word tokens and POS tags for second sentence:
a3ws2 <- annotations_in_spans(subset(a3, type == "word"), subset(a3, type == "sentence")[2L])[[1L]]
dt <- rbind(dt, data.table(word = s[a3ws2], tags = sapply(a3ws2$features, `[[`, "POS")))

return(dt)

}

# Clean Text with few know scenario and texts
cleanWords <- function(col_names_dtm_train, pattern, isSingleChar = T, isAlphaNumberWords = T, isStopWord = T)
{
  if(isSingleChar) col_names_dtm_train <- col_names_dtm_train[which(nchar(col_names_dtm_train) > 1)]
  
  # Text with number
  if(isAlphaNumberWords) {
  pat <- "^([[:alpha:]]+)([[:digit:]]+)"; index <- grep(pat,col_names_dtm_train); if(length(index) > 0) col_names_dtm_train <- col_names_dtm_train[-index];
  }
  #  two word in one to separate in two -"durabilityextra"
  #TBD
  
#   # Remove http* and www*,
#   pattern <- "^http*|^www*"; 
#   #remove TRUE and FALSE
#   pattern <- "true|false"; index <- grep(pattern,col_names_dtm_train); if(length(index) > 0) col_names_dtm_train <- col_names_dtm_train[-index];
#   
  if(nchar(pattern) > 0) {
    index <- grep(pattern,col_names_dtm_train); if(length(index) > 0) col_names_dtm_train <- col_names_dtm_train[-index];
  }
  # Remove stop word
  
  if(isStopWord) col_names_dtm_train <- setdiff(col_names_dtm_train, tm::stopwords())
  
  # verb removal. Not working as tags are mixed and hence NOT removing the verb
  # dt <- getWordTags(as.String(paste(col_names_dtm_train, collapse = " ")))
  # dim(dt[tags %in% c(",",".",":","","VBZ","","CD","","","","","","","")])
  
  return(col_names_dtm_train)
}

#NOT RUN
CheckAndFixSpell <- function(strText)
{
  
  library(qdap)
  
  which_misspelled(strText, suggest=FALSE)
  check_spell <- which_misspelled(train, suggest=TRUE)
  
  check_spell <- check_spelling_interactive(strText, click = F)
  #preprocessed(check_spell)
  fixit <- attributes(check_spell)$correct
  fixit(strText) # use it to fix train and test too
  
  check_spell <- check_spelling(strText)
  preprocessed(check_spell)
  fixit <- attributes(check_spell)$correct
  fixit(strText) # use it to fix train and test too
  
}

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,  layout.pos.col = matchidx$col))
    }
    
  }
}

#Correct spelling mistakes
#featureColumn <- col_names_train; train <- train[c(1:5)]; rowNum <- 1
correctSpelling <- function(featureColumn)
{
  library(qdap)
  
  rowNums <- length(featureColumn); 
  for(rowNum in c(1:(rowNums-1)))
  {
    obj_check_spell <- check_spelling_interactive(text.var = featureColumn[rowNum], assume.first.correct = T, click = F, parallel = F, n.suggests = 2, range = 1)
    preprocessed(obj_check_spell)
    fixit <- attributes(obj_check_spell)$correct
    featureColumn[rowNum]  <- obj_check_spell
    featureColumn[c(rowNum+1:rowNums)] <- fixit(featureColumn[c(rowNum+1:rowNums)]) # use it to fix remaining train
  } # for
  
}

# This function will replaces token with lemma.
# strFeature = "PRODUCT_TITLE";bucketLength = 10000; from = 1; saveFileName = "t_lemma.csv"; path_treetagger = "D:/trainings/tree-tagger-windows-3.2/TreeTagger"
getLemmafromTreetag <- function(train, strFeature, from = 1, bucketLength = 10000, saveFileName = "t_lemma.csv", saveAfterBucket = 10, path_treetagger = "D:/trainings/tree-tagger-windows-3.2/TreeTagger")
{
  library(koRpus)
  
  words_token_completed <- ""; t_saveAfterBucket <- 1
  
  from <- from;maxRowNum <- nrow(train); #maxRowNum <- 10^5; bucketLength <- 10000;from <- 150000
  
  for(rowNum in seq(from, maxRowNum, bucketLength)) # rowNum = 1
  {
    to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
    
    # start Process
    print(paste0("Bucket:",from,"-",to))
    tagged.results <- treetag(train[c(from:to)][[strFeature]], treetagger = "manual", format="obj", TT.tknz=FALSE , lang="en", TT.options=list(path= path_treetagger, preset="en"))
    
    # Remove not required columns to save memory
    tagged.results@TT.res$tag <- NULL; tagged.results@TT.res$lttr <- NULL; tagged.results@TT.res$wclass <- NULL; tagged.results@TT.res$desc <- NULL; tagged.results@TT.res$stop <- NULL; tagged.results@TT.res$stem <- NULL;
    
    #Remove rows for which lemma was not found
    tagged.results@TT.res <- subset(tagged.results@TT.res, !(tagged.results@TT.res$lemma %in% c("<unknown>","@card@")))
    
    # Few Lemma has been converted to first letter as upper case and hence make smaller for seamless comparision
    tagged.results@TT.res$lemma <- tolower(tagged.results@TT.res$lemma)
    
    # remove rows for which token and lemma are same
    tagged.results@TT.res <- subset(tagged.results@TT.res, tagged.results@TT.res$lemma != tagged.results@TT.res$token)
    tagged.results@TT.res <- unique(tagged.results@TT.res)
    unique(tagged.results@TT.res$lemma)
    dim(tagged.results@TT.res)
    #Use multi search and replace
    for( x in c(1:nrow(tagged.results@TT.res))) # x = 1
    { print(paste(x, tagged.results@TT.res$token[x], tagged.results@TT.res$lemma[x]))
      
      t <- train[c(from:to)];
      t[, strFeature] <- gsub(paste0("\\<", tagged.results@TT.res$token[x],"\\>"), tagged.results@TT.res$lemma[x], t[[strFeature]])
      train[c(from:to)]$product_search <- t[, strFeature, with = F]; t <- NULL
      
    }
    
    if(t_saveAfterBucket == saveAfterBucket) {
      print("Saving File"); t_saveAfterBucket <- 1
      write.csv(file = saveFileName, train, row.names = F)
    } else t_saveAfterBucket <- t_saveAfterBucket + 1;
    
    # end Process
    
    if(to >= maxRowNum) { write.csv(file = saveFileName, train, row.names = F); break}
    
    from <- to + 1
  } # for rownum
  
  detach(package:koRpus) 
  
  return(train)
}

# Correct spelling mistakes
# from = 1; strFeature = "product_search";bucketLength = 100; saveTrainFileName = "train_attr_desc_with_Average_Lemma_SpellingCorrected.csv";ignoreWordLength = 3
saveSpellingCorrected <- function(train, strFeature, from = 1, maxRowNum = nrow(train),bucketLength = 10, saveTrainFileName = "t_SpellingCorrected.csv", ignoreWordLength = 2, saveAfterBucket = 10)
{
  words_token_completed <- ""; time_per_row <- 0; t_saveAfterBucket <- 1
  
  library(qdap)
  from <- from; nrow_train <- nrow(train); maxRowNum <- min(maxRowNum,nrow_train); #maxRowNum <- 10^5; bucketLength <- 10000;from <- 150000
  
  for(rowNum in seq(from, maxRowNum, bucketLength))
  {
    to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
    
    # start Process
    temp <- paste0(from, "-", to); print(temp)
    strTextVectorOfList <- paste(train[c(from:to), strFeature, with = F], collapse = " ")
    #parallel = F,
    check_spell <- tryCatch({check_spelling(text.var = strTextVectorOfList, assume.first.correct = T,  n.suggests = 3, range = 50)}, warning = function(cond) {check_spell <- NULL;}, error = function(cond) {check_spell <- NULL;}, finally = {})
    
    if(!is.null(check_spell) ) {
      check_spell$row <- NULL; check_spell$word.no <- NULL; check_spell$more.suggestions <- NULL;
      
      check_spell <- subset(check_spell, nchar(check_spell$not.found) > ignoreWordLength)
      check_spell <- check_spell[!duplicated(check_spell[,c('not.found')]),] 
      
      
      # # Remove previously completed tokens
      # check_spell <- subset(check_spell, !(check_spell$not.found %in% words_token_completed))
      # 
      # # Store for next round eliminations
      # words_token_completed <- c(words_token_completed, check_spell$not.found)
      
      # #Use multi search and replace
      nrow_check_spell <- nrow(check_spell); t1 <- Sys.time();          # x <- 1
      for( x in c(1:nrow_check_spell))
      { print(paste(time_per_row, temp,x, nrow_check_spell, check_spell$not.found[x], check_spell$suggestion[x]))
        
        #train$product_search <- gsub(paste0("\\<", check_spell$not.found[x],"\\>"), check_spell$suggestion[x], train$product_search)
        
        t <- train[c(from:to)];
        t$product_search <- gsub(paste0("\\<", check_spell$not.found[x],"\\>"), check_spell$suggestion[x], t$product_search)
        train[c(from:to)]$product_search <- t$product_search; t <- NULL
        
      }
      time_per_row <- round((Sys.time() - t1)/nrow_check_spell,4)
      
      #train$product_search <- qdap::multigsub(paste0("\\<", check_spell$not.found[x],"\\>"), check_spell$suggestion, train$product_search)
      
      if(t_saveAfterBucket == saveAfterBucket) {
        print("Saving File"); t_saveAfterBucket <- 1; gc()
        write.csv(file = saveTrainFileName, train, row.names = F)
      } else t_saveAfterBucket <- t_saveAfterBucket + 1;
      
      # end Process
      
      check_spell <- NULL
    } else print("Null value observed") # if null
    
    # end Process
    if(to >= maxRowNum) { write.csv(file = saveTrainFileName, train, row.names = F); break}
    
    from <- to + 1
  } # for rownum
  
  #Remove error unimplemented type 'list' in 'EncodeElement'
  #check_spell_master$more.suggestions <- vapply(check_spell_master$more.suggestions, paste, collapse = ", ", character(1L))
  return(train)
}

# Predict in small chunk beccasue of memory issue
predictInChunk <- function(fit, test, type, from = 1, maxRowNum = nrow(test), bucketLength = 1000)
{
  # For data table holder for prediction
  # Systex for H2o is different and hence if is put here
  if(class(test) == "H2OFrame") pred <- predict(fit, test[c(1:2),], type = type)
  else pred <- predict(object = fit, newx = test[c(1:2),], type = type)
  
  pred <- as.data.table(pred)
  
  from <- from + 2
  
  for(rowNum in seq(from, maxRowNum, bucketLength))
  {
    to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
    
    # start Process
    print(paste("Range:", from, "-", to, collapse = ""))
    
    if(class(test) == "H2OFrame") temp <- predict(fit, test[c(from:to),], type = type)
    else temp <- predict(object = fit, newx = test[c(from:to),], type = type)
    
    pred <- rbind(pred, as.data.table(temp))
    # end Process
    
    if(to == maxRowNum) break;
    
    from <- to + 1
  } # for rowNum
  
  return(pred)
} # predictInChunk

# Get count, respective % of any vector
GetCountAndPercentage <- function(train, colName, iRoundCutoffDigit = 2, bConvertToLower = F)
{
  ifelse(bConvertToLower, dt <- as.data.table(table(tolower(train[[colName]]))), dt <- as.data.table(table(train[[colName]])))
  names(dt) <- c(colName, "COUNT"); setorder(dt, -COUNT)
  
  numCountSum <- sum(dt$COUNT); dt$COUNT_PERCENTAGE <- round((dt$COUNT / numCountSum)*100, iRoundCutoffDigit)
  
  return(dt)
} # GetCountAndPercentage

# Get Ctaegorical group for mean of respective numeric variable. Count and respective % of any vector colCat = "CAT114" colNum = "CONT1" bConvertToLower = True
#colCat = strResponse; numericColumn = 
GetGroup_Cat_Num <- function(train, colCat,numericColumn, iRoundCutoffDigit = 2, bConvertToLower = F, nSelectTopN = 10)
{
  # Select only required column
  df = train[, c(colCat, numericColumn), with = F]
  
  # Convert to lower if required
  if(bConvertToLower)  df[, (colCat) := tolower(eval(as.name(colCat)))]
  
  # Take mean of Numeric variable grouped by category variable. Sort Desecnding
  # df[, meanBycolCat := mean(eval(as.name(numericColumn)), na.rm = T), by = eval(as.name(colCat))]
  # df[,(numericColumn):= NULL]; df <- unique(df); names(df)[2] <- numericColumn; setorderv(df, numericColumn, order = -1)
  df <- df[, lapply(.SD, mean, na.rm = T), by = eval(as.name(colCat))]; names(df)[1] <- colCat; setorderv(df, numericColumn, order = -1)
    
  # Trim the trainling digit
  df[, (numericColumn) := round(eval(as.name(numericColumn)), iRoundCutoffDigit)]
  
  # Take top N
  df <- head(df, min(nrow(df), nSelectTopN))
  
  return(df)  # GetGroup_Cat_Num
  
} # GetGroup_Cat_Num

# Get count, respective % of any two vectors
GetCountAndPercentage_TwoFactors <- function(train, colName,colName2, iRoundCutoffDigit = 2, bConvertToLower = F)
{
  # Cretae temporary data after extracting two required columns
  dt <- data.table(colName = train[[colName]], colName2 = train[[colName2]]); names(dt) <- c(colName, colName2)
  
  # Convert to lower if required
  if(bConvertToLower) dt <- tolower(dt)
  
  # Make 2x2 table, convert to DF/DT and store row names (Value of colName)
  dt <- as.matrix(table(dt[[colName]], dt[[colName2]])); dt <- as.data.frame.matrix(dt); listRowNames <- row.names(dt)
  dt <- as.data.table(dt);  row.names(dt) <- listRowNames
  
  # Get Row and Col sums
  row_Sums <- rowSums(dt);
  col_Sums <- colSums(dt); col_Sums <- as.data.table(t(col_Sums)); 
  
  # Do the col sum first and sort the complete dt with col order
  dt <- rbind(dt, col_Sums)
  dt <- dt[, order(-col_Sums), with = F]
  
  # Prepare row sum column and append at last of sorted dt  
  row_Sums <- c(row_Sums, sum(row_Sums))
  dt <- cbind(dt, row_Sums)
  
  # Pre fix row names (Value of colName) and sort again on row sum value
  listRowNames <- c(listRowNames, "col_Sums")
  dt <- cbind(colName = listRowNames, dt)
  setorder(dt, -row_Sums)
  
  # Rearrange the dt. Put row with "col sum" at bottom
  if(nrow(dt) > 2) dt <- dt[c((2:nrow(dt)), 1)]
  
  # Put the name correctly
  names(dt)[1] <- colName
  
  # Cleaning
  row_Sums <- NULL; col_Sums <- NULL; listRowNames <- NULL; row_Sums <- NULL; row_Sums <- NULL;
  
  return(dt)
} # GetCountAndPercentage_TwoFactors

# Predict Clusters. Input parameters are test data table and centers
predictClusters <- function(test, centers) {
  
  test <- test[, intersect(names(test), names(centers)), with = F]
  from <- 1; maxRowNum <- nrow(test); bucketLength <- 10000; result <- integer()
  
  for(rowNum in seq(from, maxRowNum, bucketLength))
  {
    to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
    
    # start Process
    print(paste0(from,":",to))
    dt <- test[c(from:to)]
    # compute squared euclidean distance from each sample to each cluster center
    tmp <- sapply(seq_len(nrow(dt)), function(i) apply(centers, 1, function(v) sum((dt[i, ]-v)^2)))
    result <- c(result, max.col(-t(tmp)))  # find index of min distance
    
    # end Process
    
    if(to == maxRowNum) break;
    
    from <- to + 1
  }
  
  return(result)
} # PredictCluster

# Description: It gets round off number based on input number
getRoundCuttoff <- function(nNumber)
{
  nRound  <- ifelse(nNumber < 1, 2,ifelse(nNumber < 100, 1,0))
  
  return(nRound)
} # getRoundCuttoff
# getRoundCuttoff(1000)

# Description: It gets approprite divisor and corrosponding suffix based on input number
getDivisorAndSuffix<- function(nNumber)
{
  ifelse(nNumber < 1000, listDivisorAndSuffix  <- list(Divisor = 1, Suffix = ""),ifelse(nNumber < 10^5,listDivisorAndSuffix  <- list(Divisor = 10^3, Suffix = "K"),ifelse(nNumber < 10^6, listDivisorAndSuffix  <- list(Divisor = 10^5, Suffix = "L"),ifelse(nNumber < 10^7, listDivisorAndSuffix  <- list(Divisor = 10^6, Suffix = "M"), listDivisorAndSuffix  <- list(Divisor = 10^7, Suffix = "Cr")))))
  
  return(listDivisorAndSuffix)
  
}
# nNumber <- 123; listDivisorAndSuffix <- getDivisorAndSuffix(nNumber); paste0(nNumber, " is written as ", round(nNumber/listDivisorAndSuffix$Divisor,getRoundCuttoff(nNumber/listDivisorAndSuffix$Divisor)), " ", listDivisorAndSuffix$Suffix)

# fileName = "t1.csv"; listExclude = c( strResponse, strResponse2)
GetUniqueCharFacFeatures <- function(fileName, listExclude)
{
  print(paste0("Getting unique features for file ", fileName))
  
  train <- loadData(fileName, fileName, rows_to_read = 0)
  listCharFacFeatures <- sapply(train, class)
  
  listCharFacFeatures <- names(listCharFacFeatures[listCharFacFeatures %in% c("character", "factor")])
  
  rm(train)
  
  listCharFacFeatures <- setdiff(listCharFacFeatures, listExclude)
  print(paste0("Character or Factor variables for replacmenets are ", paste0(listCharFacFeatures, collapse = ",")))
  
  return(listCharFacFeatures)
} # GetUniqueCharFacFeatures

# fileName = "t1.csv"; listExclude = c( strResponse, strResponse2)
GetUniqueValueOfCharFacFeatures <- function(fileName, listCharFacFeatures)
{
  print(paste0("Getting unique values for file ", fileName))
  
  train <- loadData(fileName, fileName, col_select = listCharFacFeatures)
  
  dtResult <- as.data.table(do.call(cbind, lapply(train, unique)))
  
  rm(train, listCharFacFeatures)
  
  return(dtResult)
} # GetUniqueValueOfCharFacFeatures

# fileName = "train_merged_complete.csv"; strToAppendInFileName = "_transformed"
ReplaceAndSaveWithUniqueValueOfCharFacFeatures <- function(dtUniqueFeatures, fileName, strToAppendInFileName)
{
  print(paste0("Replacing values in file ", fileName))
  
  train <- loadData(fileName, fileName)
  
  listFeatures <- intersect(names(train), names(dtUniqueFeatures))
  
  if(length(listFeatures) > 0)
  {
    print(paste0("Features getting replaced are ", paste0(listFeatures, collapse = ",")))
    
    train[, listFeatures] <- as.data.table(lapply(train[, listFeatures, with = F], tolower))
    for(feature in listFeatures) # feature = listFeatures[1]
    {
      
      print(paste0("Replacing ", feature))
      listValues <- na.omit(unique(dtUniqueFeatures[[feature]]))
      
      dt <- data.table(feature = listValues,newValue = c(1:length(listValues))); names(dt)[1] <- feature
      
      train <- merge(train, dt, all.x = T, by = feature)
      #head(unique(train[, c(feature, "newValue"), with = F]), 100)
      # dt <- unique(train [, c(feature, "newValue.y"), with = F])
      # dim(dt) ;   names(train)
      
      train[, (feature) := NULL]; names(train)[ncol(train)] <- feature
      #head(unique(train[, c(feature), with = F]), 100)
      
      rm(dt, listValues)
      
    } # for(feature in listFeatures)
    
    #head(unique(train[,listFeatures, with = F]), 100)
    #length(unique(dtUniqueFeatures$app_id))
    
    fileName <- paste0(gsub(".csv", "", fileName), strToAppendInFileName, ".csv")
    
    print(paste0("Replaced file is saved to ", fileName))
    write.csv(file = fileName, train, row.names = F)
  } # if(length(listFeatures) > 0)
  else print(paste0("No coommon name to replace for file ", fileName))
  
  rm(train, listFeatures)
  
} # ReplaceAndSaveWithUniqueValueOfCharFacFeatures

# fileName_train = "t1.csv"; fileName_test = "t2.csv"; listExclude = c( strResponse, strResponse2); strToAppendInFileName = "_transformed"
# fileName_train = "train_merged_complete.csv"; fileName_test = "test_merged.csv"; listExclude = c( strResponse, strResponse2, "timestamp", "category", "group"); strToAppendInFileName = "_transformed"; bEnforceUnion = T
SaveWithUniqueValueOfCharFacFeatures <- function(fileName_train, fileName_test, listExclude, strToAppendInFileName = "_transformed", bEnforceUnion = T)
{
  # listFeaturesReplaced <- c("app_id","device_id","phone_brand") # ,"device_model"
  # listExclude <- c(listExclude, listFeaturesReplaced)
  listTrainCharFacFeatures <- GetUniqueCharFacFeatures(fileName_train, listExclude)
  listTestCharFacFeatures <- GetUniqueCharFacFeatures(fileName_test, listExclude)
  
  tempDiff <- setdiff(listTrainCharFacFeatures, listTestCharFacFeatures); if(length(tempDiff) > 0) print(paste0(paste0(tempDiff, collapse = ","), " extra is observed in File ", fileName_train))
  tempDiff <- setdiff(listTestCharFacFeatures, listTrainCharFacFeatures); if(length(tempDiff) > 0) print(paste0(paste0(tempDiff, collapse = ","), " extra is observed in File ", fileName_test))
 
  listCommonFeatures <- union(listTrainCharFacFeatures, listTestCharFacFeatures)
  if(bEnforceUnion) print(paste0("Replacing common set of features: ", paste0(listCommonFeatures, collapse = ", ")))
  
  if(!bEnforceUnion) listCommonFeatures <- listTrainCharFacFeatures
  dtTrain <- GetUniqueValueOfCharFacFeatures(fileName_train, listCommonFeatures)
  
  if(!bEnforceUnion) listCommonFeatures <- listTestCharFacFeatures
  dtTest <- GetUniqueValueOfCharFacFeatures(fileName_test, listCommonFeatures)
 
   
  dtUniqueFeatures <- rbindlist(list(dtTrain, dtTest), fill = T); 
  dtUniqueFeatures <- as.data.table(lapply(dtUniqueFeatures, tolower))
  dtUniqueFeatures <- as.data.table(lapply(dtUniqueFeatures, unique))
  #dim(dtUniqueFeatures);dim(dtTrain);dim(dtTest)
  
  # head(dtUniqueFeatures)
  # dm <- "a53"; dim(dtTrain[device_model %in% dm]);dim(dtTest[device_model %in% dm])
  # write.csv(file = "tr.csv", dtTrain, row.names = F)
  # write.csv(file = "te.csv", dtTest, row.names = F)
  
  rm(dtTrain, dtTest)
 
  fileName <- "unique_values.csv" ; print(paste0("Unique values getting replaced are saved to ", fileName))
  write.csv(file = fileName, dtUniqueFeatures, row.names = F)
  
  ReplaceAndSaveWithUniqueValueOfCharFacFeatures(dtUniqueFeatures, fileName_train, strToAppendInFileName)
  ReplaceAndSaveWithUniqueValueOfCharFacFeatures(dtUniqueFeatures, fileName_test, strToAppendInFileName)
  
} # SaveWithUniqueValueOfCharFacFeatures
#SaveWithUniqueValueOfCharFacFeatures(fileName_train = "t1.csv", fileName_test = "t2.csv", listExclude = c( strResponse, strResponse2), strToAppendInFileName = "_transformed")

#Description: Scale all common numeric column of Train and Test
# train = train; test = NULL; exclude = strResponse; bScale = T; bCenter = F
ScaleAllCommonNumericColumn <- function(train, test = NULL, exclude = NA)
{
  
#Get all interger/numeric column and scale
listAllIntegerNumericColumns <- names(train)[sapply(train, class) %in% c("integer", "numeric")]

if(!is.null(test)) listAllIntegerNumericColumns <- intersect(listAllIntegerNumericColumns, names(train)[sapply(test, class) %in% c("integer", "numeric")])
if(!is.na(exclude)) listAllIntegerNumericColumns <- setdiff(listAllIntegerNumericColumns, exclude)

#print(paste("Performing :", paste0(method, collapse = ", "), " -> Features: ", paste0(listAllIntegerNumericColumns, collapse = ", ")))

objPreProcessScale <- caret::preProcess(x = train[, listAllIntegerNumericColumns, with = F], method = "range", verbose = T) # scale

train[, listAllIntegerNumericColumns] <- predict(objPreProcessScale, train[, listAllIntegerNumericColumns, with = F])

if(!is.null(test)) test[, listAllIntegerNumericColumns] <- predict(objPreProcessScale, test[, listAllIntegerNumericColumns, with = F])

#rm(listAllIntegerNumericColumns, method, objPreProcessScale)

return(list(train = train, test = test))

} # ScaleAllCommonNumericColumn

# Report memory usages
reportMem <- function()
{
  # Perform garbage collection
  gc(verbose=FALSE)
  #How much memory is being used by malloc
  intvar <- round(memory.size())
  cat("Memory currently allocated is", intvar, "Mb\n")
  #Maximum amount of memory that has been obtained from the OS
  intvar <- round(memory.size(max=TRUE))
  cat("Maximum amount of memory allocated during this session is",
      intvar, "Mb\n")
  #Memory limit
  intvar <- round(memory.limit())
  cat("Current limit for total allocation is", intvar, "Mb\n")
}

# reportMem()

# Report memory of given object
reportObjSize <- function(x=x)
{
  bm_conv <- 1024*1024
  objsize <- round(object.size(x)/bm_conv)
  cat("Object Size =", objsize, "Mb", "\n")
}

# reportObjSize(train)

# Fill with last known data
fillInTheBlanks <- function(S) {
  ## NA in S are replaced with observed values
  
  ## accepts a vector possibly holding NA values and returns a vector
  ## where all observed values are carried forward and the first is
  ## also carried backward.  cfr na.locf from zoo library.
  L <- !is.na(S)
  c(S[L][1], S[L])[cumsum(L)+1]
}
#fillInTheBlanks(c(1, NA, NA, 2, 3, NA, 4))

{
  # x = ts_Direction_x; 
  # # parameters to pass to forecast.ets() and forecast.Arima():
  # h = ifelse(frequency(x) > 1, 2 * frequency(x), 10);
  # level = c(80, 95);
  # fan = FALSE;
  # simulate = FALSE;   
  # bootstrap.ets = FALSE;
  # bootstrap.aa = FALSE;  
  # npaths = 5000;
  # 
  # # parameters for both ets() and auto.arima()
  # lambda = NULL;
  # biasadj = FALSE;
  # ic = c("aicc", "aic", "bic");
  # 
  # 
  # # parameters to pass to ets()
  # xreg = train[, listPredictorNames];
  # model = "ZZZ";
  # damped = NULL;
  # alpha = NULL; 
  # beta = NULL;
  # gamma = NULL;
  # phi = NULL;
  # additive.only = FALSE;
  # lower = c(rep(0.0001, 3), 0.8);
  # upper = c(rep(0.9999, 3), 0.98);
  # opt.crit = c("lik", "amse", "mse", "sigma", "mae");
  # nmse = 3;
  # bounds = c("both", "usual", "admissible");
  # restrict = TRUE;
  # allow.multiplicative.trend = FALSE;
  # 
  # # parameters to pass to auto.arima:
  # d = NA;
  # D = NA;
  # max.p = 5;
  # max.q = 5;
  # max.P = 2;
  # max.Q = 2;
  # max.order = 5;
  # max.d = 2;
  # max.D = 1;
  # start.p = 2;
  # start.q = 2;
  # start.P = 1;
  # start.Q = 1;
  # stationary = FALSE;
  # seasonal = TRUE;
  # stepwise = TRUE;
  # trace = FALSE;
  # approximation = (length(x) > 100 | frequency(x) >12);
  # # xreg = NULL; # not yet implemented; only univariate allowed for now
  # test = c("kpss", "adf", "pp");
  # seasonal.test = c("ocsb", "ch");
  # allowdrift = TRUE;
  # allowmean = TRUE;
  # parallel = FALSE;
  # num.cores = 3
}
  hybridf <- function(x, 
                      # parameters to pass to forecast.ets() and forecast.Arima():
                      h = ifelse(frequency(x) > 1, 2 * frequency(x), 10),
                      level = c(80, 95),
                      fan = FALSE,
                      simulate = FALSE,   
                      bootstrap.ets = FALSE,
                      bootstrap.aa = FALSE,  
                      npaths = 5000,
                      
                      # parameters for both ets() and auto.arima()
                      lambda = NULL,
                      biasadj = FALSE,
                      ic = c("aicc", "aic", "bic"),
                      
                      
                      # parameters to pass to ets()
                      xreg = NULL,
                      model = "ZZZ",
                      damped = NULL,
                      alpha = NULL, 
                      beta = NULL,
                      gamma = NULL,
                      phi = NULL,
                      additive.only = FALSE,
                      lower = c(rep(0.0001, 3), 0.8),
                      upper = c(rep(0.9999, 3), 0.98),
                      opt.crit = c("lik", "amse", "mse", "sigma", "mae"),
                      nmse = 3,
                      bounds = c("both", "usual", "admissible"),
                      restrict = TRUE,
                      allow.multiplicative.trend = T,
                      
                      # parameters to pass to auto.arima:
                      d = NA,
                      D = NA,
                      max.p = 5,
                      max.q = 5,
                      max.P = 2,
                      max.Q = 2,
                      max.order = 5,
                      max.d = 2,
                      max.D = 1,
                      start.p = 2,
                      start.q = 2,
                      start.P = 1,
                      start.Q = 1,
                      stationary = FALSE,
                      seasonal = TRUE,
                      stepwise = TRUE,
                      trace = T,
                      approximation = (length(x) > 100 | frequency(x) >12),
                      # xreg = NULL, # not yet implemented, only univariate allowed for now
                      test = c("kpss", "adf", "pp"),
                      seasonal.test = c("ocsb", "ch"),
                      allowdrift = TRUE,
                      allowmean = TRUE,
                      parallel = FALSE,
                      num.cores = 3)
  {
    
    print("Ets ..")
    mod1 <- ets(x, model = model, damped = damped,
                alpha = alpha, beta = beta, gamma = gamma, phi = phi,
                additive.only = additive.only, lower = lower, upper = upper,
                opt.crit = opt.crit, nmse = nmse, bounds = bounds,
                restrict = restrict, allow.multiplicative.trend = allow.multiplicative.trend,
                lambda = lambda, biasadj = biasadj, ic = ic)
    
    print(mod1)
    print("Auto.arima ..")
    mod2 <- auto.arima(x,
                       d = d, D = D, max.p = max.p, max.q = max.q, max.Q = max.Q,
                       max.order = max.order, max.d = max.d, max.D = max.D,
                       start.p = start.p, start.q = start.q, start.P = start.P, start.Q=start.Q,
                       stationary = stationary, seasonal = seasonal, stepwise = stepwise, trace = trace,
                       approximation = approximation, test = test, allowdrift = allowdrift, allowmean = allowmean, 
                       parallel = parallel, num.cores = num.cores,
                       lambda = lambda, biasadj = biasadj, ic = ic, xreg = xreg)
    
    print("Ets forecast..")
    fc1 <- forecast(mod1, h = h, level = level, fan = fan, simulate = simulate,
                    bootstrap = bootstrap.ets, npaths = npaths, lambda = lambda,
                    biasadj = biasadj)
    
    print("Auto.arima forecast..")
    fc2 <- forecast(mod2, h = h, level = level, fan = fan, 
                    bootstrap = bootstrap.aa, npaths = npaths, lambda = lambda,
                    biasadj = biasadj, xreg = xreg)
    
    fc_comb <- list()
    fc_comb$model <- list(mod1 = mod1, mod2 = mod2)
    fc_comb$method <- paste("hybrid of", fc1$method, "and", fc2$method)
    fc_comb$mean <- (fc1$mean + fc2$mean) / 2
    fc_comb$lower <- fc1$lower
    
    # Shiv : Throwing exception after adding xreg
    if(is.null(xreg)) fc_comb$lower[fc_comb$lower > fc2$lower] <- fc2$lower[fc_comb$lower > fc2$lower]
    
    fc_comb$upper <- fc1$upper
    
    # Shiv : Throwing exception after adding xreg
    if(is.null(xreg))fc_comb$upper[fc_comb$upper < fc2$upper] <- fc2$upper[fc_comb$upper < fc2$upper]
    
    fc_comb$level <- level
    fc_comb$x <- x
    fc_comb$fitted <- (fc1$fitted + fc2$fitted) / 2
    fc_comb$residuals <- fc_comb$fitted - fc_comb$x
    fc_comb$fc_ets <- fc1
    fc_comb$fc_aa <- fc2
    class(fc_comb) <- "forecast"
    
    return(fc_comb)
  } # hybridf
  
  GetGGPlotForActualAndPred <- function(train, strResponse, fitted)
  {
    # Plot the actual and Pred
    
    dtResult <- data.table(id = seq_len(nrow(train)), Actual = train[[strResponse]], Pred = as.vector(fitted))
    #head(fc$fc_ets$fitted, 2)
    
    gg <- ggplot(data = dtResult, aes(x = id)) # 
    gg <- gg + geom_line(data = dtResult, aes(y = Actual), col = "red")
    gg <- gg + geom_line(data = dtResult, aes(y = Pred), col = "green")
    gg <- gg + xlab("Points sequence") + ylab(strResponse)
    
    print("Actual in red color and Pred is in green color")
    return(gg)
  }
  
  
  GetROC_AUC = function(pred, actual){
    # AUC approximation
    # http://stackoverflow.com/questions/4903092/calculate-auc-in-r
    # ty AGS
    predSort = sort(pred, decreasing = TRUE, index.return = TRUE)
    val = unlist(predSort$x)
    idx = unlist(predSort$ix) 
    
    roc_y = actual[idx];
    stack_x = cumsum(roc_y == 0)/sum(roc_y == 0)
    stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)   
    
    auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
    return(auc)
  }
  
  ### Load and prepare Data
  loadData <- function(fileName, col_drop = NULL, sep = ",", bTrainData = F, col_select = NULL, rows_to_read = -1) # getOption("datatable.integer64")
  {
    #require(bit64)
    
    tracingState()
    
    if (is.null(col_select)) train <- fread(input =  fileName, sep = sep, header = T, na.strings = c("NA"," ",""), drop = col_drop, nrows = rows_to_read,integer64 = "character")
    else train <- fread(input =  fileName, sep = sep, header = T, na.strings = c("NA"," ",""), drop = col_drop, select= col_select, nrows = rows_to_read,integer64 = "character")
    #head(train);  str(train); dim(train); View(train)
    
    return(train)
  } # Load and prepare Data
  
  
  #Description: This function converts data type to approprite format before Descriptive Analysis, Analysis and Analytics
  ConvertDataTypes <- function(dtData, listFactorFeatures = NA, bAllCharToFactors = F) # dtData = train
  {
    listColNames <- names(dtData)
    
    # Convert data types to Factor
    temp <- vector()
    
    if(!is.na(listFactorFeatures)) temp <- intersect(listFactorFeatures, listColNames)
    
    if(bAllCharToFactors)
    {
      temp <- c(temp, intersect(names(which(sapply(dtData, is.character) == T)), listColNames))
      temp <- unique(temp)
    } # bAllCharToFactors
    
    if(length(temp) > 0) tryCatch(dtData[, temp] <-   lapply(dtData[, (temp), with = F], as.factor) , error = function(cond){logerror(msg = cond)})
    
    # Clean variables
    rm(listColNames); rm(temp)
    
    return(dtData)
  }#ConvertDataTypes
  
  # Description: This provides data dictioery for given data table.
  GetDataDictionary <- function(train, nTopNLavels = 5)
  {
    mv <- getMissingValueDetails(train, bIDandUniqueColInvestigation = T)
    
    # Get top N of factor variables  
    listFacFeatures <- names(which(sapply(train, is.factor) == T))
    dtResult <- data.table()
    for(facFeature in listFacFeatures) # facFeature = listFacFeatures[1]
    {
      dt <- as.data.table(table(train[[facFeature]])); setorderv(dt, "N", order = -1)
      dtResult <- rbindlist(list(dtResult, data.table(colName = facFeature, TopValues = paste(dt[seq_len(min(nrow(dt), nTopNLavels)), V1], collapse = ", ")))) 
      rm(dt)  
    } # for(facFeature in listFacFeatures)
    
    # Get top N of numeric variables  
    listNumFeatures <- names(which(sapply(train, is.numeric) == T))
    for(facFeature in listNumFeatures) # facFeature = listNumFeatures[1]
    {
      dtResult <- rbindlist(list(dtResult, data.table(colName = facFeature, TopValues = paste0("Min: ", round(min(train[[facFeature]], na.rm = T),2), ", Max: ", round(max(train[[facFeature]], na.rm = T),2), ", Mean: ", round(mean(train[[facFeature]], na.rm = T),2)))))
    } # for(facFeature in listFacFeatures)
    
    dataDic <- merge(x = mv$Df, y = dtResult, by = 'colName', all = T, sort = F)
    
    rm(listFacFeatures); rm(listNumFeatures); rm(dtResult); rm(mv)
    
    return(dataDic)
  } # GetDataDictionary
  
  # Log Loss as defined on the kaggle forum
  LogLoss<-function(act, pred)
  {
    eps = 1e-15;
    nr = length(pred)
    pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)
    pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
    ll = sum(act*log(pred) + (1-act)*log(1-pred))
    ll = ll * -1/(length(act))
    return(ll);
  }
  
  # The function is called 'plot.likert' and takes the following arguments:
  #   - vec: The vector with the raw results
  # - possible.values: A vector with all the possible values. This is sometimes important if not all possible responses were actually ticked by your participants. Defaults to all values found in vec.
  # - left: Annotation for the left side of the plot
  # - right: Annotation for the right side of the plot
  # - plot.median: Plot the median as a little line at the top? Defaults to FALSE.
  # - plot.sd: Plot the standard deviation around the mean? Defaults to TRUE.
  # - include.absolutes: Include the absolute values as small bold black numbers above the plot? Defaults to TRUE.
  # - include.percentages: Include the percentage values as blue numbers above the plot? Defaults to TRUE.
  # - own.margins: Override the default margins for the plot. Defaults to c(2,2,3,2).
  # - othermean: Plot another mean into the visualisation as grey line above the bars. I used this to compare the results to older results for the same questions. Defaults to NULL (no other mean is plotted in this case).
  # - .: Other parameters might be passed that are used for the first call to plot().
  # The example call for the plot shown above is:
  #   plot.likert(sample(-2:2, 75, replace = T,
  #                      prob = c(0, .2, .2, .3, .3)),
  #               left = "strongly disagree",
  #               right = "strongly agree",
  #               own.margins = c(2,2,5,2),
  #               main = "I like this visualisation of Likert scale
  #               results.",
  #               possible.values = -2:2,
  #               othermean = 1.09)
  # 
  # NB: I know of all the stuff regarding the calculation of means on Likert scale items. However, it is still done a lot and you can also include the median after all.
  # 
  # Here's the function:
    
    plot.likert <- function (vec, possible.values = sort(unique(vec)), left = "linker Pol", right = "rechter Pol",
                             plot.median = F, plot.sd = T, include.absolutes = T, include.percentages = T, own.margins = c(2, 2, 3, 2),
                             othermean = NULL, .) {
      tab <- table(vec)
      if (length(tab) != length(possible.values)) {
        values.not.in.tab <- possible.values[!(possible.values %in% names(tab))]
        for (val in values.not.in.tab) {
          tab[as.character(val)] <- 0
        }
        tab <- tab[sort(names(tab))]
      }
      prop.tab <- prop.table(tab) * 100
      v.sd <- sd(vec, na.rm = T)
      v.m <- mean(vec, na.rm = T)
      v.med <- median(vec, na.rm = T)
      old.mar <- par("mar")
      par(mar = own.margins)
      # Setting-up plot region
      plot(x = c(min(as.numeric(names(tab)), na.rm = T) - 1.1, max(as.numeric(names(tab)), na.rm = T) + 1.1),
           y = c(0, 100), type = "n", xaxt = "n", yaxt = "n",
           xlab = "", ylab = "", bty = "n", .)
      # Bars
      rect(xleft = as.numeric(names(prop.tab)) - .4,
           ybottom = 0,
           xright = as.numeric(names(prop.tab)) + .4,
           ytop = prop.tab,
           border = "#00000000", col = "#ADD8E6E6")
           # Lower black line
           lines(x = c(min(as.numeric(names(tab)), na.rm = T) - .6, max(as.numeric(names(tab)), na.rm = T) + .6),
                 y = c(0, 0), col = "black", lwd = 2)
           # Upper black line
           lines(x = c(min(as.numeric(names(tab)), na.rm = T) - .6, max(as.numeric(names(tab)), na.rm = T) + .6),
                 y = c(100, 100), col = "black", lwd = 2)
           # Blue lines
           for (n.i in names(tab)) {
             lines(x = c(n.i, n.i), y = c(0, 100), col = "blue")
           }
           # Grey rectangles at sides
           rect(xleft = min(as.numeric(names(tab)), na.rm = T) - 1.1,
                ybottom = 0,
                xright = min(as.numeric(names(tab)), na.rm = T) - .6,
                ytop = 100,
                border = "#00000000", col = "grey")
                rect(xleft = max(as.numeric(names(tab)), na.rm = T) + .6,
                     ybottom = 0,
                     xright = max(as.numeric(names(tab)), na.rm = T) + 1.1,
                     ytop = 100,
                     border = "#00000000", col = "grey")
                     mtext(names(prop.tab), side = 1, at = names(prop.tab))
                     # Percentages and numbers at the top
                     if (include.percentages) mtext(paste(round(prop.tab, 0), "%"), side = 3, at = names(prop.tab), line = -.3, col = "blue")
                     if (include.absolutes) mtext(tab, side = 3, at = names(tab), line = .5, cex = .8, font = 2)
                     # Mean line
                     lines(x = c(v.m, v.m), y = c(95, 85), lwd = 6, col = "blue")
                     # Median line
                     if (plot.median) lines(x = c(v.med, v.med), y = c(95, 85), lwd = 4, col = "#00FF00AA")
                                            # Other mean line
                                            if (!is.null(othermean)) lines(x = c(othermean, othermean), y = c(85, 75), lwd = 6, col = "#00000099")
                                                                           # SD lines
                                                                           if (plot.sd) { arrows(x0 = c(v.m, v.m), x1 = c(v.m - v.sd, v.m + v.sd), y0 = c(90, 90), y1 = c(90, 90),
                                                                                                 angle = 90, length = 0, lwd = 1) }
                                                                           # Left label
                                                                           mtext(left, side = 2, line = -.5)
                                                                           # Right label
                                                                           mtext(right, side = 4, line = -.5)
                                                                           par(mar = old.mar)
    }

#Get all interger/numeric column
GetNumericColumns <- function(train) return (names(train)[sapply(train, class) %in% c("integer", "numeric")])
GetFactorLogicalColumns <- function(train) return (names(train)[sapply(train, class) %in% c("factor", "logical")])

# This Scales and create dummy variable for factors
#listFactorFeatures = catColumns; file_output
scale_and_dummy_creation <- function(train, strResponse, listFactorFeatures, file_output)
{
  listFactorFeatures <- intersect(names(train), listFactorFeatures)
  
  print("Scalling numeric features")
  train <- ScaleAllCommonNumericColumn(train, exclude = strResponse)$train
  
  print("Ceating dummy variable for factor features")
  # Manual creation of dummy features
  for(facFeature in listFactorFeatures) # facFeature = listFactorFeatures[1]
  {
    for(facLevel in levels(train[[facFeature]])) train[,(paste0(facFeature, "_",facLevel)) := ifelse(eval(as.name(facFeature)) == facLevel,1,0)]  # facLevel = levels(train[[facFeature]])[1]
  }
  
  # Delete original factor column
  train[, (listFactorFeatures) := NULL]
  
  # Set name as upper case
  names(train) <- toupper(names(train))
  
  print(paste0("Output file is saved at ", file_output))
  fwrite(train, file_output)
  
  return(train)
} #scale_and_dummy_creation

# Residual plot
# actual = test[,strResponse]
ResidualPlot <- function(actual, pred, folder_tosave, algorithum_name)
{
  suppressPackageStartupMessages(library(ggplot2)); suppressPackageStartupMessages(library(ggrepel))
  
  # create temporary data table as ggplot need it
  dt <- data.table(actual = actual, pred = pred, residual = actual - pred); names(dt) = c('actual', 'pred', 'residual')
  
  png(paste0(folder_tosave, algorithum_name, '_residual_actual_blue_and_pred_pink.png'))
  # Draw pred and actual for each observation. Maximum blue color visibility is good fit
  gg  <- ggplot(data = dt, aes(x = seq_len(nrow(dt)), y = pred))
  gg <- gg + geom_point(col = "pink")
  gg <- gg + geom_point(aes(y = actual), col = "lightblue")
  gg <- gg + geom_smooth(aes(y = pred), col = "red", se = F) + geom_smooth(aes(y = actual), col = "green", se = F) + theme_bw()
  gg <- gg + labs(title='Actual (blue/green) and Pred (pink/red)', size = 10) + labs(y='Actual and Pred', x= 'Observation sequence', size = 10)
  plot(gg); dev.off()
  
  png(paste0(folder_tosave, algorithum_name,'_Residual_vs_Actual.png'))
  # Draw residual vs actual
  gg  <- ggplot(data = dt, aes(x = actual, y = residual))
  gg <- gg + geom_point(col = "lightblue")
  gg <- gg + geom_line(aes(y = 0), col = 'black')
  gg <- gg + geom_smooth() + theme_bw()
  gg <- gg + labs(title='Residual vs Actual', size = 10, y='Residual', x= 'Actual')
  plot(gg); dev.off()
  
  png(paste0(folder_tosave, algorithum_name,'_Residual_vs_Pred.png'))
  # Draw residual vs pred.  nd actual for each observation. Maximum blue color visibility is good fit
  gg  <- ggplot(data = dt, aes(x = pred, y = residual))
  gg <- gg + geom_point(col = "lightblue")
  gg <- gg + geom_line(aes(y = 0), col = 'black')
  gg <- gg + geom_smooth() + theme_bw()
  gg <- gg + labs(title='Residual vs Pred', size = 10, y='Residual', x= 'Pred (fitted)')
  plot(gg); dev.off()
  
  png(paste0(folder_tosave, algorithum_name,'_Residual_vs_Observation_Sequence.png'))
  # Draw residual vs observation count
  gg  <- ggplot(data = dt, aes(x = seq_len(nrow(dt)), y = residual))
  gg <- gg + geom_point(col = "lightblue")
  gg <- gg + geom_line(aes(y = 0), col = 'red')
  gg <- gg + theme_bw()
  gg <- gg + labs(title='Residual vs Observation sequence', size = 10, y='Residual', x= 'Observation sequence')
  plot(gg); dev.off()
  
  # Description: QQ plot  
  png(paste0(folder_tosave, algorithum_name,'_residual_QQ_plot.png'))
  qqnorm(scale(dt$residual)); qqline(scale(dt$residual), col = 'red')
  dev.off(); 
  
  # Cleaning
  detach(package:ggrepel); detach(package:ggplot2) 
  
  Desc_Numeric_Single(dt, listNumericFeatures = 'residual', strResponse = NA, folderImageDescriptive = folder_tosave, str_prefix = algorithum_name, skip_if_present = F)
  
  print(paste0("Residual plots are saved to ", folder_tosave))
  
  # Cleaning
  rm(dt); 
  
} # ResidualPlot

# pred = pred_ranger; actual = train[[strResponse]]
classification_summary <- function(pred, actual)
{
  pred <- as.factor(pred); actual <- as.factor(actual)
  
  cm <- caret::confusionMatrix(pred, actual,dnn = c("Prediction", "Actual")) # , positive = '1'
  
  dt <- data.table(Prediction = pred, Actual = actual)
  dt <- GetCountAndPercentage_TwoFactors(dt, "Prediction", "Actual")
  print("Confusion matrix is"); print(dt)
  print("Overall statistics is"); print(cm$overall)
  
  # Kappa rating as per discussion http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english
  str_kappa_rating <- ifelse(cm$overall['Kappa'] <= 0.2, 'poor (5)', ifelse(cm$overall['Kappa'] <= 0.4, 'fair (4)', ifelse(cm$overall['Kappa'] <= 0.6, 'moderate (3)', ifelse(cm$overall['Kappa'] <= 0.8, 'good (2)', 'perfect (1)'))))
  print(paste0("Kappa rating is ", str_kappa_rating)); rm(str_kappa_rating)
  
  print("By each class statistics is"); print(cm$byClass)
  
  rm(dt, cm)
} # classification_summary

# Description: Similarity between two time series
#ts1 <- dt[[strResponse]]; ts2 <- dt[[strResponse]]
TimeSeriesSimilarity <- function(ts1, ts2)
{
  # Corelations
  Cor <- round(cor(ts1, ts2)  ,2)
  
  # F test of Variance
  Ftest <-var.test(ts1, ts2)
  ifelse(Ftest$p.value > 0.05, Ftest <- paste0("Same variance, p = ", round(Ftest$p.value,2)), Ftest <- paste0("Different variance, p = ", round(Ftest$p.value,2)))
  
  return(list(Cor = Cor, Ftest = Ftest))
} # TimeSeriesSimilarity
