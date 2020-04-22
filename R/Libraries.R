cran <- "https://cran.rstudio.org";  options(repos = cran, verbose = F)

#From Cran
{ 
  
  # Miscellaneous
  {
    install.packages("rpart", quiet = T)
    install.packages("scales", quiet = T) #  
  }
  
  # Utility
  {
    install.packages("Rtools", quiet = T)
    install.packages("devtools", quiet = T)
    drat::addRepo("dmlc"); 
    install.packages("packrat", quiet = T) # Takes care of  package dependencies. Similar to pkgsnap
  }
  
  # Various file I/O
  {install.packages("XLConnect", quiet = T)
  }
  
  #Logging
  {
    install.packages("logging", quiet = T)
  }
  
  # Data formatting
  {
    install.packages("unbalanced", quiet = T)
    install.packages("readr", quiet = T) # Compared to the equivalent base functions, readr functions are around 10x faster. 
    install.packages("data.table", quiet = T) # Extension of data.frame. Fast aggregation of large data (e.g. 100GB in RAM), fast ordered joins, fast add/modify/delete of columns by group using no copies at all, list columns and a fast file reader (fread). Offers a natural and flexible syntax, for faster development.
    install.packages("tidyr", quiet = T)
    install.packages("plyr", quiet = T)
    install.packages("dplyr", quiet = T) # Data manipulation in simple verb language
    install.packages("reshape2", quiet = T)
    install.packages("lubridate", quiet = T) # makes working with dates fun instead of frustrating
    install.packages("FeatureHashing", quiet = T) # "Feature hashing, also called the hashing trick, is a method to transform features to vector. Without looking up the indices in an associative array, it applies a hash function to the features and uses their hash values as indices directly. The method of feature hashing in this package was proposed in Weinberger et. al. (2009). The hashing algorithm is the murmurhash3 from the digest package.
    install.packages("rjson", quiet = T)
    install.packages("fakeR", quiet = T) # generates fake or dummy data based on a given data set using function simulate_dataset 
  }
  
  # Machine learning
  {
    install.packages("caret", quiet = T) #ML names(getModelInfo()) to see all supported packages
    install.packages("xgboost", quiet = T) #eXtreme Gradient Boosting package. 10  times faster than the classical gbm. The only thing that XGBoost does is a regression. Use Probaility for CLassificationion
    install.packages("mlbench", quiet = T) # Machine Learning Benchmark Problems (data sets). A collection of artificial and real-world machine learning benchmark problems, including, e.g., several data sets from the UCI repository.
    install.packages("mlr", quiet = T)
    install.packages("kknn", quiet = T)
  }
  
  # Sparse matrix
  {
    install.packages("Matrix", quiet = T) # sparse.model.matrix, sparseM
    install.packages("MatrixModels", quiet = T)
    install.packages("SAM", quiet = T) # Sparse Additive Modelling. The package SAM targets at high dimensional predictive modeling (regression and classification) for complex data analysis. SAM is short for sparse additive modeling, and adopts the computationally efficient basis spline technique. We solve the optimization problems by various computational algorithms including the block coordinate descent algorithm, fast iterative soft-thresholding algorithm, and newton method. The computation is further accelerated by warm-start and active-set tricks.
    install.packages("sparsenet", quiet = T)
    install.packages("SparseM", quiet = T)
    install.packages("smac", quiet = T) # Sparse Multi-category Angle-Based Large-Margin Classifiers
  }
  
  # Associations
  {
    install.packages("arules", quiet = T)
    install.packages("arulesViz", quiet = T) # library for association rules specific visualizations:
  }
  
  # Regressions and Classsification
  { 
    install.packages("ranger", quiet = T) # #the random forest implementation. The ranger package tends to be significantly faster(around 5x) and more memory efficient than the randomForest 
    install.packages("BayesTree", quiet = T)
    install.packages("C50", quiet = T)
    install.packages("car", quiet = T) #Companion to Applied Regression
    install.packages("elasticnet", quiet = T)
    install.packages("glmnet", quiet = T) #For penalized maximum likelihood. The regularization path is computed for the lasso (??=1) or  ridge (??=0) or elasticnet  penalty.The algorithm is extremely fast, and can exploit sparsity in the input matrix x. It fits linear, logistic and multinomial, poisson, and Cox regression models. A variety of predictions can be made from the fitted models. It can also fit multi-response linear regression. The package also includes methods for prediction and plotting, and a function that performs K-fold cross-validation.
    install.packages("gbm", quiet = T) # Boosting method
    install.packages("mvtboost", quiet = T) # extends the GBM model to fit boosted decision trees to multivariate continuous variables. The algorithm jointly fits models with multiple outcomes over common set of predictors. According to the authors this makes it possible to: (1) chose the number of trees and shrinkage to jointly minimize prediction error in a test set over all outcomes, (2) compare tree models across outcomes, and (3) estimate the "covariance explained" by predictors in pairs of outcomes
    install.packages("e1071", quiet = T) # SVM 
    install.packages("Metrics", quiet = T) # Metrics is a set of evaluation metrics (errors) that is commonly used in supervised machine learning.
    install.packages("ipred", quiet = T) # Bagging
    install.packages("adabag", quiet = T) # Bagging
    install.packages("tree", quiet = T) # Basic tree package
    install.packages("nls2", quiet = T) # Non linear
    install.packages("nls", quiet = T) # Non linear regression
    install.packages("pls", quiet = T) # Partial Least Squares and Principal Component Regression (PCR). Multivariate regression methods Partial Least Squares Regression (PLSR), Principal Component Regression (PCR) and Canonical Powered Partial Least Squares (CPPLS).
  }
  
  # Time series
  {
    install.packages("forecast", quiet = T) # for bats - Time series forecast
    install.packages("nlme", quiet = T) # Time series prediction
    install.packages("lme4", quiet = T) # Time series prediction
    install.packages("tsoutliers", quiet = T) # Detection of Outliers in Time Series
    install.packages("dyn", quiet = T) # Time series regression.  The dyn class interfaces ts, irts, its, zoo and zooreg time series classes to lm, glm, loess, quantreg::rq, MASS::rlm, MCMCpack::MCMCregress, quantreg::rq, randomForest::randomForest and other regression functions allowing those functions to be used with time series including specifications that may contain lags, diffs and missing values 
    install.packages("gts", quiet = T) # Group time series
    install.packages("hts", quiet = T) # Hierarchical time series
    install.packages("timeSeries", quiet = T) # class for representing general time series
    install.packages("zoo", quiet = T) #infrastructure for both regularly- and irregularly-spaced time series. zoo is particularly aimed at irregular time series of numeric vectors/matrices, but it also supports regular time series (
    install.packages("xts", quiet = T) # The xts package is the must have when it comes to times series in R. The example below loads the package and creates a daily time series of 400 days normaly distributed returns
    install.packages("tseries", quiet = T) # to measure Augmented Dickey-Fuller test for Cointegration analysis of two timeseries. contains many specialised time series functions e.g. GARCH (Generalised AutoRegressive Conditional Heteroscedastic) model fitting
    install.packages("longitudinalData", quiet = T) # Contains utility functions, including interpolation routines for imputing missing time series values
    install.packages("cts", quiet = T)	#continuous-time AutoRegressive models
    install.packages("trend", quiet = T) # Non-Parametric Trend Tests and Change-Point Detection. The analysis of environmental data often requires the detection of trends and change-points. This package provides the Mann-Kendall Trend Test,seasonal Mann-Kendall Test,correlated seasonal Mann-Kendall Test,partial Mann-Kendall Trend test,(Seasonal) Sen's slope, partial correlation trend test and change-point test after Pettitt.
    install.packages("Kendall", quiet = T) # The Kendall package has a function named MannKendall() which implements the non-parametric test for monotonic trend detection known as the Mann-Kendall test.   (A monotonic trend can be either an upward trend or a downward trend.)
    install.packages("fpp", quiet = T) # Data for ``Forecasting: principles and practice. All packages required to run the examples are also loaded.
    install.packages("portes", quiet = T) # Portmanteau Tests for Univariate and Multivariate Time Series Models
    install.packages("forecastHybrid", quiet = T) # Convenient functions for ensemble forecasts in R combining approaches from the 'forecast' package. Forecasts generated from auto.arima(), ets(), nnetar(), stlm(), and tbats() can be combined with equal weights or weights based on in-sample errors. Future methods such as cross validation are planned.
    install.packages("msm", quiet = T) # Functions for fitting continuous-time Markov and hidden Markov multi-state models to longitudinal data. Designed for processes observed at arbitrary times in continuous time (panel data) but some other observation schemes are supported. Both Markov transition rates and the hidden Markov output process can be modelled in terms of covariates, which may be constant or piecewise-constant in time.
    install.packages("msmtools", quiet = T)
    #install.packages("AnomalyDetection", quiet = T) # Anomaly Detection in presence of seasonality and an underlying trend. he underlying algorithm - referred to as Seasonal Hybrid ESD (S-H-ESD) builds upon the Generalized ESD test for detecting anomalies. Note that S-H-ESD can be used to detect both global as well as local anomalies.https://anomaly.io/anomaly-detection-twitter-r/  https://github.com/martin-magakian/Anomaly-Detection-test/blob/master/AnomalyTest.R
    install.packages("BreakoutDetection", quiet = T)
    install.packages("lmtest", quiet = T) # A collection of tests (grangertest), data sets, and examples for diagnostic checking in linear regression models. Furthermore, some generic tools for inference in parametric models are provided.
    install.packages('prophet', quiet = T)
  }
  
  # Outliers
  { 
    install.packages("outliers", quiet = T)
    install.packages("mvoutlier", quiet = T) # Multivariate outlier detection based on robust methods
    install.packages("extremevalues", quiet = T) # Detect outliers in one-dimensional data.
    install.packages("MVN", quiet = T) # Multivariate outlier
  }
  
  # Clusters
  {
    install.packages("pmclust", quiet = T) # Error. Unsupervised model-based clustering for high dimensional (ultra) large data
    install.packages("fastcluster", quiet = T) # fast hierarchical hclust, agglomerative clustering routines. 
    install.packages("cluster", quiet = T)
    install.packages("clusteval", quiet = T)
    install.packages("fpc", quiet = T) # Has a function called clusterboot() that uses bootstrap resampling to evaluate how stable a given cluster is. also has kmeansruns for providing runtime info of cluster count
    install.packages("ClustOfVar", quiet = T) # Cluster analysis of a set of variables. Variables can be quantitative, qualitative or a mixture of both.
    install.packages("clusterSim", quiet = T) # Package  allows to search for the optimal clustering procedure for a given dataset. 
    install.packages("sparcl", quiet = T) # Package  provides clustering for a set of n observations when p variables are available, where p >> n . It adaptively chooses a set of variables to use in clustering the observations. Sparse K-means clustering and sparse hierarchical clustering are implemented. 
    install.packages("dynamicTreeCut", quiet = T) # Package  contains methods for detection of clusters in hierarchical clustering dendrograms
    install.packages("sigclust", quiet = T) # Package  provides a statistical method for testing the significance of clustering results. 
    install.packages("clustvarsel", quiet = T) # Package  provides variable selection for model-based clustering.
    install.packages("apcluster", quiet = T) #An R Package for Affinity Propagation Clustering
    install.packages("upclass", quiet = T) # Updated Classification Methods using Unlabeled Data
    install.packages("NbClust", quiet = T) # have helper function to get requisite number of cluster. suppressWarnings(max(NbClust(data=ClusterData,distance = "euclidean", min.nc = 2, max.nc = 5,method="kmeans")$Best.partition))
    install.packages("ClusterR", quiet = T) # consists of centroid-based (k-means, mini-batch-kmeans, k-medoids) and distribution-based (GMM) clustering algorithms
    install.packages("factoextra", quiet = T)
    
  }
  
  # Operations
  {
    install.packages("compare", quiet = T) # Functions to compare a model object to a comparison object. If the objects are not identical, the functions can be instructed to explore various modifications of the objects (e.g., sorting rows, dropping names) to see if the modified versions are identical.
    install.packages("sets", quiet = T) # Similarity and Dissimilarity Functions , "Jaccard", "Manhattan", "Euclidean","L1", "L2"
    install.packages("vegan", quiet = T) # Distance between two sets
    install.packages("binom", quiet = T)
    install.packages("lsa", quiet = T) # For Cosine
    install.packages("bizdays", quiet = T) # Business days calculations based on a list of holidays and nonworking weekdays. Quite useful for fixed income and derivatives pricing
    install.packages("nortest", quiet = T) # to test normal distribution. Shapiro-Wilk, Anderson-Darling, Jarque-Bera, test. shapiro.test tests the NULL hypothesis that the samples came from a Normal distribution.
    install.packages("likert", quiet = T) # For likert scale
    install.packages("sqldf", quiet = T)
    install.packages("pROC", quiet = T) # for AUC. # Syntax auc(response, predictor): 
    install.packages("ROCR", quiet = T) # for AUC
    install.packages("AUC", quiet = T)# for AUC
    install.packages("futile.matrix", quiet = T) # A collection of functions for manipulating matrices and generating ensembles of random matrices
    install.packages("matrixStats", quiet = T) # A Faster Scale Function
    install.packages("FactoMineR", quiet = T) # Multiple Correspondence Analysis (MCA) with supplementary individuals, supplementary quantitative variables and supplementary categorical variables.
    install.packages("pROC", quiet = T)
  }
  
  # Visualization and graphics  
  {
    install.packages("ggplot2", quiet = T)
    install.packages("ggrepel", quiet = T) # Repel overlapping text labels away from each other.
    install.packages("jpeg", quiet = T)
    install.packages("RColorBrewer", quiet = T)
    install.packages("googleVis", quiet = T)
    install.packages("gridExtra", quiet = T) # for arranging the graphs
    install.packages("corrgram", quiet = T)
    install.packages("corrplot", quiet = T)
    install.packages("DescTools", quiet = T) # Tools for Descriptive Statistics
    install.packages("DataExplorer", quiet = T) # Tools for Descriptive Statistics
  }
  
  # Text mining
  {install.packages("PGRdup", quiet = T) # implementation of Double Metaphone for sounds like. 
    install.packages("koRpus", quiet = T) # for Text Analysis, allow to use Treetagger. Lemmantize
    install.packages("qdap", quiet = T) # spell check. Bridging the Gap Between Qualitative Data and Quantitative Analysis. 'qdap' is designed for transcript analysis, however, many functions are applicable to other areas of Text Mining/Natural Language Processing.
    install.packages("wordnet", quiet = T) #Dictionary.  provides an R interface to WordNet , a large lexical database of English
    install.packages("text2vec", quiet = T) # fast text vectorization and state-of-the art word embeddings. text analysis and natural language processing (NLP) 
    install.packages("syuzhet", quiet = T) # Extracts sentiment and sentiment-derived plot arcs from text using three sentiment dictionaries conveniently packaged for consumption by R users. Implemented dictionaries include ``afinn'' developed by Finn {\{}AA}rup Nielsen, ``bing'' developed by Minqing Hu and Bing Liu, and ``nrc'' developed by Mohammad, Saif M. and Turney, Peter D. Applicable references are available in README.md and in the documentation for the ``get_sentiment'' function. The package also provides a method for implementing Stanford's coreNLP sentiment parser. The package provides several methods for plot arc normalization.
    install.packages("RTextTools", quiet = T) # Automatic Text Classification via Supervised Learning. RTextTools is a machine learning package for automatic text classification that makes it simple for novice users to get started with machine learning, while allowing experienced users to easily experiment with different settings and algorithm combinations. The package includes nine algorithms for ensemble classification (svm, slda, boosting, bagging,random forests, glmnet, decision trees, neural networks, maximum entropy), comprehensive analytics, and thorough documentation.
    install.packages("sentiment", quiet = T)
    install.packages("twitteR", quiet = T)
    install.packages("tm", quiet = T)
    install.packages("SnowballC", quiet = T) # Text mining helper
    install.packages("wordcloud", quiet = T) # Text mining helper
    install.packages("openNLP", quiet = T) #provides an R interface to OpenNLP , a collection of natural language processing tools including a sentence detector, tokenizer, pos-tagger, shallow and full syntactic parser, and named-entity detector, using the Maxent Java package for training and using maximum entropy models. 
    install.packages("tm.plugin.webmining", quiet = T) # allow importing news feeds in XML (RSS, ATOM) and JSON formats. Currently, the following feeds are implemented: Google Blog Search, Google Finance, Google News, NYTimes Article Search, Reuters News Feed, Yahoo Finance, and Yahoo Inplay. 
    install.packages("wordcloud", quiet = T) #  provides a visualisation similar to the famous wordle ones: it horizontally and vertically distributes features in a pleasing visualisation with the font size scaled by frequency. 
    install.packages("maxent", quiet = T) #  is an implementation of maxinum entropy minimising memory consumption of very large data-sets.
    install.packages("wordnet", quiet = T) #  provides an R interface to WordNet , a large lexical database of English.
    install.packages("hunspell", quiet = T) # Hunspell is the spell checker, tokenizing, stemming and spelling in almost any language or alphabet
    install.packages("pdftools", quiet = T) #
    #quanteda for text mining, textTinyR. It is designed for processing big text data files in batches efficiently.
  }
  
  # Statistics and tests
  {
    install.packages("lsr", quiet = T) # For Cramer V
    install.packages("norm", quiet = T) # uses the multivariate normal distribution
    install.packages("usdm", quiet = T) # Calculates variance inflation factor (VIF) for a set of variables and exclude the highly correlated variables from the set through a stepwise procedure.
    install.packages("RRegrs", quiet = T) # to explore the various models and automatically calculate a number of statistics to allow to compare them 
    install.packages("propagate", quiet = T) # it allows you to fit your data to many distributions at once and tells you which one is the most appropriate. For example, we can use the fitDistr() function to fit your data to a variety of distributions. The fits for the data will be sorted by ascending Akaike information criterion (AIC) value, where the preferred model is the one with the minimum AIC value. The AIC value takes into account the trade-off between the goodness-of-fit of the model and the complexity of the model
    install.packages("fitdistrplus", quiet = T) # Fitting distribution
    install.packages("logspline", quiet = T)
    install.packages("reldist", quiet = T) # compute Gini coefficient (a measure of the inequality of a distribution, most commonly used to compare inequality in income or wealth among countries.)
    install.packages("mvtnorm", quiet = T) # Computes multivariate normal and t probabilities, quantiles, random deviates and densities.
  }
  
}
