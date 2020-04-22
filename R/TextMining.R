#https://www.kaggle.com/c/home-depot-product-search-relevance
#http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
############### Load Library ###############################
rm(list = ls(all.names = TRUE)) # Clear environment
folder_current <- "D:/Trainings/R"; setwd(folder_current); set.seed(123)

suppressPackageStartupMessages(library(data.table)); suppressPackageStartupMessages(library(text2vec))
source("CommonUtils.R")

#par(mar=c(2,2,1,1)); 

# Constants
strResponse = toupper('relevance'); g_folderImage = "./Images/"

############### Load and prepare Data ###############################
# Read raw text data
train <- fread(input = "./data/HomeDepotProductSearchRelevance_small.csv")
names(train) <- toupper(names(train))
dim(train)

# Combine title and search terms
train$PRODUCT_TITLE <- paste(train$PRODUCT_TITLE, train$SEARCH_TERM)

#Drop SEARCH_TERM now
train[, SEARCH_TERM := NULL]

head(train)
############### Various types of Cleaning ###############################
# Keep in lower case
train$PRODUCT_TITLE <- tolower(train$PRODUCT_TITLE)

library(tm)

# Clean the the contents
train$PRODUCT_TITLE <- stripWhitespace(train$PRODUCT_TITLE)
train$PRODUCT_TITLE <- stemDocument(train$PRODUCT_TITLE)
train$PRODUCT_TITLE <- removeWords(train$PRODUCT_TITLE, stopwords())
train$PRODUCT_TITLE <- removePunctuation(train$PRODUCT_TITLE, preserve_intra_word_dashes = T)

# Custom identifications
ptrn <- c("inch","ft","feet","psi","/","cu","dBA","lb","#","watt","oz","sq","hp","str","oz")
train$PRODUCT_TITLE <- removeWords(train$PRODUCT_TITLE, ptrn)

rows <- grep('\\<inch\\>', train$PRODUCT_TITLE) # See if 'inch' is there in data
#train$PRODUCT_TITLE[rows[2]]

#Remove URL patterns
ptrn <- paste0("", c("http","www","href"),"\\S+\\s*")
for(x in c(1:length(ptrn)))
{
  train$PRODUCT_TITLE <- gsub(ptrn[x], "", train$PRODUCT_TITLE)
}

# Correct few double words
str_doubleWords <- c('formationinteriorexterior','convenienceantimicrobial','applicationsresidential','underlaymentappropriate','wwwhomedepotcomcbpdont','corporationresidential','applicationscalifornia','appearanceexceptional','durabilityexceptional','applicationcalifornia','resistantresidential','appearanceformulated','appearancecalifornia','stallationcalifornia','conditionsimpervious','warrantyencapsulated','conditionscalifornia','subfloorsresidential','durabilitycalifornia','representationsclick','countertopbacksplash','elementspolyurethane','representationsonle','formationprotection','followedresidential','formationfiberglass','conuctiongalvanized','seenbspproposition','formationunlimited','formationcomposite','acrylicexceptional','stallationincludes','separatelyincludes','gradesinstallation','durabilitydesigned','filtrationsilljamb','surfacescalifornia','warrantycalifornia','stallationconucted','detailsrecommended','performancedurable','formationsuitable','surfacescustomize','applicationsdries','projectdecorative','gradeinstallation','matenanceenhanced','stallationlimited','projectcalifornia','separatelyplywood','fencesexceptional','hardwareframeless','storageadjustable','sulationcomposite','contentcalifornia','availableincludes','formationincludes','formationdesigned','conuctionrequires','formationpowerful','applicationssteel','fishescalifornia','engthresidential','conuctionstaless','retentiondurable','porosityadvanced','interiorexterior','porositydesigned','warrantyincludes','durabilitycovers','cludedcalifornia','environmentally','lumensestimated','representations','fishexceptional','fishappropriate','applicationsnot','stallcalifornia','durabilityextra','characteristics','slidesconcealed','durabilitysolid','maneuverability','floorgprefished','groundcoordates','downresidential','formationenergy','decorcalifornia','warrantyproduct','valuecalifornia','formationideal','homecalifornia','sophistication','specifications','configurations','cludedincludes','appearanceeasy','ratgresistance','recommendedfor','needcalifornia','professionally','collectionuses','engthcertified','touchconcealed','identification','fishcalifornia','onlyimpervious','formationsteel','cabetframeless','conuctionwhite','stallationhome','certifications','stallationuses','separatelyhome','certifiediapmo','dirtimpervious','transportation','formationsolid','designincludes','simultaneously','performancelow','lookcalifornia','teriorexterior','fishadjustable','operationheavy','automatically','functionality','architectural','approximately','manufacturers','sophisticated','polypropylene','adjustability','thermoplastic','polycarbonate','environmental','international','configuration','resistanteasy','hoursdaylight','formationmade','professionals','complimentary','usecompletely','revolutionary','antimicrobial','processhighly','formationwood','cludedlimited','imperfections','surfaceactual','comprehensive','warrantyclick','certification','stairsdurable','terchangeable','upexceptional','setupincludes','conuctioneasy','spaceincludes','degreesactual','communication','roompreferred','exceptionally','possibilities','trimcoordates','discoloration','specification','trafficmaster','refrigerators','complementary','refrigeration','drawersplenty','compatibility','significantly','formationhigh','characterized','electrostatic','effectiveness','reliabilityma','ergonomically','usecalifornia','dimensionally','thickunglazed','potcalifornia','traditionally','checkoutclick','aesthetically','subfloorsthis','dependability','distguishable','formationeasy','floorsresists','formationlead','indooroutdoor','fishbeautiful','accessibility','stallationcan','spiritsactual','reproductions','applications','contemporary','professional','installation','refrigerator','manufactured','temperatures','manufacturer','transitional','accommodates','requirements','polyurethane','conveniently','measurements','weatherproof','specifically','polyethylene','environments','conventional','transmission','warrantyhome','disabilities','fishprovides','electricians','retentionfor','sourcehedrix')#
str_doubleWords <- c(str_doubleWords,c('backsplashes','warrantydont','productivity','organization','alternatives','formationfor','fishincludes','guidewarning'))
str_doubleWords <- c(str_doubleWords,c('informationinteriorexterior','installationpushconnect','representationsonline','constructionstainless','consistencyengineered','infiltrationsilljamb','constructiondesigned','constructionrequires','flooringprefinished','followedresidential','maintenanceenhanced','resistantcompletely','universalreversible','seenbspproposition','finishexceptional','finishappropriate','separatelyplywood','groundcoordinates','constructionwhite','indistinguishable','ratingresistance','constructionzinc','weatherstripping','interiorexterior','constructioneasy','includedincludes','includedmatching','installationuses','environmentally','interchangeable','informationmade','maneuverability','trimcoordinates','noneconstructed','stoneimpervious','1290appropriate','constructionfor','finishbeautiful','instructionsall'))

str_doubleWordsCorrection <- c('formation interior exterior','convenience antimicrobial','applications residential','underlayment appropriate','','corporation residential','applications california','appearance exceptional','durability exceptional','application california','resistant residential','appearance formulated','appearance california','stallation california','conditions impervious','warranty encapsulated','conditions california','subfloors residential','durability california','representations click','countertop backsplash','elements polyurethane','representations onle','formation protection','followed residential','formation fiberglass','conuction galvanized','seenbsp proposition','formation unlimited','formation composite','acrylic exceptional','stallation includes','separately includes','grades installation','durability designed','filtrations illjamb','surfaces california','warranty california','stallation conucted','details recommended','performance durable','formation suitable','surfaces customize','application sdries','project decorative','grade installation','matenance enhanced','stallation limited','project california','separately plywood','fences exceptional','hardware frameless','storage adjustable','sulation composite','content california','available includes','formation includes','formation designed','conuction requires','formation powerful','applications steel','fishes california','ength residential','conuction staless','retention durable','porosity advanced','interior exterior','porosity designed','warranty includes','durability covers','cluded california','environmentally','lumens estimated','representations','fish exceptional','fish appropriate','applications not','stall california','durability extra','characteristics','slides concealed','durability solid','maneuverability','floor gprefished','ground coordates','down residential','formation energy','décor california','warranty product','value california','formation ideal','home california','sophistication','specifications','configurations','cluded includes','appearance easy','ratg resistance','recommended for','need california','professionally','collection uses','ength certified','touch concealed','identification','fish california','only impervious','formation steel','cabet frameless','conuction white','stallation home','certifications','stallationuses','separately home','certified iapmo','dirt impervious','transportation','formation solid','design includes','simultaneously','performance low','look california','terior exterior','fish adjustable','operation heavy','automatically','functionality','atrchitectural','approximately','manufacturers','sophisticated','polypropylene','adjustability','thermoplastic','polycarbonate','environmental','international','configuration','resistanteasy','hoursdaylight','formationmade','professionals','complimentary','use completely','revolutionary','anti microbial','process highly','formation wood','cluded limited','imperfections','surface actual','comprehensive','warranty click','certification','stairs durable','terchangeable','upexceptional','setup includes','conuction easy','space includes','degrees actual','communication','room preferred','exceptionally','possibilities','trim coordates','dis coloration','specification','traffic master','refrigerators','complementary','refrigeration','drawers plenty','compatibility','significantly','formation high','characterized','electrostatic','effectiveness','reliability ma','ergonomically','use california','dimensionally','thick unglazed','pot california','traditionally','checkout click','aesthetically','subfloors this','dependability','distguishable','formation easy','floors resists','formation lead','indoor outdoor','fish beautiful','accessibility','stallation can','spirits actual','reproductions','applications','contemporary','professional','installation','refrigerator','manufactured','temperatures','manufacturer','transitional','accommodates','requirements','poly urethane','conveniently','measurements','weatherproof','specifically','polyethylene')
str_doubleWordsCorrection <- c(str_doubleWordsCorrection, c('environments','conventional','transmission','warranty home','disabilities','fish provides','electricians','retention for','source hedrix','back splashes','warranty dont','productivity','organization','alternatives','formation for','fish includes','guide warning'))
str_doubleWordsCorrection <- c(str_doubleWordsCorrection, c('information interior exterior','installation push connect','representation online','construction stainless','consistency engineer','infiltrations ill jamb','construction design','construction require','flooring prefinished','followed residential','maintenance enhanced','resistant completely','universal reversible','seen bsp proposition','finish exceptional','finish appropriate','separately plywood','ground coordinates','construction white','indistinguishable','rating resistance','construction zinc','weather stripping','interior exterior','construction easy','include include','include match','installation use','environmental','interchangeable','information made','maneuverability','trim coordinates','non econstructed','stone impervious','1290 appropriate','construction for','finish beautiful','instructions all'))

# Full words '\\< \\>' only 
for( x in c(1:length(str_doubleWords)))
{
  train$PRODUCT_TITLE <- gsub(paste0("\\<", str_doubleWords[x],"\\>"), str_doubleWordsCorrection[x], train$PRODUCT_TITLE)
}

rm(str_doubleWords, str_doubleWordsCorrection)

# Few spelling mistakes. Adding space before and after
train$PRODUCT_TITLE <- gsub("california"," california ",train$PRODUCT_TITLE)

############### Lemmantisation from 3rd party library "textstem"###############################
# Library load
library(textstem)

# Few papameters for doing lemma in bucket
from = 1;maxRowNum <- nrow(train); bucketLength = min(maxRowNum, 100);
#words_token_completed <- ""; t_saveAfterBucket <- 1

# For each bucket, get lemma and repalce in actual extries
for(rowNum in seq(from, maxRowNum, bucketLength)) # rowNum = 1; to = 2 
{
  to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
  
  # start Process
  print(paste0("Bucket:",from,"-",to))
  train[c(from:to)]$PRODUCT_TITLE <- lemmatize_strings(train[c(from:to)]$PRODUCT_TITLE)
  
  # end Process
  
  if(to >= maxRowNum) break
  
  from <- to + 1;
} # for rownum

head(train, 20)

detach(package:textstem) 

############### Lemmantisation from 3rd party library 'koRpus::treetag"###############################
# If you have run "textstem" then rerun to the line just before "textstem"
# Library load
library(koRpus)

# Few papameters for doing lemma in bucket
bucketLength = 100; from = 1;maxRowNum <- nrow(train); path_treetagger = "D:/trainings/tree-tagger-windows-3.2/TreeTagger"
#words_token_completed <- ""; t_saveAfterBucket <- 1

# For each bucket, get lemma and repalce in actual extries
for(rowNum in seq(from, maxRowNum, bucketLength)) # rowNum = 1; to = 2 
{
  to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
  
  # start Process
  print(paste0("Bucket:",from,"-",to))
  token_lemma <- treetag(train[c(from:to)]$PRODUCT_TITLE, treetagger = "manual", format="obj", TT.tknz=FALSE , lang="en", TT.options=list(path= path_treetagger, preset="en"))
  
  # Remove not required columns to save memory
  token_lemma@TT.res$tag <- NULL; token_lemma@TT.res$lttr <- NULL; token_lemma@TT.res$wclass <- NULL; token_lemma@TT.res$desc <- NULL; token_lemma@TT.res$stop <- NULL; token_lemma@TT.res$stem <- NULL;
  
  # By default, it is data.frame and convert to data.table for uniform processing
  token_lemma <- as.data.table(token_lemma@TT.res)
  
  #View token lemma
  head(token_lemma)
  
  # Few Lemma has been converted to first letter as upper case and hence make smaller for seamless comparision
  token_lemma$lemma <- tolower(token_lemma$lemma)
  
  # Keep rows for which token and lemma are different
  token_lemma <- token_lemma[lemma != token]
  token_lemma <- unique(token_lemma)
  
  #Remove rows for which lemma was not found
  unique(token_lemma$lemma)
  token_lemma <- token_lemma[!(lemma %in% c("<unknown>","@card@"))]
  
  #Use multi search and replace
  for( x in c(1:nrow(token_lemma))) # x = 1
  { 
    print(paste0(x, ". Replacing '",token_lemma$token[x], "' by '",token_lemma$lemma[x],"'"))
    train[c(from:to)]$PRODUCT_TITLE <- gsub(paste0("\\<", token_lemma$token[x],"\\>"), token_lemma$lemma[x], train[c(from:to)]$PRODUCT_TITLE)
  }
  
  # end Process
  
  if(to >= maxRowNum) break
  
  from <- to + 1; rm(token_lemma)
} # for rownum

detach(package:koRpus) 

############### Spelling Correction ###############################
# Library load
suppressPackageStartupMessages(library(qdap))

# Few papameters for doing correction in bucket. Keep bucket length small so that maximum correction is done in begining
from = 1; ignoreWordLength = 3; maxRowNum <- nrow(train); bucketLength = min(maxRowNum, 100);

# do spelling correction bucket by bucket
for(rowNum in seq(from, maxRowNum, bucketLength)) # rowNum = 1
{
  to <- from + bucketLength  -1; if(to > maxRowNum) to <- maxRowNum;
  
  # start Process
  print(paste0(from, "-", to))
  check_spell <- tryCatch({check_spelling(text.var = train[c(from:to)]$PRODUCT_TITLE, assume.first.correct = T,  n.suggests = 3, range = 50)}, warning = function(cond) {check_spell <- NULL;}, error = function(cond) {check_spell <- NULL;}, finally = {})
  
    # Extract correct spelling
    if(!is.null(check_spell) ) {
      
      #Remove not required columns
      check_spell$row <- NULL; check_spell$word.no <- NULL; check_spell$more.suggestions <- NULL;
      
      # By default, it is data.frame like object and convert to data.table for uniform processing
      check_spell <- as.data.table(check_spell)
      
      # There may be few small or incomplete words. Remove those
      check_spell <- check_spell[nchar(check_spell$not.found) > ignoreWordLength]
      
      # If correct spelling suggestion is duplicate then remove that
      check_spell <- check_spell[!duplicated(check_spell[['not.found']])]
      
      #Use multi search and replace
      nrow_check_spell <- nrow(check_spell)
      for( x in c(1:nrow_check_spell)) # x = 1
      { 
        print(paste0(x, ". Replacing '",check_spell$not.found[x], "' by '",check_spell$suggestion[x],"'"))
        train[c(from:to)]$PRODUCT_TITLE <- gsub(paste0("\\<", check_spell$not.found[x],"\\>"), check_spell$suggestion[x], train[c(from:to)]$PRODUCT_TITLE)
      }
      
      # end Process
      
      rm(check_spell); gc()
    } else print("Null value observed. May not be spelling mistake") # if null
  
  # end Process
  if(to >= maxRowNum) break
  
  from <- to + 1
} # for rownum

detach(package:qdap)

# Save clean data
fwrite(train,"./data/HomeDepotProductSearchRelevance_cleaned_small.csv")

############### DTM Using Vocabulary ###############################
# Read clean text data
system.time(train <- fread(input = "./data/HomeDepotProductSearchRelevance_cleaned_small.csv"))

# Few papameters initialisation
ngram_max = 1; preprocess_function = tolower; chunks_number = 10; tokenizer = word_tokenizer

# To represent documents in vector space, first of all we have to create term -> term_id mappings. We use term in 
# term instead  of word, because it can be arbitrary ngram, not just single word. Having set of documents we want 
# represent them as sparse matrix, where each row should corresponds to document and each column should corresponds
# to term. This can be done in 2 ways: using vocabulary, or by feature hashing (hashing trick).

# Vocabulary based vectorization. Lets examine the first choice. He we collect unique terms from all documents and
# mark them  with unique_id. vocabulary() function designed specially for this purpose.

# Iterator
it <- itoken(train$PRODUCT_TITLE, preprocess_function = preprocess_function, tokenizer = tokenizer, chunks_number = chunks_number)

# Create vocablary using unigrams here
vocab <- create_vocabulary(it, ngram = c(1L, ngram_max))
class(vocab)
dim(vocab)
head(vocab, 10) 

# Now we can costruct DTM. Again, since all functions related to corpus construction have streaming API, we have 
#to create iterator and provide it to create_vocab_corpus function:

#Pruning vocabulary. We will prune our vocabulary. For example we can find words "a", "the", "in" in almost all 
# documents, but actually they don't give any useful information. Usually they called stop words. But in contrast to
# them, corpus also contains very uncommon terms, which contained only in few documents. These terms also useless, 
# because we don't have sufficient statistics for them. Here we will filter them out

# Defining constants
term_count_min = 10; # minimum number of occurences over all documents
doc_proportion_max = 0.5; # maximum proportion of documents which should contain term
doc_proportion_min = 0.001; # minimum proportion of documents which should contain term

print("Pruning vocablary: Remove very common and uncommon words")
pruned_vocab <- prune_vocabulary(vocab, term_count_min = term_count_min, doc_proportion_max = doc_proportion_max, doc_proportion_min = doc_proportion_min)
dim(pruned_vocab)
head(pruned_vocab)

# Take fresh iterator token as previous pointer is changed  
it <- itoken(train$PRODUCT_TITLE, preprocess_function = preprocess_function, tokenizer = tokenizer, chunks_number = chunks_number)

# get dtm
dtm <- create_dtm(it, vocab_vectorizer(pruned_vocab)) 
dim(dtm)
dtm[1,1:10]

#TF-IDF (term frequency-inverse document frequency). Also we can (and usually should!) apply TF-IDF transofrmation, 
# which will increase weight for document-specific terms and decrease weight for widely used terms
# TF * IDF = [ (Number of times term t appears in a document) / (Total number of terms in the document) ] 
#            * log10(Total number of documents / Number of documents with term t in it)

fit_tfidf = TfIdf$new()
dtm <- fit_tfidf$fit_transform(dtm)
dim(dtm)
dtm[1,1:10]

# Merge the data in original train object
dt <- as.data.table(as.matrix(dtm))
dt$RELEVANCE <- train$RELEVANCE

rm(train); train <- dt

dim(train)
train[1:10,1:10]

rm(dtm, pruned_vocab, it, vocab, dt, fit_tfidf)

############### Supervised Analytics using glmnet ###############################
# Create partition for validation
suppressPackageStartupMessages(library(caret))
inTrain = createDataPartition(y = train[[strResponse]], p = 0.85, list = F, times = 1)
test <- train[-inTrain,]; train <- train[ inTrain,]
dim(train); dim(test)
detach(package:caret) 

# fit glmnet (Lasso and Ridge) regression
suppressPackageStartupMessages(library(glmnet))

listPredictorNames <- setdiff(colnames(train),strResponse)

# It need matrix as input
train <- as.matrix(train)

# fit glmnet. alpha=1 is the lasso penalty, and alpha=0 the ridge penalty.
glmnet_fit <- glmnet(x = train[,listPredictorNames], y = train[,strResponse] , alpha=0.5, family = 'gaussian',  standardize = F)

# Predict
test <- as.matrix(test)
pred <- predict(object = glmnet_fit, newx = test[,listPredictorNames], s = min(glmnet_fit$lambda)) # 'lambda.min' , type = "response"
pred <- as.vector(pred)

# Error Caluclations
MAE_glmnet <- ModelMetrics::mae(predicted = pred,actual = test[, strResponse]) 
RMSE_glmnet <- ModelMetrics::rmse(predicted = pred,actual = test[, strResponse]) #RMSE <- sqrt(sum((pred_xgb - getinfo(train_xgbMat, "label"))^2)/length(pred_xgb))
print(paste0("Overall MAE: ", round(MAE_glmnet, 2), ", RMSE: ", round(RMSE_glmnet, 2))) #"Overall MAE: 0.49, RMSE: 0.58"
summary(test[, strResponse])

# Plot actual and pred in lines
library(ggplot2)
test <- as.data.table(test)
gg <- GetGGPlotForActualAndPred(test, strResponse, pred)
plot(gg)
detach(package:ggplot2) 

# CW: Plot actual and pred in points

# ResidualPlot
ResidualPlot(actual = test[,strResponse, with = F], pred, folder_tosave = g_folderImage, algorithum_name = "tm_glmnet")

############### Word cloud ###############################
library(wordcloud)

# Get count (frequency by sum of each column)
words_freq <- apply(train, 2, sum)
words_freq

wordcloud(words = names(words_freq), freq = words_freq, min.freq = 3, random.order = F, max.words = 250)
wordcloud(words = names(words_freq), freq = words_freq,random.order = F, max.words = 100, min.freq = 5, colors=brewer.pal(10, "Dark2"))

rm(words_freq); detach(package:wordcloud)
