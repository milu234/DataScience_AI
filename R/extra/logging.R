suppressPackageStartupMessages(library(logging))

g_folder_log <- "D:/trainings/R/extra/log/"

#Adding logger
addHandler(handler = writeToFile, file = paste0(g_folder_log,"test_log_", format.Date(Sys.Date(), "%b_%d_%Y"), ".log"))
basicConfig(level = "INFO") #INFO

loginfo("It is information")
logwarn("It is Warning")
logerror("It is error")
