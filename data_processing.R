setwd("C:/Predictive Analytics/Kaggle")

print(paste0("Load libraries ", Sys.time()))

packages <- c("lubridate"
              ,"dplyr"
              #,"caret"
              ,"xgboost"
              ,"ggplot2"
              ,"corrgram"
              ,"corrplot"
              ,"mlr"
              ,"Matrix"
              ,"data.table"
              ,"parallel"
              ,"parallelMap"
              ,"corrgram"
              ,"corrplot"
              ,"caTools"
              ,"pROC")

for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package, character.only=T)
  }
}

print(paste0("Load data ", Sys.time()))
members <- read.csv("members_v3.csv" , sep=",", na.strings = "" )#, nrows = 1e5)
train2 <- read.csv("data/churn_comp_refresh/train_v2.csv")#, nrows = 1e5)
transactions2 <- read.csv("data/churn_comp_refresh/transactions_v2.csv")#, nrows = 1e6)
user_log <- read.csv("data/churn_comp_refresh/user_logs_v2.csv")#, nrows = 1e5)
sample_submission_zero <-  read.csv("data/churn_comp_refresh/sample_submission_v2.csv")#, sep=",")
print(paste0("Done", Sys.time()))

#######################################################################################
###############   Structure and Missing Values check    ###############################
#######################################################################################

#str(train2)
#str(members)
#summary(members)
#sapply(members, function(x) sum(is.na(x)))

#str(transactions2)
#summary(transactions2)
#sapply(transactions2, function(x) sum(is.na(x)))

#str(user_log)
#summary(user_log)
#sapply(user_log, function(x) sum(is.na(x)))

cat("Replace incorrect values for age")


# Filter out age outside of borders
members <- members %>%
  filter(bd > 10 & bd < 100)

print(paste0("Feature type correction + Date fields extraction ", Sys.time()))

members <- members %>%
  mutate(city = factor(city)
         ,gender = factor(gender)
         ,registered_via = factor(registered_via)
         ,registration_init_time = as.Date(as.character(registration_init_time), '%Y%m%d')
         ,reg_init = ymd(registration_init_time)
         ,reg_year = year(reg_init)
         ,reg_month = month(reg_init)
         ,reg_mday = mday(reg_init)
         ,reg_wday = wday(reg_init))

#levels(members$gender)[1] <- NA
#levels(members$gender)[1] <- 'Unknown'

transactions2 <- transactions2 %>%
  mutate(payment_method_id = factor(payment_method_id)
         ,is_auto_renew = factor(is_auto_renew)
         ,is_cancel = factor(is_cancel)
         ,transaction_date = as.Date(as.character(transaction_date), '%Y%m%d')
         ,membership_expire_date = as.Date(as.character(membership_expire_date), '%Y%m%d')
         ,trans_date = ymd(transaction_date)
         ,trans_year = year(trans_date)
         ,trans_month = month(trans_date)
         ,trans_mday = mday(trans_date)
         ,trans_wday = wday(trans_date)
         ,exp_date = ymd(membership_expire_date)
         ,exp_year = year(exp_date)
         ,exp_month = month(exp_date)
         ,exp_mday = mday(exp_date)
         ,exp_wday = wday(exp_date))

user_log <- user_log %>%
  mutate(date = ymd(date)
         ,year = year(date)
         ,month = month(date)
         ,day = day(date))

cat("Combine train and test files")
data <- rbind(train2, sample_submission_zero)

cat("Merge training set and members")
data <- merge(train2, members, by = "msno", all.x = T)
sapply(data, function(x) sum(is.na(x)))

#################################################################################
#########################  Transactions   #######################################
#################################################################################

print(paste0("Transaction data processing", Sys.time()))

#cat("Reduce size of transactions a bit")
#transactions2 <- transactions2[transactions2$msno %in% levels(data$msno),]

cat("Get amount of transactions per user")
transactions2 <- transactions2 %>% add_count(msno)
colnames(transactions2)[20] <- "TotalTrans"

cat("Get discount")
transactions2 <- transactions2 %>% 
  mutate(discount = plan_list_price - actual_amount_paid)

cat("Get means of plan days and prices")
tran_mean_cols <- transactions2[,c(1,3:5,21)] %>% 
  group_by(msno) %>%
  summarise_all(mean)

cat("Get count of AutoRenews")
tran_count_col1 <- transactions2[,c(1,6)] %>% 
  filter(is_auto_renew == 1) %>%
  group_by(msno) %>%
  summarise(CountOfAutoRenew = n())

cat("Get count of Cancellations")
tran_count_col2 <- transactions2[,c(1,9)] %>% 
  filter(is_cancel == 1) %>%
  group_by(msno) %>%
  summarise(CountOfCancelation = n())

cat("Merge transaction and previously prepared dataset")
ftrans <- merge(tran_mean_cols, transactions2[,c(1,20)], by = "msno", all.x = T)
ftrans <- unique(ftrans)
ftrans <- merge(ftrans, tran_count_col1, by = "msno", all.x = T)
ftrans <- merge(ftrans, tran_count_col2, by = "msno", all.x = T)

ftrans$CanRenRation <- ftrans$CountOfCancelation / ftrans$CountOfAutoRenew
ftrans$CanTranRation <- ftrans$CountOfCancelation / ftrans$TotalTrans

data <- merge(data, ftrans, by = "msno", all.x = T)

#################################################################################
#########################  Logs   ###############################################
#################################################################################

print(paste0("Log data processing", Sys.time()))

#User Log Trends
cat("Get user log trends")
desired_col_pos <- c(1,3:9) 
colnames <- colnames(user_log[,desired_col_pos])

log_means_beg_mon <- user_log %>% 
  filter(day <= 15) %>%
  select(desired_col_pos) %>%
  group_by(msno) %>%
  summarise_all(mean)

colnames(log_means_beg_mon) <- paste(colnames, "b")
colnames(log_means_beg_mon)[1] <- "msno"

log_means_end_mon <- user_log %>% 
  filter(day > 15) %>%
  select(desired_col_pos) %>%
  group_by(msno) %>%
  summarise_all(mean)

colnames(log_means_end_mon) <- paste(colnames, "e")
colnames(log_means_end_mon)[1] <- "msno"

temp <- merge(log_means_end_mon, log_means_beg_mon, by = "msno", all = T)
temp[is.na(temp)]<-0

temp$trend <- ifelse(temp$`total_secs e` > temp$`total_secs b`, "Increase", "Decrease" )
temp$switch_trend <- ifelse(temp$`num_25 e` >  temp$`num_25 b`, "Increase", "Decrease" )

#All categories are means
cat("Get means from log")
log_means <- user_log[,-2] %>%
  group_by(msno) %>%
  summarise_all(mean)

log_means$BiggestGroup <- as.factor(colnames(log_means[,2:6])[apply(log_means[,2:6],1,which.max)])
log_means$SmallestGroup <- as.factor(colnames(log_means[,2:6])[apply(log_means[,2:6],1,which.min)])
log_means$SwitchMoreOften <- as.factor(ifelse(log_means$num_50 + log_means$num_75 + log_means$num_985 + log_means$num_100 > 3*log_means$num_25
                                              , F, T))
log_means$SwitchVeryOften <- as.factor(ifelse(log_means$num_50 + log_means$num_75 + log_means$num_985 + log_means$num_100 > 2*log_means$num_25
                                              , F, T))

cat("Num of days since last login")
LastLog <- user_log %>%
  group_by(msno) %>%
  summarise(LastLog = max(day))
LastLog$DaySinceLsatLog <- 31 - LastLog$LastLog

cat("Get average monthly logins")
avg_monthly_logins <- user_log %>%
  group_by(msno,year,month) %>%
  summarise(AvgMonthlyLogs = n())

avg_monthly_logins <- avg_monthly_logins %>%
  group_by(msno) %>% 
  summarise_at(vars(AvgMonthlyLogs), mean, na.rm = T)

cat("Merge logs and previously prepared dataset")
flogs <- merge(log_means[1:15], avg_monthly_logins, by =  "msno", all.x = T)
flogs$DidNFindGoodMusic <- as.factor(ifelse(flogs$num_unq / flogs$num_25 > 1.5, "T", "F"))

flogs <- merge(flogs, temp, by = "msno", all = T)
flogs <- merge(flogs, LastLog, by = "msno", all = T)
#flogs <- merge(flogs, avg_yearly_logins, by =  "msno", all.x = T)
data <- merge(data, flogs, by = "msno", all.x = T)


write.csv(data, file = "All_data_final.csv")
write.csv(data[complete.cases(data),], file = "Compelte_Cases_final.csv")
write.csv(data[complete.cases(data$gender),], file = "Half_Compelte_Cases_final.csv")


str(data)
sapply(data, function(x) sum(is.na(x)))

#fac_col <- sapply(data, is.factor)
#fac_data <- data[,fac_col]

cat("Missing values replacing")
data$BiggestGroup <- `levels<-`(addNA(data$BiggestGroup), c(levels(data$BiggestGroup), "None"))
data$SmallestGroup <- `levels<-`(addNA(data$SmallestGroup), c(levels(data$SmallestGroup), "None"))
data$SwitchMoreOften <- `levels<-`(addNA(data$SwitchMoreOften), c(levels(data$SwitchMoreOften), "None"))
data$SwitchVeryOften <- `levels<-`(addNA(data$SwitchVeryOften), c(levels(data$SwitchVeryOften), "None"))
data$DidNFindGoodMusic <- `levels<-`(addNA(data$DidNFindGoodMusic), c(levels(data$DidNFindGoodMusic), "None"))
data$registered_via <- `levels<-`(addNA(data$registered_via), c(levels(data$registered_via), "None"))
data$gender <- `levels<-`(addNA(data$gender), c(levels(data$gender), "None"))
data$city <- `levels<-`(addNA(data$city), c(levels(data$city), "None"))

data$registration_init_time <- NULL
data$reg_init <- NULL

data[is.na(data)] <- 0

cat("Correlation Test")
num_col <- sapply(data, is.numeric)
num_data <- data[,num_col]

M <- cor(num_data)
corrplot.mixed(M)
corrplot(M, order = "hclust", addrect = 3)

