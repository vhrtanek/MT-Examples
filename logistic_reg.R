######################################################################################################################
# Approach
# 1 Basic Logistic Regression model
# 2 Hyper parameter tunning, Regularization with the Lasso
######################################################################################################################
######################################################################################################################
# Packages loading
######################################################################################################################
print(paste("Packages Loading", Sys.time()))

library(caTools)
library(caret)
library(dplyr)
library(ROCR)
library(e1071)
library(MLmetrics)
library(pROC)

######################################################################################################################
# Data loading
######################################################################################################################
print(paste("Data Loading", Sys.time()))

complete_data <- read.csv("Compelte_Cases_final.csv")
sampled_to_10K_all_cases <- read.csv("sampled_to_10K_all_cases.csv")
sampled_all_cases_1_1 <- read.csv("sampled_all_cases_1_1.csv")
sampled_all_cases_1_5 <- read.csv("sampled_all_cases_1_5.csv")
sampled_complete_cases_1_1 <- read.csv("sampled_complete_cases_1_1.csv")
sampled_complete_cases_1_5 <- read.csv("sampled_complete_cases_1_5.csv")


data <- sampled_all_cases_1_1

data$X <- NULL
# da sa spravit aj krajsie
#data$reg_mday <- as.factor(data$reg_mday)
#data$reg_month <- as.factor(data$reg_month)
#data$reg_wday <- as.factor(data$reg_wday)
#data$reg_year <- as.factor(data$reg_year)
#data$day <- round(data$day)
#data$day <- as.factor(data$day)
#data$month <- as.factor(data$month)
#data$year <- as.factor(data$year)
#data$is_churn <- as.factor(data$is_churn)

msno <- data[,1]
data <- data[,2:length(data)]

train.index <- createDataPartition(data$is_churn, p = .8758, list = FALSE)
data <- data[ train.index,]

train.index <- createDataPartition(data$is_churn, p = .7, list = FALSE)
train <- data[ train.index,]
test  <- data[-train.index,]

######################################################################################################################
# Data Check - Temporary
######################################################################################################################
# NA check
#sapply(data, function(x) sum(is.na(x)))
#data %>% group_by(city) %>% summarise(n = n())

######################################################################################################################
# 1 Basic Logistic Regression model
######################################################################################################################

#print(paste("Log. regression training starts:", Sys.time()))

# Fit Model
#log_fit <- glm(is_churn ~. , family = binomial(link = 'logit'), data = train)

#print(paste("Log. regression training ends:", Sys.time()))
#str(train)
# Prediction

#log_fit$xlevels[[4]] <- c("num_100","num_25","num_50", "num_75", "num_985","None")
#log_fit$xlevels[[5]] <- c("num_100","num_25","num_50", "num_75", "num_985","None")
#log_fit$xlevels[[6]] <- c("Decrease", "Increase","None")
#log_fit$xlevels[[7]] <- c("Decrease", "Increase","None")

#predict_log <- predict(log_fit, test, type = 'response')
# Confusion Matrix
#m1 <- ifelse(predict_log > .5, 1, 0)
#confusionMatrix(test$is_churn, m1)

#    FALSE   TRUE
# 0  350228  3224
# 1   22387 12545

#Variable Importance
#varImp(model) %>% arrange(desc(Overall))

#print(paste("Anova test starts:", Sys.time()))

#Anova Test
#anova(log_fit, test = 'Chisq')


#print(paste("Log. reg. 2 training starts:", Sys.time()))

# Log model without non-significant variables
#log_fit_2 <- glm(is_churn ~ city + gender + bd + registered_via + reg_year + reg_mday + 
#                  payment_plan_days + plan_list_price + actual_amount_paid + TotalTrans + 
#                  CountOfAutoRenew + CountOfCancelation + num_25 + num_50 + num_75 + num_985 +
#                  num_100 + num_unq + total_secs + year + day + BiggestGroup
#                , family = binomial(link = 'logit'), data = train)

#print(paste("Log. reg. 2 training ends:", Sys.time()))

#summary(log_fit_2)
#predict_log_2 <- predict(log_fit_2, test, type = 'response')
#m2 <- ifelse(predict_log_2 > .5, 1, 0)
#confusionMatrix(test$is_churn, m2)


#   FALSE   TRUE
#0 350216   3236
#1  22396  12536

#Comparison of two models/ Analysis of Deviance
#anova(log_fit, log_fit_2, test = "Chisq")

#Validation of Predicted Values: ROC Curve
#pred_log_fit <- prediction(predict, test$is_churn)
#perf_log_fit <- performance(pred_log_fit, measure = "tpr", x.measure = "fpr")
#plot(perf_log_fit)

#pred_log_fit_2 <- prediction(predict_log_2, test$is_churn)
#perf_log_fit_2 <- performance(pred_log_fit_2, measure = "tpr", x.measure = "fpr")
#plot(perf_log_fit_2)

#AUC
#auc_log_fit <- performance(pred_log_fit_2, measure = "auc")
#auc_log_fit <- auc_log_fit@y.values[[1]]
#auc_log_fit

#auc_log_fit_2 <- performance(pred_log_fit, measure = "auc")
#auc_log_fit_2 <- auc_log_fit_2@y.values[[1]]
#auc_log_fit_2

######################################################################################################################
# 2 Hyper parameter tunning 
######################################################################################################################

#test %>% group_by(is_churn) %>% summarize(n = n()) %>% mutate(nn = round(n/sum(n),2))
sapply(train, function(x) sum(is.na(x)))

#str(train)

print(paste("Log. reg. 3 training starts:", Sys.time()))
train$is_churn <- as.factor(train$is_churn)
test$is_churn <- as.factor(test$is_churn)
train_control<- trainControl(method="repeatedcv", number=10, repeats = 5, savePredictions=T)
#, summaryFunction=mnLogLoss)
log_fit_3 <- train(is_churn~., 
                   train[,-c(6:8)], 
                   trControl=train_control, 
                   method="glm", 
                   family=binomial(link="logit")
)

print(paste("Log. reg. 3 training ends:", Sys.time()))

summary(log_fit_3)

predict_log_3 <- predict(log_fit_3, test, type = 'prob')

m3 <- ifelse(predict_log_3$`0` > .5, 0, 1)
confusionMatrix(test$is_churn, m3)


predict_log_3$obs <- test$is_churn
predict_log_3$pred <- m3

mnLogLoss(predict_log_3, lev = c("0","1"))

m3 <- as.numeric(m3)
test$is_churn <- as.numeric(test$is_churn)
plot(pROC::roc(m3, test$is_churn))

######################################################################################################################
# 2 Regularization with the Lasso
######################################################################################################################
train1 <- train[,-c(6:8)]
test1 <- test
train1$is_churn <- as.factor(train1$is_churn)
feature.names=names(train1)

for (f in feature.names) {
  if (class(train1[[f]])=="factor") {
    levels <- unique(c(train1[[f]]))
    train1[[f]] <- factor(train1[[f]],
                          labels=make.names(levels))
  }
}



test1$is_churn <- as.factor(test1$is_churn)

feature.names=names(test1)

for (f in feature.names) {
  if (class(test1[[f]])=="factor") {
    levels <- unique(c(test1[[f]]))
    test1[[f]] <- factor(test1[[f]],
                         labels=make.names(levels))
  }
}



#test1$day <- round(test1$day)
#test1$day <- as.factor(test1$day)
#test1$SwitchMoreOften <- ifelse(test1$SwitchMoreOften == "Doesn't Switch", "NSwitch", "Switch")
#test1$SwitchVeryOften <- ifelse(test1$SwitchVeryOften == "Doesn't Switch", "NSwitch", "Switch")
#test1$DidNFindGoodMusic <- ifelse(test1$DidNFindGoodMusic == "Didn't", "No", "Yes")


trainControl <- trainControl(#method = "cv",
  method="repeatedcv",
  number = 10,
  repeats = 5, 
  summaryFunction = prSummary,
  classProbs = T
)

tuneGrid=expand.grid(alpha= seq(0,1, by = .2), lambda=seq(0, 100, by = .1))

modelFit <- train(is_churn ~.,
                  train1, 
                  method = "glmnet", 
                  trControl = trainControl,
                  #metric = "logloss", #The metric "logloss" was not in the result set. AUC will be used instead.
                  tuneGrid = tuneGrid,
                  family="binomial")

modelFit$bestTune 
# alpha = 0 lambda = 1       only lambda tuning
# alpha = 0.8 lambda = 0     lamba + alpha tunning
#modelFit$finalModel
#coef(modelFit$finalModel, modelFit$bestTune$lambda)


confusionMatrix(modelFit, norm = "none")

predict_log_4 <- predict(modelFit, test1, type = 'prob')

m4 <- ifelse(predict_log_4$X1 > .5, "X1", "X2")

predict_log_4$obs <- test1$is_churn
predict_log_4$pred <- m4

mnLogLoss(predict_log_4, lev = c("X1","X2"))

m4 <- as.numeric(m4)
test$is_churn <- as.numeric(test$is_churn)
plot(pROC::roc(m4, test1$is_churn))








