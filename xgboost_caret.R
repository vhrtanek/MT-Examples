library(caret)

complete_data <- read.csv("Compelte_Cases_final.csv")
sampled_to_10K_all_cases <- read.csv("sampled_to_10K_all_cases.csv")
sampled_all_cases_1_1 <- read.csv("sampled_all_cases_1_1.csv")
sampled_all_cases_1_5 <- read.csv("sampled_all_cases_1_5.csv")
sampled_complete_cases_1_1 <- read.csv("sampled_complete_cases_1_1.csv")
sampled_complete_cases_1_5 <- read.csv("sampled_complete_cases_1_5.csv")

data <- complete_data

data$X <- NULL
msno <- data[,1]
data <- data[,2:length(data)]
data[is.na(data)] <- "None"
#data <- data[,-c(27:40)]

data$is_churn <- as.factor(data$is_churn)
data$SwitchMoreOften <- as.factor(data$SwitchMoreOften)
data$SwitchVeryOften <- as.factor(data$SwitchVeryOften)
data$DidNFindGoodMusic <- as.factor(data$DidNFindGoodMusic)

train.index <- createDataPartition(data$is_churn, p = .7, list = FALSE)
train <- data[ train.index,]
test  <- data[-train.index,]

setDT(train)
setDT(test)
#setDT(validate)

labels <- train$is_churn
ts_label <- test$is_churn
#vl_labels <- validate$is_churn 

cat("Create matricies")
new_tr <- model.matrix(~.+0,data = train[,-c("is_churn"),with=F])
#new_tr <- as.matrix(train[,-c("is_churn")])
new_ts <- model.matrix(~.+0,data = test[,-c("is_churn"),with=F])
#new_vl <- model.matrix(~.+0,data = validate[,-c("is_churn"),with=F])

labels <- as.matrix(labels)
ts_label <- as.numeric(ts_label)
#vl_labels <- as.numeric(vl_labels)

dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label = ts_label)


test <- factor(train$is_churn)
levels(test)

make.names(levels(test))
train$is_churn <- ifelse (train$is_churn == "0","No", "Yes")

library(doMC)

registerDoMC(cores = 4)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 5, 
                        #summaryFunction = twoClassSummary,
                        summaryFunction=mnLogLoss,
                        classProbs = TRUE,
                        allowParallel=T)

xgb.grid <- expand.grid(nrounds = 10,
                        eta = c(0.01:0.1),
                        max_depth = c(2:14),
                        gamma = c(1:10),
                        colsample_bytree = c(.5:1),
                        min_child_weight = c(1:10),
                        subsample = c(0.5:1)
)

xgb_tune <-caret::train(is_churn~.,
                        data=train,
                        method="xgbTree",
                        trControl=cv.ctrl,
                        tuneGrid=xgb.grid,
                        verbose=T,
                        metric="logloss",
                        nthread = 4
)

xgb_tune$bestTune

xgb_tune$results$logLoss

xgb1 <- xgb.train (data = dtrain
                   , params = xgb_tune$bestTune
                   , nrounds = 200
                   , watchlist = list(train=dtrain) #list(val=dtest,train=dtrain)
                   , print.every.n = 10
                   , verbose = 1
                   , early.stop.round = 50
                   , maximize = F 
                   , classProbs = TRUE
                   , eval_metric="logloss"
)

xgb1$best_score
xgb1$best_iteration
xgb1$evaluation_log

xgbpred <- predict (xgb1,dtest)




