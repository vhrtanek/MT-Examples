packages <- c("lubridate"
              ,"dplyr"
              #,"caret"
              ,"xgboost"
              ,"ggplot2"
              ,"mlr"
              ,"Matrix"
              ,"data.table"
              ,"parallel"
              ,"parallelMap"
              ,"caTools"
              ,"pROC")

for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package, character.only=T)
  }
}

cat("Create Training/Test sets")
set.seed(12345)
sample <- sample.split(data$is_churn, SplitRatio = .7)
train <- subset(data, sample == TRUE)
test <- subset(data, sample == FALSE)

train$X <- NULL
test$X <- NULL

setDT(train)
setDT(test)
#setDT(validate)

labels <- train$is_churn 
ts_label <- test$is_churn
#vl_labels <- validate$is_churn 

cat("Create matricies")
new_tr <- model.matrix(~.+0,data = train[,-c("is_churn"),with=F])
new_ts <- model.matrix(~.+0,data = test[,-c("is_churn"),with=F])
#new_vl <- model.matrix(~.+0,data = validate[,-c("is_churn"),with=F])

labels <- as.numeric(labels)
ts_label <- as.numeric(ts_label)
#vl_labels <- as.numeric(vl_labels)

dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)
#dvalidate <- xgb.DMatrix(data = new_vl,label=vl_labels)


cat("Set default parameters")
params <- list(booster = "gbtree"
               ,eta=0.3
               ,gamma=0
               ,max_depth=6
               ,min_child_weight=1
               ,subsample=1
               ,colsample_bytree=1)

cat("Xgb cross-validation")
xgb_cv <- xgb.cv(data = dtrain
                 ,params = params
                 ,nrounds = 100
                 ,nfold = 5
                 ,showsd = T 
                 ,stratified = T
                 ,maximize = FALSE
                 ,prediction = TRUE
                 ,print_every_n = 10
                 ,metrics = "logloss"
                 ,early_stopping_round = 50)

best_iter <- xgb_cv$best_iteration #43
min(xgb_cv$evaluation_log$test_rmse_mean)

print(paste0("Train default XGB", Sys.time()))

parallelStartSocket(cpus = detectCores())

xgb1 <- xgb.train (data = dtrain
                   , params = params
                   , nrounds = best_iter
                   , watchlist = list(train=dtrain) #list(val=dtest,train=dtrain)
                   , print.every.n = 10
                   , verbose = 1
                   , early.stop.round = 50
                   , maximize = F 
                   , eval_metric="logloss")

xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)
confusionMatrix(xgbpred, ts_label)

xgb1$best_score
xgb1$evaluation_log$train_logloss

plot(pROC::roc(xgbpred, ts_label))

cat("Feature importance")
importance_matrix <- xgb.importance(feature_names = colnames(new_tr), model=xgb1)
xgb.plot.importance(importance_matrix = importance_matrix[1:8])

str(train)

print(paste0("Prepare task and learner for tuning", Sys.time()))
cat("Create task for mlr")
fact_col <- colnames(train)[sapply(train,is.factor)]
for(i in fact_col) set(train,j=i,value = factor(train[[i]]))
for(i in fact_col) set(test,j=i,value = factor(test[[i]]))

traintask <- makeClassifTask (data = train,target = "is_churn")
testtask <- makeClassifTask (data = test,target = "is_churn")

traintask <- normalizeFeatures(traintask,method = "standardize")

traintask <- createDummyFeatures (obj = traintask)
testtask <- createDummyFeatures (obj = testtask)

#Feature importance - works only with rJava => under 32bit Rstudio => unusable with bigger dataset
#im_feat <- generateFilterValuesData(traintask, method = c("information.gain","chi.squared"))
#plotFilterValues(im_feat,n.show = 20)

#Feature Selection - top 8
#top_task <- filterFeatures(trainTask, method = "rf.importance", abs = 8)

cat("Create learner for mlr")
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="logloss", nrounds=100, print.every.n = 10)

cat("Set values ranges for tuning")
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear"))
                        , makeNumericParam("eta", lower = .01 , upper = .3)
                        , makeNumericParam("gamma", lower = 1 , upper = 10)
                        , makeIntegerParam("max_depth",lower = 3,upper = 10)
                        , makeNumericParam("min_child_weight",lower = 1,upper = 10)
                        , makeNumericParam("subsample",lower = 0.5,upper = 1)
                        
                        , makeNumericParam("colsample_bytree",lower = .5,upper = 1))

rdesc <- makeResampleDesc("CV",stratify = T,iters=3L)
ctrl <- makeTuneControlRandom(maxit = 5L)

detach("package:parallel", unload=TRUE)
detach("package:parallelMap", unload=TRUE)
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

print(paste0("Tune hyperparameters ", Sys.time()))
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc
                     , par.set = params, control = ctrl, show.info = T);

lrn_tune <- setHyperPars(lrn, par.vals = mytune$x)
Sa
mytune$x
mytune$y
mytune$learner

#detach("package:caret", unload=TRUE)
print(paste0("Train XGB", Sys.time()))
xgbmodel <- train(lrn_tune, traintask)
xgpred <- predict(xgbmodel,testtask)

confusionMatrix(xgpred$data$response,xgpred$data$truth)