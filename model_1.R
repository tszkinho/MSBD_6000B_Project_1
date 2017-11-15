require(rio)
require(e1071)
require(caret)
require(xgboost)
require(lattice)
set.seed(20392904)
train_d<-import("traindata.csv")
train_l<-import("trainlabel.csv")
test_d<-import("testdata.csv")
#str(train_d)
#summary(train_d)
table(train_l)
maxminnorm <- function(x)   {
  (x - min(x, na.rm=TRUE))/(max(x,na.rm=TRUE) - min(x, na.rm=TRUE))} 
train_dlog<-rbind(train_d,test_d)
#train_dlog[,58]<-train_dlog[,57]*train_dlog[,55]
#train_dlog[,55:57]<-log(train_dlog[,55:57])
train_transform<-as.data.frame(lapply(train_dlog,scale) )
train_d1<-train_transform[1:nrow(train_d),]
testd1<-train_transform[-(1:nrow(train_d)),]
test_ac<-matrix(0,nc=5,nr=10)

for ( i in 1:10){
  print(paste("Loop:",i))
train_ind <- sample(seq_len(nrow(train_d1)), size = floor(0.85 * nrow(train_d1)))

train_d0<-train_d1[train_ind,]
valid_d0<-train_d1[-train_ind,]
train_l0<-train_l[train_ind,]
valid_l0<-train_l[-train_ind,]



param <- list(max_depth = 5, eta = 0.03, lambda=2,
              objective = "binary:logistic", eval_metric = "logloss",silence=1)
xgb_model<-xgboost(data= xgb.DMatrix(data=as.matrix(train_d0),label=train_l0),params=param,nrounds=500,verbose = 0)
xgb_model_valid<-predict(xgb_model,xgb.DMatrix(data=as.matrix(valid_d0)))
xgb_model_valid[xgb_model_valid<0.5]<-0
xgb_model_valid[xgb_model_valid>=0.5]<-1
#confusionMatrix(xgb_model_valid,valid_l0)

test_ac[i,1]<-mean(xgb_model_valid==valid_l0)

svm_model_rbf<-svm(train_d0,train_l0,kernel="radial",type="C-classification",scale = FALSE,cost=1,gamma=0.03)
#confusionMatrix(predict(svm_model_rbf,valid_d0),valid_l0)
test_ac[i,2]<-mean(predict(svm_model_rbf,valid_d0)==valid_l0)
svm_model_poly<-svm(train_d0,train_l0,kernel="polynomial",type="C-classification",scale = FALSE,cost=1,gamma=0.1,degree=2,coef0=100)
#confusionMatrix(predict(svm_model_poly,valid_d0),valid_l0)
test_ac[i,3]<-mean(predict(svm_model_poly,valid_d0)==valid_l0)
svm_model_lin<-svm(train_d0,train_l0,kernel="linear",type="C-classification",scale = FALSE)
#confusionMatrix(predict(svm_model_lin,valid_d0),valid_l0)
test_ac[i,4]<-mean(predict(svm_model_lin,valid_d0)==valid_l0)
svm_model_sigmoid<-svm(train_d0,train_l0,kernel="sigmoid",type="C-classification",scale = FALSE,cost=1,gamma=0.005,coef0=0)
#confusionMatrix(predict(svm_model_sigmoid,valid_d0),valid_l0)
test_ac[i,5]<-mean(predict(svm_model_sigmoid,valid_d0)==valid_l0)
}
colnames(test_ac)<-c("xgboost","svm_rbf","svm_poly","svm_linear","svm_sigmoid")
print(summary(test_ac))


plot11<-histogram(test_ac[,1],main=colnames(test_ac)[1])
plot12<-histogram(test_ac[,2],main=colnames(test_ac)[2])
plot13<-histogram(test_ac[,3],main=colnames(test_ac)[3])
plot21<-histogram(test_ac[,4],main=colnames(test_ac)[4])
plot22<-histogram(test_ac[,5],main=colnames(test_ac)[5])
print(plot11,split=c(1,1,3,2),more=TRUE)
print(plot12,split=c(1,2,3,2),more=TRUE)
print(plot12,split=c(3,1,3,2),more=TRUE)
print(plot21,split=c(2,1,3,2),more=TRUE)
print(plot22,split=c(2,2,3,2),more=TRUE)

xgb_model_final<-xgboost(data= xgb.DMatrix(data=as.matrix(rbind(train_d0,valid_d0)),label=c(train_l0,valid_l0)),params=param,nrounds=500,verbose = 0)
xgb_model_test<-predict(xgb_model_final,xgb.DMatrix(data=as.matrix(testd1)))
xgb_model_test[xgb_model_valid<0.5]<-0
xgb_model_test[xgb_model_valid>=0.5]<-1
export(data.frame(xgb_model_test),"project1_20392904.csv", col.names = FALSE)
