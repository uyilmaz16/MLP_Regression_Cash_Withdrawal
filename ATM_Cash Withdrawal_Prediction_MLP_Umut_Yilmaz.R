library(h2o)
library(lubridate)

X_train <- read.csv("training_data.csv", header = TRUE)
X_test <- read.csv("test_data.csv", header = TRUE)

#addition of a feature which represents the day of the week
X_train$DAY_OF_WEEK <- sapply(X = 1:nrow(X_train), 
                              function(b) wday(as.Date(paste(X_train[b,5], X_train[b,4],X_train[b,3] ,sep="-"))))
X_train <- X_train[,c(1:6,8,7)]

X_test$DAY_OF_WEEK <- sapply(X = 1:nrow(X_test), 
                              function(b) wday(as.Date(paste(X_test[b,5], X_test[b,4],X_test[b,3] ,sep="-"))))

#shuffling data before before validation
set.seed(41)
X_train_s <- X_train[sample(nrow(X_train)),]

#sampling data for validation
M <- 42001
X_val <- X_train_s[M:nrow(X_train_s),]
X_train_s <- X_train_s[-(M:nrow(X_train_s)),]

#initializing for package before modeling
rmse <- c()
h2o.init()
X_train_h <- as.h2o(X_train_s)
X_val_h <- as.h2o(X_val)
X_test_h <- as.h2o(X_test)

#model
mp <- h2o.deeplearning(x = c(1:7), y = 8, training_frame = X_train_h, epochs = 200, hidden = c(100,50,50,50,100), seed = 110, reproducible = TRUE)   

#validation performance
performance = h2o.performance(model = mp)
print(performance)

# validation data prediction 
predictions <- h2o.predict(mp, X_val_h)
predictions.R = as.data.frame(predictions)

#rmse of validation data prediction
rmse <- sqrt(mean(((predictions.R - X_val$TRX_COUNT)[,1])^2))

#test data prediction 
test_predictions <- h2o.predict(mp, X_test_h)
test_predictions.R = as.data.frame(test_predictions)

#setting negatives to zero if there are any
test_predictions.R[which(test_predictions.R[,1]<0),] <- 0

write.table(test_predictions.R, file = "test_predictions.csv", row.names = FALSE, col.names = FALSE)


# KNN model to compare during training MPL
# knn <- knn.reg(as.matrix(X_train_s[,c(1,2,6,7)]), y = as.vector(X_train_s[,8]), k = 30, test = as.matrix((X_val[,c(1,2,6,7)])), )
# prediction <- as.matrix(knn$pred)
# rmse_2 <- sqrt(mean((prediction - X_val$TRX_COUNT)^2))
