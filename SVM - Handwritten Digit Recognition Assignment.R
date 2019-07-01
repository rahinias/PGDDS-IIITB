
############################ Assignment - Support Vector Machine ############################
# 1. Business Understanding
# 2. Objective
# 3. Data Understanding
# 4. Data Preparation
# 5. Model Building & Evaluation
#  5.1 Linear Kernel
#   5.1.1 Linear kernel using default parameters
#   5.1.2 Linear kernel using stricter C
#   5.1.3 Using cross validation to optimise C
#  5.2 Radial Kernel
#   5.2.1 Radial kernel using default parameters
#   5.2.2 Redial kernel with higher sigma
#   5.2.3 Using cross validation to optimise C and sigma
####################################################################################

#####################################################################################
########################1. Business Understanding####################################
#####################################################################################
##A classic problem in the field of pattern recognition is that of handwritten digit recognition. 
##Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other 
##digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) 
##written in an image. 

#####################################################################################
##################################2. Objective#######################################
#####################################################################################
##develop a model using Support Vector Machine which should correctly classify the handwritten 
##digits based on the pixel values given as features.

#####################################################################################
##################################3. Data Understanding##############################
#####################################################################################

#Referenced https://en.wikipedia.org/wiki/MNIST_database & http://yann.lecun.com/exdb/mnist/
#The MNIST database of handwritten digits, available from this page, has a training set of 
#60,000 examples (mnist_train), and a test set of 10,000 examples (mnist_test). #It is a subset of a larger set available from NIST.
#The digits have been size-normalized and #centered in a fixed-size image. It is a good database for 
#people who want to try learning techniques and pattern recognition methods on real-world data while 
#spending minimal efforts on preprocessing and formatting.
#mnist_test - considered 785 columns and 10000 rows
#mnist_train - considered 785 columns and 60000 rows
#After setting up the working directory
mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = FALSE, header = FALSE)
mnist_test <- read.csv("mnist_test.csv", stringsAsFactors = FALSE, header = FALSE)

str(mnist_test) 
# All variables are integers, 10,000 observations (rows), 785 variables (columns) imported
str(mnist_train) 
# All variables are integers, 60,000 observations (rows), 785 variables (columns) imported

#Add first column name, change to Label
names(mnist_test)[1] <- "Label"
names(mnist_train)[1] <- "Label"

summary(mnist_test[ , 2:100]) 
summary(mnist_train[ , 2:100]) 
# Some columns contain only 0 values, Maximum Pixel value is 255

#####################################################################################
##################################4. Data Preparation##############################
#####################################################################################

#Install and load required libraries
install.packages("kernlab")
install.packages("ggplot2")
install.packages("caret")
install.packages("caTools")
install.packages("gridExtra")
install.packages("readr")
install.packages("dplyr")
library(kernlab)
library(ggplot2)
library(caret)
library(caTools)
library(gridExtra)
library(readr)
library(dplyr)

##Data Cleansing & Preparation
#Check for NAs
sum(sapply(mnist_test, function(x) sum(is.na(x)))) 
sum(sapply(mnist_train, function(x) sum(is.na(x))))
#Both return 0 indicating no NA Values

str(mnist_test) 
# All variables are integers, 10,000 observations (rows), 785 variables (columns) imported
str(mnist_train) 
# All variables are integers, 60,000 observations (rows), 785 variables (columns) imported
#No headers and footers

#Convert label variable to Factor
mnist_train$label <- factor(mnist_train$label)
summary(mnist_train$label)

mnist_test$label <- factor(mnist_test$label)
summary(mnist_test$label)

# Sampling training dataset as Test set has 10,000 rows and 785 columns while
# Train dataset has 60,000 rows and 785 columns
#Pick 5000 rows i.e., 15% from Train data Set

#To reuse data, set.seed is being used
set.seed(100)
sample_indices <- sample(1: nrow(mnist_train), 9000) 
train <- mnist_train[sample_indices, ]

# Scaling data  - maximum pixel value is 255, using this to scale data
#scaling is a method used to standardize the range of independent variables or features of data. 
#In data processing, it is also known as data normalization and is generally performed during the 
#data preprocessing step.

max(train[ ,2:ncol(train)]) 
train[ , 2:ncol(train)] <- train[ , 2:ncol(train)]/255
test <- cbind(label = mnist_test[ ,1], mnist_test[ , 2:ncol(mnist_test)]/255)

#####################################################################################
########################5. Model Building & Evaluation###############################
#####################################################################################

########################5.1 Linear Kernel


## 5.1.1 Linear kernel using default parameters

model1_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 1)
print(model1_linear) 
eval1_linear <- predict(model1_linear, newdata = test, type = "response")
confusionMatrix(eval1_linear, test$label) 

# Observations:
# Overall accuracy of 92.03%
# Specificities == 99.28%
# Sensitivities == 87.31%

## 5.1.2 Linear kernel using stricter C

model2_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 10)
print(model2_linear) 
eval2_linear <- predict(model2_linear, newdata = test, type = "response")
confusionMatrix(eval2_linear, test$label) 

# Observations:
# Overall accuracy of 91.83%
# Specificities is == 99.11%
# Sensitivities == 87.51%
# Model performance has slightly decreased w.r.t Accuracy and Specificities, 
#perhaps model is overfitting


##5.1.3 Using cross validation to optimise C
# defining range of C
grid_linear <- expand.grid(C= c(0.001, 0.1 ,1 ,10 ,100)) 
fit.linear <- train(label ~ ., data = train, metric = "Accuracy", method = "svmLinear",
                    tuneGrid = grid_linear, preProcess = NULL,
                    trControl = trainControl(method = "cv", number = 5))
# printing results of 5 cross validation
print(fit.linear) 
plot(fit.linear)
# Observations:
# Best accuracy of 91% at C = 0.1
#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was C = 0.1.
eval_cv_linear <- predict(fit.linear, newdata = test)
confusionMatrix(eval_cv_linear, test$label)
# Observations:
# Overall accuracy of 93.1%, slightly imporved
# Specificities == 99.33%
# Sensitivities = 89.99% 
#Improved from model1 by making model more generic i.e. lower C 


########################5.2 Radial Kernel

## 5.2.1 Radial kernel using default parameters

model1_rbf <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "rbfdot", C = 1, kpar = "automatic")
print(model1_rbf) 
eval1_rbf <- predict(model1_rbf, newdata = test, type = "response")
confusionMatrix(eval1_rbf, test$label) 

# Observations:
# Overall accuracy of 95.54%
# Specificities == 99.41%
# Sensitivities == 92.27%
# Increase in overall accuracy and sensitivty from linear kernel using C = 1, sigma = 0.0105571647707048
# data seems to have non linearity to it


##5.2.2 Redial kernel with higher sigma

model2_rbf <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "rbfdot",
                   C = 1, kpar = list(sigma = 1))
print(model2_rbf) 
eval2_rbf <- predict(model2_rbf, newdata = test, type = "response")
confusionMatrix(eval2_rbf, test$label) 

# Observations:
# Accuracy drops to == 15% and class wise results are poor
# sigma = 1 is too much non-linear & model is overfitting


## 5.2.3 Using cross validation to optimise C and sigma

# defining ranges of C and sigma
grid_rbf = expand.grid(C= c(0.01, 0.1, 1, 5, 10), sigma = c(0.001, 0.01, 0.1, 1, 5)) 
# Using only two folds to optimise runtime
fit.rbf <- train(label ~ ., data = train, metric = "Accuracy", method = "svmRadial",tuneGrid = grid_rbf,
                 trControl = trainControl(method = "cv", number = 2), preProcess = NULL)

# printing results of two crossvalidation
print(fit.rbf) 
plot(fit.rbf)

# Observations:
# Best sigma value is 0.01 and c=10
#Resampling: Cross-Validated (2 fold) 
#Summary of sample sizes: 4501, 4499 
#Resampling results across tuning parameters:
  
#  C      sigma  Accuracy   Kappa     
#0.01  0.001  0.1071111  0.00000000
#0.01  0.010  0.4442200  0.37987156
#0.01  0.100  0.1071111  0.00000000
#0.01  1.000  0.1071111  0.00000000
#0.01  5.000  0.1071111  0.00000000
#0.10  0.001  0.7383337  0.70882991
#0.10  0.010  0.9026660  0.89182953
#0.10  0.100  0.2032223  0.10976255
#0.10  1.000  0.1071111  0.00000000
#0.10  5.000  0.1071111  0.00000000
#1.00  0.001  0.8997780  0.88861944
#1.00  0.010  0.9469995  0.94110111
#1.00  0.100  0.8293338  0.81030549
#1.00  1.000  0.1611119  0.06279027
#1.00  5.000  0.1071111  0.00000000
#5.00  0.001  0.9213332  0.91257611
#5.00  0.010  0.9576666  0.95295464
#5.00  0.100  0.8401112  0.82228604
#5.00  1.000  0.1641117  0.06613453
#5.00  5.000  0.1071111  0.00000000
#10.00  0.001  0.9261110  0.91788439
#10.00  0.010  0.9577775  0.95307796
#10.00  0.100  0.8401112  0.82228604
#10.00  1.000  0.1641117  0.06613453
#10.00  5.000  0.1071111  0.00000000

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 0.01 and C = 10.
# Higher sigma values are overfitting and lower sigma values are not capturing non linearity 
# Accuracy increases with C until 5 and then decreases again 
#Further Optimising C 

grid_rbf = expand.grid(C= c(1,2, 3, 4, 5, 6 ,7, 8, 9, 10), sigma = 0.01)
fit.rbf2 <- train(label ~ ., data = train, metric = "Accuracy", method = "svmRadial",tuneGrid = grid_rbf,
                  trControl = trainControl(method = "cv", number = 5), preProcess = NULL)
# printing results of cross validation
print(fit.rbf2) 
plot(fit.rbf2)
eval_cv_rbf <- predict(fit.rbf2, newdata = test)
confusionMatrix(eval_cv_rbf, test$label)
# Observations:

#C   Accuracy   Kappa    
#1  0.9530004  0.9477693
#2  0.9598894  0.9554251
#3  0.9632236  0.9591307
#4  0.9637792  0.9597478
#5  0.9650011  0.9611056
#6  0.9654457  0.9615997
#7  0.9654455  0.9615995
#8  0.9655565  0.9617228
#9  0.9657789  0.9619700
#10  0.9657790  0.9619701

#Tuning parameter 'sigma' was held constant at a value of 0.01
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 0.01 and C = 10.
#Accuracy is 96.35%
#Statistics by Class:
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity            0.9888   0.9903   0.9641   0.9653   0.9735   0.9484   0.9666   0.9514   0.9425  0.9395
#Specificity            0.9959   0.9976   0.9950   0.9948   0.9947   0.9956   0.9981   0.9967   0.9951  0.9960
#Pos Pred Value         0.9632   0.9817   0.9567   0.9540   0.9522   0.9549   0.9820   0.9702   0.9543  0.9634
#Neg Pred Value         0.9988   0.9988   0.9959   0.9961   0.9971   0.9950   0.9965   0.9944   0.9938  0.9932
#Prevalence             0.0980   0.1135   0.1032   0.1010   0.0982   0.0892   0.0958   0.1028   0.0974  0.1009
#Detection Rate         0.0969   0.1124   0.0995   0.0975   0.0956   0.0846   0.0926   0.0978   0.0918  0.0948
#Detection Prevalence   0.1006   0.1145   0.1040   0.1022   0.1004   0.0886   0.0943   0.1008   0.0962  0.0984
#Balanced Accuracy      0.9923   0.9940   0.9796   0.9801   0.9841   0.9720   0.9824   0.9740   0.9688  0.9678

############################ The End ############################

