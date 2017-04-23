# Practical Machine Learning Project
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
The project needs to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. We should create a report describing how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we did. We will also use our prediction model to predict 20 different test cases.
## Executive Summary  
Using devices such as JawboneUp, NikeFuelBand, and Fitbitit is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.
We will use data from `accelerometers` on the `belt`, `forearm`, `arm`, and `dumbell` of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the [website](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).
## Data
The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). 
The data for this project come from this [source](http://groupware.les.inf.puc-rio.br/har). If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.
## Expected Results
1. Our submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if we submit a repo with a gh-pages branch so the HTML page can be viewed online.
2. We should also apply our machine learning algorithm to the 20 test cases available in the test data above. We will submit our predictions in appropriate format to the programming assignment for automated grading.
## Reproducibility
Due to security concerns with the exchange of R code, our code will not be run during the evaluation by our classmates. We will ensure that if they download the repo, they will be able to view the compiled HTML version of our analysis.
In order to reproduce the same results, we will need a certain set of packages*Note:To install, for instance, the caret package in R, run this command: `install.packages(“caret”)`
The following Libraries were used for this project and load as part of our working environment:
```rlibrary(caret)```*Output:*```## Loading required package: lattice## Loading required package: ggplot2```
```rlibrary(rpart)library(rpart.plot)library(RColorBrewer)library(rattle)```*Output:*```## Rattle: A free graphical interface for data mining with R.## Version 3.1.0 Copyright (c) 2006-2014 Togaware Pty Ltd.## Type 'rattle()' to shake, rattle, and roll your data.```
```rlibrary(randomForest)```*Output:*```## randomForest 4.6-10## Type rfNews() to see new features/changes/bug fixes.```
Finally, load the same seed with the following line of code:
```rset.seed(12345)```
## Data Normalization For Consumption
Environment Variables for Data consumption.
Training data set:
```rtrainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"```Testing data set:
```rtestUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"```Load data to memory:
```rtraining <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))```
## Training Set Partioning
Partioning Training data set into two data sets, 60% for `myTraining`, 40% for `myTesting`:
```rinTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]dim(myTraining); dim(myTesting)```
*Output:* ```## [1] 11776   160``````## [1] 7846  160```
The following transformations were used to normalize the data:
Transformation 1: Cleaning NearZeroVariance Variables
*View possible NZV Variables*
```rmyDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)```
Run this code to create another subset without NZV variables:
```rmyNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt","kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt","max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm","var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm","stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm","kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm","max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm","kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell","skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell","amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm","skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm","max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm","amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm","avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm","stddev_yaw_forearm", "var_yaw_forearm")```
```rmyTraining <- myTraining[!myNZVvars]```
Verify the new NZV variables of observations
```rdim(myTraining)```
*Output:* ```## [1] 11776   100```
Transformation 2: Remove first ID variable to eliminate impact to the Machine Learning Algorithms:
```rmyTraining <- myTraining[c(-1)]```
*Output:* ```## [1] 11776   99```
Transformation 3: Cleaning Variables with many NAs. 
*For Variables that have more than a 60% threshold of NA’s we will leave them out*
```rtrainingV3 <- myTraining #creatingsubset to iterate in loopfor(i in 1:length(myTraining)) { #for every column in the training dataset        if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .6 ) { #if NZV NAs > 60% of total observations        for(j in 1:length(trainingV3)) {            if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:                trainingV3 <- trainingV3[ , -j] #Remove that column            }           }     }}```
Verify the new NZV of observations
```rdim(trainingV3)```
*Output:* ```## [1] 11776    58```
Seting back to our set
```rmyTraining <- trainingV3rm(trainingV3)```
Next Step is execute the same 3 transformations for our `myTesting` and `testing` data sets.
```rclean1 <- colnames(myTraining)clean2 <- colnames(myTraining[, -58]) #classe column removedmyTesting <- myTesting[clean1]testing <- testing[clean2]```
Verify the new NZV of observations
```rdim(myTesting)```
*Output:* ```## [1] 7846   58
```
```rdim(testing)```
*Output:* ```## [1] 20 57```
To ensure proper functioning of Decision Trees and RandomForest Algorithm with the Test data set, we will need to coerce the data into the same type.
```rfor (i in 1:length(testing) ) {        for(j in 1:length(myTraining)) {        if( length( grep(names(myTraining[i]), names(testing)[j]) ) ==1)  {            class(testing[j]) <- class(myTraining[i])        }          }      }```
Ensure Coertion works as expected:
```rtesting <- rbind(myTraining[2, -58] , testing) #note row 2 does not mean anything, this will be removed right.. now:testing <- testing[-1,]```
## Decision Tree: Machine Learning Algorithms for Prediction
```rmodFitA1 <- rpart(classe ~ ., data=myTraining, method="class")```
View the Decision Tree
```rfancyRpartPlot(modFitA1)```
![Decision Tree Plot](./plot/Rplot1.png)
