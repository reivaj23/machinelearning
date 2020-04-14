---
title: "Machine Learning Class Project"
author: "Gomez"
date: "4/10/2020"
output: 
  html_document: 
    keep_md: yes
---



## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity. One thing that is regularly quantified is how much of a particular activity is performed, but rarely is quantified how well the activity is done. In this project, the goal was to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

Machine learning techniques are applied to classify the quality of the activities. Prediction with trees, random forest, and boosting techniques were applied to a training dataset, with random forest providing the most acurate classification.

## Data Processing

### Load Datasets

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har].
We first download and load the train and test datasets.


```r
## Download dataset
fileName.train <- "trainData.csv"
if (exists(fileName.train)==FALSE) {
        fileURL.train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"  
        download.file(fileURL.train, destfile = fileName.train)
        downloadDate <- date()
}


fileName.test <- "testData.csv"
if (exists(fileName.test)==FALSE) {
        fileURL.test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(fileURL.test, destfile = fileName.test)
        downloadDate <- date()
}

## Load training data
trainData <- read.csv("./trainData.csv")

## Load testing data
testData <- read.csv("./testData.csv")
```

The datasets consist of 19,622 observations and 160 variables for the train dataset, and 20 observations and 160 variables for the test dataset.
The goal is to predict the "classe" variable, which classifies the manner in which participants performed the exercises.

### Variable selection for models

I decided to use all variables that registered movement in (x,y,z) direction in the belt, arm, dumbbell, and forearm. The search returned the variables for gyros, accel, and magnet.


```r
names(trainData)[grep(c("_belt_|_arm_|_dumbbell_|_forearm_"), colnames(trainData))]
```

```
##  [1] "gyros_belt_x"      "gyros_belt_y"      "gyros_belt_z"     
##  [4] "accel_belt_x"      "accel_belt_y"      "accel_belt_z"     
##  [7] "magnet_belt_x"     "magnet_belt_y"     "magnet_belt_z"    
## [10] "gyros_arm_x"       "gyros_arm_y"       "gyros_arm_z"      
## [13] "accel_arm_x"       "accel_arm_y"       "accel_arm_z"      
## [16] "magnet_arm_x"      "magnet_arm_y"      "magnet_arm_z"     
## [19] "gyros_dumbbell_x"  "gyros_dumbbell_y"  "gyros_dumbbell_z" 
## [22] "accel_dumbbell_x"  "accel_dumbbell_y"  "accel_dumbbell_z" 
## [25] "magnet_dumbbell_x" "magnet_dumbbell_y" "magnet_dumbbell_z"
## [28] "gyros_forearm_x"   "gyros_forearm_y"   "gyros_forearm_z"  
## [31] "accel_forearm_x"   "accel_forearm_y"   "accel_forearm_z"  
## [34] "magnet_forearm_x"  "magnet_forearm_y"  "magnet_forearm_z"
```

## Test Different Machine Learning Techniques

Three different models were created using the tree, random forest, and boosting methods. The prediction was that random forest would give the most accurate results, but also take longer to run.

### Prediction with Trees

```r
## Prediction with tree using "Caret" package
tree <- train(classe ~ gyros_belt_x + gyros_belt_y + gyros_belt_z +
                      accel_belt_x + accel_belt_y + accel_belt_z + 
                      magnet_belt_x + magnet_belt_y + magnet_belt_z + 
                      gyros_arm_x + gyros_arm_y + gyros_arm_z + 
                      accel_arm_x + accel_arm_y + accel_arm_z + 
                      magnet_arm_x + magnet_arm_y + magnet_arm_z + 
                      gyros_dumbbell_x + gyros_dumbbell_y + gyros_dumbbell_z + 
                      accel_dumbbell_x + accel_dumbbell_y + accel_dumbbell_z + 
                      magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
                      gyros_forearm_x + gyros_forearm_y + gyros_forearm_z + 
                      accel_forearm_x + accel_forearm_y + accel_forearm_z + 
                      magnet_forearm_x + magnet_forearm_y + magnet_forearm_z
              , method = "rpart", data = trainData)
```


```r
## Print information from the tree model
print(tree)
```

```
## CART 
## 
## 19622 samples
##    36 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03254522  0.4105513  0.23092230
##   0.05077624  0.3693168  0.16551050
##   0.07278166  0.3342604  0.07819992
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03254522.
```

```r
## Create confusion Matrix for accuracy results
confusionMatrix(trainData$classe, predict(tree, newdata = trainData))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3711    0    0 1863    6
##          B 1436    0    0 2360    1
##          C 1560    0    0 1862    0
##          D  640    0    0 2576    0
##          E  560    0    0 2019 1028
## 
## Overall Statistics
##                                          
##                Accuracy : 0.3728         
##                  95% CI : (0.366, 0.3796)
##     No Information Rate : 0.5443         
##     P-Value [Acc > NIR] : 1              
##                                          
##                   Kappa : 0.2025         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4693       NA       NA   0.2412  0.99324
## Specificity            0.8405   0.8065   0.8256   0.9284  0.86125
## Pos Pred Value         0.6651       NA       NA   0.8010  0.28500
## Neg Pred Value         0.7012       NA       NA   0.5060  0.99956
## Prevalence             0.4030   0.0000   0.0000   0.5443  0.05275
## Detection Rate         0.1891   0.0000   0.0000   0.1313  0.05239
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639  0.18382
## Balanced Accuracy      0.6549       NA       NA   0.5848  0.92724
```

### Random Forest

```r
## Random Forest
rf <- train(classe ~ gyros_belt_x + gyros_belt_y + gyros_belt_z +
                    accel_belt_x + accel_belt_y + accel_belt_z + 
                    magnet_belt_x + magnet_belt_y + magnet_belt_z + 
                    gyros_arm_x + gyros_arm_y + gyros_arm_z + 
                    accel_arm_x + accel_arm_y + accel_arm_z + 
                    magnet_arm_x + magnet_arm_y + magnet_arm_z + 
                    gyros_dumbbell_x + gyros_dumbbell_y + gyros_dumbbell_z + 
                    accel_dumbbell_x + accel_dumbbell_y + accel_dumbbell_z + 
                    magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
                    gyros_forearm_x + gyros_forearm_y + gyros_forearm_z + 
                    accel_forearm_x + accel_forearm_y + accel_forearm_z + 
                    magnet_forearm_x + magnet_forearm_y + magnet_forearm_z,
            data=trainData, method="rf", ntree = 150)
```


```r
## Print information from the tree model
print(rf)
```

```
## Random Forest 
## 
## 19622 samples
##    36 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9859478  0.9822255
##   19    0.9823577  0.9776853
##   36    0.9762720  0.9699848
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
## Create confusion Matrix for accuracy results
confusionMatrix(trainData$classe, predict(rf, newdata = trainData))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```


### Boosting

```r
## Boosting
gbm <- train(classe ~ gyros_belt_x + gyros_belt_y + gyros_belt_z +
                     accel_belt_x + accel_belt_y + accel_belt_z + 
                     magnet_belt_x + magnet_belt_y + magnet_belt_z + 
                     gyros_arm_x + gyros_arm_y + gyros_arm_z + 
                     accel_arm_x + accel_arm_y + accel_arm_z + 
                     magnet_arm_x + magnet_arm_y + magnet_arm_z + 
                     gyros_dumbbell_x + gyros_dumbbell_y + gyros_dumbbell_z + 
                     accel_dumbbell_x + accel_dumbbell_y + accel_dumbbell_z + 
                     magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
                     gyros_forearm_x + gyros_forearm_y + gyros_forearm_z + 
                     accel_forearm_x + accel_forearm_y + accel_forearm_z + 
                     magnet_forearm_x + magnet_forearm_y + magnet_forearm_z,
             method="gbm", data=trainData, verbose=F)
```


```r
## Print information from the gbm model
print(gbm)
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    36 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.6769864  0.5868639
##   1                  100      0.7453439  0.6757089
##   1                  150      0.7784357  0.7182891
##   2                   50      0.7873057  0.7296480
##   2                  100      0.8427156  0.8003116
##   2                  150      0.8702249  0.8353621
##   3                   50      0.8298347  0.7838604
##   3                  100      0.8792506  0.8468418
##   3                  150      0.9046059  0.8791014
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
## Create confusion Matrix for accuracy results
confusionMatrix(trainData$classe, predict(gbm, newdata = trainData))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5421   42   44   70    3
##          B  245 3313  186   30   23
##          C   79  122 3161   48   12
##          D   65   21  210 2869   51
##          E   27   76   72   73 3359
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9236          
##                  95% CI : (0.9198, 0.9273)
##     No Information Rate : 0.2975          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9032          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9287   0.9270   0.8606   0.9285   0.9742
## Specificity            0.9885   0.9698   0.9836   0.9790   0.9847
## Pos Pred Value         0.9715   0.8725   0.9237   0.8921   0.9312
## Neg Pred Value         0.9704   0.9835   0.9684   0.9865   0.9944
## Prevalence             0.2975   0.1821   0.1872   0.1575   0.1757
## Detection Rate         0.2763   0.1688   0.1611   0.1462   0.1712
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9586   0.9484   0.9221   0.9537   0.9794
```

Random forest proved to be the most reliable technique when predicting the "classe" variable. We will use this model to predict "classe" on the testing dataset.

## Using model with testing dataset


```r
## Predict classe values for testing dataset using random forest method
## This method provides the best accuracy on the training dataset
predict(rf, newdata = testData)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The model correctly predicted the 20 cases included in the dataset, as checked on the quiz part of this project.
