
# Module 4 Project: Applying Classification Modeling

## Scope of the project:

For this module's project, we have three days to answer a classification data science question using multiple models and present the results of the project. 
We will utilize all of the different tools we have learned over this course: data cleaning, EDA, feature engineering/transformation,
feature selection, hyperparameter tuning, and model evaluation. This culminates in a four minute presentation to explain our project and findings. 


## Prediction: 

Predicting Diabetes in Women of the Pima Indigenous People.

## Data Set Info: 

The data set came from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database). Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima heritage.

## Dataset Attributes:

Pregnancies:  Number of times pregnant

Glucose:  Plasma glucose concentration a 2 hours in an oral glucose tolerance test 

Blood Pressure:  Diastolic blood pressure (mm Hg) 

Skin Thickness:  Triceps skin fold thickness (mm) 

Insulin:  2-Hour serum insulin (mu U/ml) 

BMI:  Body mass index (weight in kg/(height in m)^2) 

[Diabetes Pedigree Function](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf):  measure of genetic influence and hereditary risk 

Age:  Age (years) 

Outcome:  Class variable (0 or 1)


## Research Key Points:

The data came from a longitudinal study started in 1965. 

Pima population has the world’s highest prevalence of diabetes. 

50% of Pima people will have diabetes by the age of 35.

The Pima have Type 2 diabetes which is caused by genetics and lifestyle.

15% of diabetic Pima people develop end stage renal cancer after living with 
diabetes for 20 years.

Their way of life was centered on the river, which is considered holy. 

Their rivers are currently dry, due to the upstream dams that block the flow and 
the diversion of water by non-pima farmers. 

They engaged in a century long legal battle with the US which they eventually 
won, but the river is still dry.

The diversion of the water and the introduction of non-native diet is said to 
have been the leading contributing factor in the high rate of diabetes among the tribe.

[Pima Film](https://www.youtube.com/watch?v=RZfA0ucMCGA&list=PLCi5KPiuty6hBQ3WPmtC2ObV-gp2IfZoZ&index=4) 

## Data Cleaning

Dropped the outliers first:

- Insulin > 600

- Skin thickness > 70  

- BMI > 55


Dealing with missing data:

- Dropped data with missing Glucose levels.
 
- Replaced BMI, Blood Pressure, and Insulin with their Median values.

- Skin Thickness and BMI have a linear relationship. Missing skin thickness measures were replaced by mean of skin thickness within a certain BMI range.

![Jointplot Skin & BMI](https://user-images.githubusercontent.com/47832231/60525563-963ee600-9cbc-11e9-9c2f-33660baeb47e.png)

    BinBMI
    
    (10, 20]     9.642857
    
    (20, 30]    14.363309
    
    (30, 40]    23.373684
    
    (40, 50]    28.261364
    
    (50, 60]    32.714286
    
## Exploratory Data Analysis

![Graphs](https://user-images.githubusercontent.com/47832231/60469276-1f590d00-9c2a-11e9-96d2-6740aff78f5c.png)


Logged Insulin and DPF.

![Log Graphs](https://user-images.githubusercontent.com/47832231/60469405-8080e080-9c2a-11e9-8553-76c1521ec643.png)


logging insulin did not improve the distribution. 

## Feature Engineering

After researching healthy levels of biometric measurements, I created binary features to determine if they are within healthy ranges,
then added binary data together to provide a health score. 

- Healthy Number of Pregnancies < 5
- Healthy Glucose Level <= 100
- Healthy Blood Pressure <= 120
- Healthy Insulin Level <= 100
- Healthy BMI (18.5 - 24.9)
- Overall Health Score 

Binned Pregnancies and Age

![preg graph](https://user-images.githubusercontent.com/47832231/60469782-f89bd600-9c2b-11e9-9156-2a3ffd3c56f4.png)
![age graph](https://user-images.githubusercontent.com/47832231/60469841-2254fd00-9c2c-11e9-95e2-af8e7c4d23e5.png)

Using sklearn Polynomial Features, I generated interaction features. 

## Feature Selection 
Using logistic regression, features were tested in combinations to determine if they would improve the F1 score. 

The binned data did not improve the model. 

The interaction features improved the model, but when the new "healthy" features were added the F1 score dropped.

L1 Regularization (Lasso) was used to remove unnecessary features. 

The resulting features are: 
    
    Index(['glucose', 'pregnancies skinthickness', 'pregnancies bmi', 'glucose^2',
       'glucose bmi', 'glucose healthy_bmi', 'bloodpressure healthy_bmi',
       'skinthickness dpf_log', 'insulin healthy_bmi', 'insulin healthy_ins',
       'bmi age', 'age healthy_preg'],

## Class Imbalance

Using resampling, the training data outcomes were balanced evenly.

![class imbal](https://user-images.githubusercontent.com/47832231/60516548-fe390080-9cab-11e9-85c6-9a7fa671a70f.png)

## Models

### KNN 

Using a for loop function, best K = 46.

       [[74 21]
        [10 46]]


                  precision    recall  f1-score   support

               0       0.88      0.78      0.83        95
               1       0.69      0.82      0.75        56

       micro avg       0.79      0.79      0.79       151
       macro avg       0.78      0.80      0.79       151
    weighted avg       0.81      0.79      0.80       151


![knn](https://user-images.githubusercontent.com/47832231/60522726-88d32d00-9cb7-11e9-8a81-c63dc36b04b6.png)

### Decision Tree

   Gridsearch gave the hyperparamters:  criterion= 'gini', max_depth= 2, min_samples_leaf= 110
  
     [[69 26]
      [15 41]]

                     precision    recall  f1-score   support

                  0       0.82      0.73      0.77        95
                  1       0.61      0.73      0.67        56

          micro avg       0.73      0.73      0.73       151
          macro avg       0.72      0.73      0.72       151
       weighted avg       0.74      0.73      0.73       151

  
![dtree](https://user-images.githubusercontent.com/47832231/60523007-f67f5900-9cb7-11e9-8f90-99a692e1cd05.png)


### Random Forest

Gridsearch gave the hyperparamters:  criterion= gini, max_depth= 10, min_samples_leaf=  5, 'min_samples_split': 20, 'n_estimators': 300

     [[68 27]
      [ 9 47]]
                      
                       precision    recall  f1-score   support

                    0       0.88      0.72      0.79        95
                    1       0.64      0.84      0.72        56

            micro avg       0.76      0.76      0.76       151
            macro avg       0.76      0.78      0.76       151
         weighted avg       0.79      0.76      0.77       151
         
### XGBoost

Max depth = 3

    [[71 24]
     [12 44]]
                       precision    recall  f1-score   support

                   0       0.86      0.75      0.80        95
                   1       0.65      0.79      0.71        56

           micro avg       0.76      0.76      0.76       151
           macro avg       0.75      0.77      0.75       151
        weighted avg       0.78      0.76      0.77       151
        
## Results

The primary metric used to predict model strength is the F1 score which is the harmonic average of precision and recall. 

The secondary metric used is Recall, because it would be more harmful to fail to recognize whether someone is at risk for diabetes then it would be to falsely diagnose them.

KNN is the strongest model with a weighted F1 = .80 and Recall = .79. 

