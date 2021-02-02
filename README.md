# Multi-text-classification
Multi-Class Text Classification (Model Comparision &amp; Selection)

### Assignment: Author Prediction

**Problem Statement** -Given the content, your task is to predict the author.

---
**About Dataset**
>File 1 - content_author_assignment_train.csv

>File 2 - content_author_assignment_test.csv

The train file for any analysis and training
The test file can solely be used for prediction.

Columns - content, author

**Evaluation criteria** -cross entropy loss

### Dataset Description
`Both dataset files are available in folder: data/`

### Utlis Files : 
 > author_prediction.py
 all functions related to classification 
 > nlp_utlis.py
 basic nlp_cleaning functions
 > text_analysis.py
 support files required for EDA
 
 #### Testing

For running training and prediction together: `python3 model.py --opt='classification'`

For training: `python3 model.py --opt=training --file='data/content_author_assignment_train.csv'`

For prediction: `python3 model.py --opt=prediction --file='data/content_author_assignment_test.csv'`

```python

def model_exec(opt,file):
    if opt == "training":
        AuthPredict.training(file)
    elif opt == "prediction":
        AuthPredict.prediction(file)
    else:
        print("Invalid process entered via command line")

if  __name__  ==  "__main__" : 
    #call main function
    model_exec( args . opt,args . file )
   
```
`RESULTS: Training`

--------------------Training Started---------------------

-------------Preprocessing Started--------------
RangeIndex: 712 entries, 0 to 711
Data columns (total 4 columns):
dtypes: int64(1), object(3)
memory usage: 22.4+ KB
None

 Shape of dataframe passed: (712, 4)
-------------Preprocessing Ended--------------

X_train shape:  (569,)

X_test shape:  (143,)

y_train shape:  (569,)

y_test shape:  (143,)

Converting text into vectors..........................

Classifiers used in training:  ['XGBoost Classifier', 'Multinomial Naive Bayes', 'OneVsRest Classifier', 'Random Forest', 'AdaBoost', 'K Nearest Neighbor', 'Stochastic Gradient Descent']

|    | model_name                  |   accuracy_score |   precision_score |   recall_score |   f1_score |   log_loss |
|---:|:----------------------------|-----------------:|------------------:|---------------:|-----------:|-----------:|
|  0 | XGBoost Classifier          |         0.622378 |          0.610936 |       0.572246 |   0.583308 |    1.1072  |
|  6 | Stochastic Gradient Descent |         0.531469 |          0.554818 |       0.462336 |   0.481236 |    1.08219 |
|  3 | Random Forest               |         0.58042  |          0.699655 |       0.43664  |   0.417319 |    1.05376 |
|  2 | OneVsRest Classifier        |         0.559441 |          0.467945 |       0.416984 |   0.402733 |    1.23398 |
|  5 | K Nearest Neighbor          |         0.384615 |          0.389509 |       0.346919 |   0.314551 |    6.43052 |
|  4 | AdaBoost                    |         0.391608 |          0.580093 |       0.295762 |   0.271179 |    1.13501 |
|  1 | Multinomial Naive Bayes     |         0.461538 |          0.250413 |       0.298536 |   0.240056 |    1.61109 |

---------------------Training Complete !!-------------------

`RESULTS: Prediction`

--------------------Testing on unseen data-------------------

Fetching all the predictions..............
 Shape of author_prediction: (855, 3)
Saved the prediction file to current directory
 Prediction completed.....Check file for results
```python
