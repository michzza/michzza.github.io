---
layout: post
title: "Did Jack and Rose survive?"
categories: journal
tags: [documentation,sample]
image: jack-and-rose.jpg
---

This post is part II of a walkthough of how I built and improved my submission to the Titanic Machine Learning competition on Kaggle. The goal of the competition is to create a machine learning model that predicts which passengers survived the Titanic shipwreck. 

Part I covered data exploration, cleansing and transformation. At the end of my last post, we had a set of features ready to be fed into our machine learning models.. 

We are going to build models based on several classification algorithms and fit and test them on our training data set. Based on the models' scores on the training set, we will select the most performing model and tune it's hyperparameters.

We are going to explore the following classification algorithms - using scikit-learn, we will build classification models based on each of the below algorithms and compute an average cross validation score for each one.

- Logistic Regression
- K Nearest Neighbors
- Support Vector Machine
- Perceptron
- Stochastic Gradient Descent Classifier
- Decision Tree
- Random Forest

Let's first have a look at our features.

```python
features = ['Sex', 'Title', 'Age_group', 'Fare_cat', 'Pclass', 'Embarked', 'Deck', 'SibSp', 'Parch', 'Relatives','Alone']
train_data[features][:5]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Title</th>
      <th>Age_group</th>
      <th>Fare_cat</th>
      <th>Pclass</th>
      <th>Embarked</th>
      <th>Deck</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Relatives</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
y = train_data["Survived"]

features = ['Sex', 'Title', 'Age_group', 'Fare_cat', 'Pclass', 'Embarked', 'Deck', 'SibSp', 'Parch', 'Relatives','Alone']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

scores = {}
cross_val_scores = {}
```

### Data standardization 

Some of these algorithms - such as K Nearest Neighbors and Support Vector Machine - rely on the distance between different data points in their classification process. For these algorithms, it is best to standardize all our features so they are on a similar scale. It is worth noting that our features are already on a relatively similar scale, as they are all integers ranging from 0 to 8!


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
scaled_X = sc.fit_transform(X)
scaled_X_test = sc.transform(X_test)
```

### Building models 

We will now go through all of the algorithms above and build and fit models with our training data. For each model we will compute the model's score on the training data, as well as an average cross-validation score in order to identify the best performing model.


```python
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = LogisticRegression()
model.fit(X, y)

scores['Logistic Regression'] = model.score(X, y)
cross_val_scores['Logistic Regression'] = cross_val_score(model, X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 4)
model.fit(scaled_X, y)

scores['K Nearest Neighbors'] = model.score(scaled_X, y)
cross_val_scores['K Nearest Neighbors'] = cross_val_score(model, scaled_X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn.svm import SVC

model = SVC()
model.fit(scaled_X, y)

scores['Support Vector Machine'] = model.score(scaled_X, y)
cross_val_scores['Support Vector Machine'] = cross_val_score(model, scaled_X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn.linear_model import Perceptron


model = Perceptron(max_iter=25)
model.fit(X, y)

scores['Perceptron'] = model.score(X, y)
cross_val_scores['Perceptron'] = cross_val_score(model, X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

sgd = linear_model.SGDClassifier(max_iter=15, tol=None)
sgd.fit(X, y)

scores['Stochastic Gradient Descent'] = model.score(X, y)
cross_val_scores['Stochastic Gradient Descent'] = cross_val_score(model, X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier() 
model.fit(X, y)

scores['Decision Tree'] = model.score(X, y)
cross_val_scores['Decision Tree'] = cross_val_score(model, X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

scores['Random Forest'] = model.score(X, y)
cross_val_scores['Random Forest'] = cross_val_score(model, X, y, cv=10, scoring = "accuracy").mean()
```


```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X, y)

scores['Gaussian Naive Bayes'] = model.score(X, y)
cross_val_scores['Gaussian Naive Bayes'] = cross_val_score(model, X, y, cv=10, scoring = "accuracy").mean()
```


```python
cross_val_scores = dict(sorted(cross_val_scores.items(), key=lambda item: item[1], reverse = True))
cross_val_scores_df = pd.DataFrame.from_dict(cross_val_scores, orient='index').reset_index().rename(columns = {"index" : "Model", 0 : "Cross Val Score"})

scores_df = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(columns = {"index" : "Model", 0 : "Score"})

cross_val_scores_df = cross_val_scores_df.merge(scores_df)
cross_val_scores_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Cross Val Score</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Support Vector Machine</td>
      <td>0.829401</td>
      <td>0.845118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest</td>
      <td>0.821610</td>
      <td>0.921437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K Nearest Neighbors</td>
      <td>0.821536</td>
      <td>0.854097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression</td>
      <td>0.809213</td>
      <td>0.817059</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decision Tree</td>
      <td>0.800275</td>
      <td>0.921437</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.789001</td>
      <td>0.790123</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Perceptron</td>
      <td>0.708202</td>
      <td>0.802469</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stochastic Gradient Descent</td>
      <td>0.708202</td>
      <td>0.802469</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (16, 5))
sns.barplot(data = cross_val_scores_df, x = "Model", y = "Cross Val Score")
plt.axis([-0.5, 7.5, 0.69, 0.84])
plt.show()
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_63_0.png)
    


The SVM, Random Forest and K Nearest Neighbors models seem to outperform other models we used both in terms of score using a single training set and using cross validation. 

## Tuning hyper parameters
In the following section we will explore these 3 models and try to improve them by tuning their hyper parameters. We will then submit our predictions for each improved model and note how they score on Kaggle.

### Random Forest Classifier 

We will use scikit-learn's Grid Search CV to perform a search to find the best parameters values for our model. 

```python
from sklearn.model_selection import GridSearchCV

param_grid = { "criterion" : ["gini", "entropy"], 
              "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
              "min_samples_split" : [4, 12, 25], 
              "n_estimators": [100, 400, 700, 1000],
              "max_features" : [3, 5, 9, 10]}

model = RandomForestClassifier(n_estimators=100)
gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
gridsearch.fit(X, y)
gridsearch.best_params_

```


    {'criterion': 'gini',
    'max_features': 9,
    'min_samples_leaf': 1,
    'min_samples_split': 12,
    'n_estimators': 700}

   

We will now use the features returned by the search to build our model and then fit it to our training data. The Random Forst model with tuned hyper parameters has an average cross validation score of 0.824 on the training data.


```python
rf_model = RandomForestClassifier(criterion =  'gini', max_features =  9, min_samples_leaf = 1, min_samples_split = 12,
 n_estimators = 700)
rf_model.fit(X, y)

print("Score: ", round(rf_model.score(X, y), 3))
print("Average cross validation score: ", round(cross_val_score(rf_model, X, y, cv=10, scoring = "accuracy").mean(), 3))

```

    Score:  0.884
    Average cross validation score:  0.824


We predict the outputs for our test data and submit it to the competition on Kaggle. This model score 0.775 on the test data.


```python
predictions = rf_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```

### Support Vector Machine

We will now perform a search to find the best parameter values for our Support Vector Machine model. 


```python
from sklearn.model_selection import GridSearchCV

param_grid = { "kernel" : ["linear", "poly", "rbf", "sigmoid"], 
              "C" : [0.1, 1, 10, 50, 100], 
              "gamma" : ["scale", "auto"]}

model = SVC()
gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
gridsearch.fit(scaled_X, y)
gridsearch.best_params_
```




    {'C': 1, 'gamma': 'scale', 'kernel': 'poly'}



The SVM model with tuned hyper parameters has an average cross-validation sore of 0.828 on the training data and scored 0.787 on the test data when submitted to the competition on Kaggle.


```python
svc_model = SVC(kernel = 'poly', gamma = "scale", C = 1)
svc_model.fit(scaled_X, y)

print("Score: ", round(svc_model.score(scaled_X, y), 3))
print("Average cross validation score: ", round(cross_val_score(svc_model, scaled_X, y, cv=10, scoring = "accuracy").mean(), 3))

predictions = svc_model.predict(scaled_X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```

    Score:  0.85
    Average cross validation score:  0.828


### K Nearest Neighbors Classifier


```python
from sklearn.model_selection import GridSearchCV

param_grid = { "n_neighbors" : [3, 5, 8, 10], 
              "weights" : ["uniform", "distance"], 
              "algorithm"  :["auto", "ball_tree", "kd_tree", "brute"]}

model = KNeighborsClassifier(n_neighbors = 4)
gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
gridsearch.fit(scaled_X, y)
gridsearch.best_params_
```




    {'algorithm': 'brute', 'n_neighbors': 10, 'weights': 'uniform'}



The K Nearest Neighbors model with tuned hyper parameters has an average cross-validation sore of 0.835 on the training data and scored 0.769 on the test data when submitted to the competition on Kaggle.


```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 10, algorithm = 'brute', weights = 'uniform' )
knn_model.fit(scaled_X, y)

print("Score: ", round(knn_model.score(scaled_X, y), 3))
print("Average cross validation score: ", round(cross_val_score(knn_model, scaled_X, y, cv=10, scoring = "accuracy").mean(), 3))

predictions = knn_model.predict(scaled_X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```

    Score:  0.852
    Average cross validation score:  0.835


### Woud Jack and Rose have survived? 
Let's see what our models have to say on Jack and Rose's survival. Below we built a dataframe with Jack and Rose's features. For example, Jack's fare category is 0 because he was a 'poor' artist, according to wikipedia, Rose was in first class because if her family's upper-class status...


```python
Jack_and_Rose = pd.DataFrame({"Sex" : [0, 1],   
                              "Title": [1, 2],  
                              "Age_group": [0, 0],  
                              "Fare_cat": [0, 2],    
                              "Pclass": [3, 1], 
                              "Embarked": [0, 0], 
                              "Deck": [8, 1], 
                              "SibSp": [0, 1], 
                              "Parch": [0, 1], 
                              "Relatives": [0, 2], 
                              "Alone": [1, 0]})

print(rf_model.predict(Jack_and_Rose))
print(knn_model.predict(Jack_and_Rose))
print(svc_model.predict(Jack_and_Rose))

```

    [0 1]
    [0 1]
    [1 1]


Both our Random Forest and K Nearest Neighbors predicted that Rose survived, and Jack did not. Our Support Vector Machine is more optimistic and predicted both survived! 


