---
layout: post
title: "The Legendary Titanic - Part I"
categories: journal
tags: [documentation,sample]
image: titanic.jpg
---

This post is part I of a walkthough of how I built and improved my submission to the Titanic Machine Learning competition on Kaggle. The goal of the competition is to create a machine learning model that predicts which passengers survived the Titanic shipwreck. 

In this post and the next, I will walk through the process of creating a machine learning classification model using the Titanic dataset, which provides various information on the passengers of the Titanic and their survival.

Part I covers data exploration, cleansing and transformation. At the end of this post, we'll have a set of features ready to be fed into our machine learning models.

The dataset can be found on [Kaggle](https://www.kaggle.com/c/titanic/data). It is split into two group, the training set (train.csv) and the test set (test.csv). 

The training set, which is meant to be used to build machine learning models, comes with the outcome (survived or not) for each passenger. The test set has the same features as the training set, apart from the outcome. 

## Loading and exploring the data 
---

```python
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
```

The output of the cell below shows the first few lines of our training set. The dataset contains the following variables : 
- 'PassengerId'- a unique identifier for each passenger
- 'Pclass' - the passenger's class on the ship (1st, 2nd or 3rd)
- 'Name' 
- 'Sex'
- 'Age'
- 'SibSp' - total number of siblings and spouse(s?) on the ship
- 'Parch' - total number of parents and children on the ship
- 'Ticket' - the passenger's ticket number
- 'Fare' 
- 'Cabin'
- 'Embarked' - the port from which the passenger embarked (Cherbourg, Queenstown, Southampton)

as well as the outcome 'Survived'.


```python
train_data = pd.read_csv('train.csv')
train_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data = pd.read_csv('test.csv')
test_data.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Using panda's describe function shows us that 38% of passengers in the training data survived the Titanic. We can also see that the passengers' ages range from 0.4 to 80. We can see some features with missing data, such as "Age", "Cabin" and "Embarked".


```python
train_data.describe(include = "all")
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891</td>
      <td>891</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891</td>
      <td>891.000000</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>891</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>681</td>
      <td>NaN</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sadlier, Mr. Matthew</td>
      <td>male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA. 2343</td>
      <td>NaN</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>577</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>4</td>
      <td>644</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>NaN</td>
      <td>32.204208</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>NaN</td>
      <td>49.693429</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>7.910400</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>14.454200</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>NaN</td>
      <td>512.329200</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Exploring correlation between different features of our data set and "Survival"
---
In this section, we explore every feature in our training set and check whether it is correlated to survival. This will help us decide whether to use it as a feature in our machine learning model, and whether we need to transform it beforehand. 

We will start exploring the gender and age features - it is common knowledge that women and children were evacuated first from the sinking ship.

### Gender
The below bar plot shows that around 75% of women on board survived, whereas only 19% of men did! It looks like women had a lot more chance of surviving the shipwreck, so gender would be a helpful features to predict surival.


```python
women = train_data[train_data['Sex'] == "female"]
perc_women_survived = round(women['Survived'].sum()/len(women)*100)

men = train_data[train_data['Sex'] == "male"]
perc_men_survived = round(men['Survived'].sum()/len(men)*100)

title = f"{perc_women_survived}% of women passengers survived, \nwhereas only {perc_men_survived}% of men did"

sns.set_palette('Pastel1')
sns.set_style('whitegrid')
sns.barplot(data = train_data, x = "Sex", y = "Survived")
plt.title(title)
plt.show()
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_10_0.png)
    


### Age

The figure below shows the spead of age for passengers who survived and those who did not, for both men and women.
We can see that the spread is quite different for men and women. Also, it looks like there are more children in the group who survived, and more older people in the group who did not survive.


```python
ax = sns.violinplot(data = train_data, y = "Survived", x = "Age", hue = "Sex", orient= "h", split = True)
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_12_0.png)
    


### Passenger class

If you've seen the [Titanic](https://en.wikipedia.org/wiki/Titanic_(1997_film)) movie, you know that socioeconomic status played a role in deciding which passengers were given the priority to evacuate the ship. Survival was not only based on gender or age but also class as can be seen on the bar plot below: more than 60% of first class passngers survived the shipwreck, and no more than 25% of third class passengers did.



```python
ax = sns.barplot(data = train_data, x = "Pclass", y = "Survived")
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_14_0.png)
    


### Fare

The price a passenger paid for their ticket is also an indicator of their socioeconomic status. The violin plot below shows that more people paid a higher rate for their ticket in the group who survived. Most fare amounts are on the lower side of the x axis, this is probably due to a small number of outliers.


```python
plt.figure(figsize = (10, 6))
ax = sns.violinplot(data = train_data, y = "Survived", x = "Fare", orient= "h")
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_16_0.png)
    


###  Relatives

The dataset provides information on the number of siblings, spouses, children and parents accompanying each passenger. Let us explore whether the number of relatives a passenger had on board affected their chance of survival. The bar plot shows that passengers accompanied by up to 3 relatives had a higher chance of survival than passengers travelling alone. The chance of survival decreases beyond this point.


```python
relatives = train_data["SibSp"] + train_data["Parch"]

plt.figure(figsize = (12, 6))
ax = sns.barplot(x= relatives, y = train_data["Survived"])
plt.xlabel("Number of relatives")
```




    Text(0.5, 0, 'Number of relatives')




    
![png](\assets\img\titanic-classification_files/titanic-classification_18_1.png)
    


### Port of embarkation

One of the variables provided in the dataset in the port of embarkation "Embarked", which takes one of three values: Cherbourg, Queenstown or Southampton. It looks like passengers who embarked from Cherbourg had a higher rate of survivals.


```python
ax = sns.barplot(data = train_data, x = "Embarked", y = "Survived")
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_20_0.png)
    


So far, we've explored all variables aside from PassengerId, Name, Ticket, and Cabin.

- PassengerId is unique to each passenger and won't be of much use for our predictions.
- Name is also unique to each passenger, however, it also contains the person's title, which could be correlated to the survival. We will later explore this information.
- Ticket
- Cabin field has lots of missing data, however, the cabin number looks like it has the deck information.


## Filling out missing data
---
Let's have a look at the missing data in each of the data sets. It looks like the "Cabin" variable has a huge number of missing entries in both datasets. "Age" comes next with quite a few missing values. We also have a couple of missing values for "Embarked" and "Fare". 

In this section we are going to deal with the missing data, discard the data we don't need, and fill out the missing values with sensible data where relevant.

We are going to create an array containing both data sets, training and test, so we can perform the same operations on both.


```python
pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1).rename(columns = {0 : "Train_data", 1: "Test_data"})
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
      <th>Train_data</th>
      <th>Test_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>177</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>327.0</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_data = [train_data, test_data]
```

### Age

Let's start with age. We are going to compute the mean age for every gender and class combination, and use these values to fill out missing "Age" data - this will be stored in a new column "Age_full".


```python
for data_set in combined_data: 
    mean_age_age_class = data_set.groupby(["Sex", "Pclass"])["Age"].mean()
 
    data_set['Age_full'] = data_set.apply(lambda row : row['Age'] if not pd.isnull(row["Age"]) else   
                                        mean_age_age_class[row['Sex']][row['Pclass']] , axis = 1)
    data_set.drop(['Age'], axis = 1)
    
train_data.head(10)
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_full</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>38.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
      <td>26.507589</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>54.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
      <td>14.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Port of embarkation

We have a couple missing "Embarked" values in the training set. We are going to fill them out with the most common value, which happens to be "S" for Southampton.


```python
for data_set in combined_data: 
   embarked_mode = data_set['Embarked'].mode()
   data_set['Embarked'] = data_set['Embarked'].fillna(embarked_mode[0])
```

### Fare

We have one missing "Fare" value in the test set for a third class passenger who embarked at Southampton, which we are going to replace with the average fare for third class passengers who embarked there in the test set.


```python
mean_fare_port_class = test_data.groupby(["Embarked", "Pclass"])["Fare"].mean()

test_data['Fare'] = test_data['Fare'].fillna(mean_fare_port_class["S"][3])

```

### Cabin

As mentioned above, the "Cabin" variable has tons of missing values. We could completely drop this feature, however, it looks like cabin numbers have a letter which could be the deck number, or particular sections of the ship. Let's extract this info and store it in a new "Deck" variable.


```python
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8, "T": 9}

for data_set in combined_data:
    data_set['Cabin'] = data_set['Cabin'].fillna("U0")
    data_set['Deck'] = data_set['Cabin'].apply(lambda x: x[0])
    
    data_set['Deck'] = data_set['Deck'].map(deck)
    data_set['Deck'] = data_set['Deck'].astype(int)
```

## Transforming variables
---
Now that we are done dealing with missing data, we have 2 new columns: "Age_full" which replaces "Age", and "Deck" that we've extracted from the "Cabin" data. 

In the following section, we are going to transform the features we are using for our model into integer variables to feed into our models. 

Numerical variables such as "Age" and "Fare" will be transformed into categorical variables by splitting the data in groups/intervals. We will then transform all our categorical variables into integer variables.

### Age 

Let's start by splitting passengers into age groups and see the survival rate of different age groups. It looks like children (0-18) is the age group with the highest survival rate, whereas 70+ year olds have a survival rate around 15%.


```python
for data_set in combined_data:
    data_set['Age_group'] = data_set['Age_full'].map(lambda x: 0 if (x >= 0) & (x < 18) else 
                                                 (1 if (x >= 18) & (x < 30) else (
                                                  2 if (x >= 30) & (x < 40) else (
                                                  3 if (x >= 40) & (x < 50) else (
                                                  4 if (x >= 50) & (x < 60) else (
                                                  5 if (x >= 60) & (x < 70) else (
                                                  6 if x >= 70 else "NaN")))))))
```


```python
df = pd.DataFrame((train_data.groupby(["Age_group"])["Survived"].sum())*100/(train_data.groupby(["Age_group"])["Survived"].count()))

df['Not Survived'] = df["Survived"].apply(lambda x : 100 - x)

ax = plt.subplot()
plt.bar(range(len(df)), df["Survived"], label = "Survived")
plt.bar(range(len(df)), df["Not Survived"], bottom = df["Survived"], label = "Not Survived", alpha = 0.3)
plt.legend()
ax.set_xticks(range(0, 7))
ax.set_xticklabels(['0-18', '18-30', '30-40', '40-50', '50-60', '60-70', '70+'])
ax.set_yticks(range(0, 105, 10))
ax.set_yticklabels(str(x) + "%" for x in range(0, 105, 10))
plt.title('Percentage of survival in each age group')
plt.xlabel('Age groups')
plt.show()
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_36_0.png)
    


### Rate 

We will now create a categorical variable out of the "Rate" variable. Using Pandas `qcut`, we'll split rate values into 4 rate categories, each corresponding to a quartile of the variable. 
The bar plot shows that the higher the rate category (and hence the rate), the higher the survival rate.


```python
fare_labels = [0, 1, 2, 3]

for data_set in combined_data:
    data_set['Fare_cat'] = pd.qcut(data_set['Fare'], 4, fare_labels)
    data_set['Fare_cat'] = data_set['Fare_cat'].astype(int)
    
train_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_full</th>
      <th>Deck</th>
      <th>Age_group</th>
      <th>Fare_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>U0</td>
      <td>S</td>
      <td>22.0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>38.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>U0</td>
      <td>S</td>
      <td>26.0</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>35.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>U0</td>
      <td>S</td>
      <td>35.0</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame((train_data.groupby(["Fare_cat"])["Survived"].sum())*100/(train_data.groupby(["Fare_cat"])["Survived"].count()))

df['Not Survived'] = df["Survived"].apply(lambda x : 100 - x)

ax = plt.subplot()
plt.bar(range(len(df)), df["Survived"], label = "Survived")
plt.bar(range(len(df)), df["Not Survived"], bottom = df["Survived"], label = "Not Survived", alpha = 0.3)
plt.legend()
ax.set_xticks(range(0, 4))
ax.set_xticklabels(['Cat1', 'Cat2', 'Cat3', 'Cat4'])
ax.set_yticks(range(0, 105, 10))
ax.set_yticklabels(str(x) + "%" for x in range(0, 105, 10))
plt.title('Percentage of survival in each fare category')
plt.xlabel('Fare categories')
plt.show()
```


    
![png](\assets\img\titanic-classification_files/titanic-classification_39_0.png)
    


### Names and titles

We have not explored yet the "Name" feature. As mentioned earlier, a passenger's name is a unique value for each passenger and is hence not likely to give any indications about their survival. However, if we look closer, this variable also holds the passenger's title. 

The value counts of the extracted titles show that the most common titles are "Mr", "Miss", "Mrs" and Master". Other titles are quite rare and will be all grouped under one fifth category.


```python
for data_set in combined_data:
    data_set['Title'] = data_set['Name'].str.extract(' ([A-Za-z]+)\.')

df = pd.concat([train_data, test_data])

pd.DataFrame(df['Title'].value_counts()).reset_index()
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
      <th>index</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mr</td>
      <td>757</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miss</td>
      <td>260</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mrs</td>
      <td>197</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Master</td>
      <td>61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rev</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dr</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Col</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ms</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Major</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Mlle</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Jonkheer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Countess</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Dona</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Capt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Don</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sir</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Lady</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Mme</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


for data_set in combined_data:
    data_set['Title'] = data_set['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Dona', 'Jonkheer'], 'Rare')
    data_set['Title'] = data_set['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')
 
    data_set['Title'] = data_set['Title'].map(titles)
    
    data_set['Title'] = data_set['Title'].fillna(0)
    data_set['Title'] = data_set['Title'].astype(int)
    
    data_set = data_set.drop(['Name'], axis=1)

```

### Relatives, gender and ports of embarkation

In the below we create a new column "Relatives" which represents the total number of siblings, spouses, parents and children accompagniying a passenger, as well as a variable "Alone" which has either a value of 1 if the passenger had no relatives accompagnying them, or 0 if they did.
We also map string values in columns "Sex" and "Embarked" to numerical values. 


```python
for data_set in combined_data:
    data_set['Relatives'] = data_set['SibSp'] + data_set['Parch']
    data_set['Alone'] = data_set['Relatives'].apply(lambda x : 1 if x == 0 else 0) 
```


```python
gender = {'female' : 1, 'male': 0}

for data_set in combined_data:
    data_set['Sex'] = data_set['Sex'].map(gender)
```


```python
ports = {'S' : 0, 'C': 1, 'Q' : 2}

for data_set in combined_data:
    data_set['Embarked'] = data_set['Embarked'].map(ports)
    data_set['Embarked'] = data_set['Embarked'].astype(int)
    
train_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_full</th>
      <th>Deck</th>
      <th>Age_group</th>
      <th>Fare_cat</th>
      <th>Title</th>
      <th>Relatives</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>U0</td>
      <td>0</td>
      <td>22.0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>1</td>
      <td>38.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>U0</td>
      <td>0</td>
      <td>26.0</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>0</td>
      <td>35.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>U0</td>
      <td>0</td>
      <td>35.0</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We are now ready to build our models using the following features which all have integer values 'Sex', 'Title', 'Age_group', 'Fare_cat', 'Pclass', 'Embarked', 'Deck', 'SibSp', 'Parch', 'Relatives' and 'Alone'. Stay tuned for part II of the Legendary Titanic where we'll built different models and create a submission.
