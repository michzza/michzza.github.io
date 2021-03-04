---
layout: post
title: "ATP linear regression"
categories: journal
tags: [documentation,sample]
image: tennis.jpg
---

In this notebook we are going to explore data from the men’s professional tennis league, which is called the ATP (Association of Tennis Professionals). We are going to familiarize ourselves with the data, analyse it, and build a model that predicts the outcome for a tennis player based on their playing habits.

The data on-hand is about the top 1500 ranked players in the ATP over the span of 2009 to 2017. The statistics recorded for each player in each year include service game (offensive) statistics, return game (defensive) statistics and outcomes. The different fields in the data set are described in the appendix at the end of this notebook.

#### Importing librairies


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

#### Importing the data from 'tennis_stats.csv' in a Pandas dataframe and displaying the first few lines of the data


```python
df = pd.read_csv('tennis_stats.csv')
df.head()
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
      <th>Player</th>
      <th>Year</th>
      <th>FirstServe</th>
      <th>FirstServePointsWon</th>
      <th>FirstServeReturnPointsWon</th>
      <th>SecondServePointsWon</th>
      <th>SecondServeReturnPointsWon</th>
      <th>Aces</th>
      <th>BreakPointsConverted</th>
      <th>BreakPointsFaced</th>
      <th>...</th>
      <th>ReturnGamesWon</th>
      <th>ReturnPointsWon</th>
      <th>ServiceGamesPlayed</th>
      <th>ServiceGamesWon</th>
      <th>TotalPointsWon</th>
      <th>TotalServicePointsWon</th>
      <th>Wins</th>
      <th>Losses</th>
      <th>Winnings</th>
      <th>Ranking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pedro Sousa</td>
      <td>2016</td>
      <td>0.88</td>
      <td>0.50</td>
      <td>0.38</td>
      <td>0.50</td>
      <td>0.39</td>
      <td>0</td>
      <td>0.14</td>
      <td>7</td>
      <td>...</td>
      <td>0.11</td>
      <td>0.38</td>
      <td>8</td>
      <td>0.50</td>
      <td>0.43</td>
      <td>0.50</td>
      <td>1</td>
      <td>2</td>
      <td>39820</td>
      <td>119</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Roman Safiullin</td>
      <td>2017</td>
      <td>0.84</td>
      <td>0.62</td>
      <td>0.26</td>
      <td>0.33</td>
      <td>0.07</td>
      <td>7</td>
      <td>0.00</td>
      <td>7</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.20</td>
      <td>9</td>
      <td>0.67</td>
      <td>0.41</td>
      <td>0.57</td>
      <td>0</td>
      <td>1</td>
      <td>17334</td>
      <td>381</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro Sousa</td>
      <td>2017</td>
      <td>0.83</td>
      <td>0.60</td>
      <td>0.28</td>
      <td>0.53</td>
      <td>0.44</td>
      <td>2</td>
      <td>0.38</td>
      <td>10</td>
      <td>...</td>
      <td>0.16</td>
      <td>0.34</td>
      <td>17</td>
      <td>0.65</td>
      <td>0.45</td>
      <td>0.59</td>
      <td>4</td>
      <td>1</td>
      <td>109827</td>
      <td>119</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rogerio Dutra Silva</td>
      <td>2010</td>
      <td>0.83</td>
      <td>0.64</td>
      <td>0.34</td>
      <td>0.59</td>
      <td>0.33</td>
      <td>2</td>
      <td>0.33</td>
      <td>5</td>
      <td>...</td>
      <td>0.14</td>
      <td>0.34</td>
      <td>15</td>
      <td>0.80</td>
      <td>0.49</td>
      <td>0.63</td>
      <td>0</td>
      <td>0</td>
      <td>9761</td>
      <td>125</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Daniel Gimeno-Traver</td>
      <td>2017</td>
      <td>0.81</td>
      <td>0.54</td>
      <td>0.00</td>
      <td>0.33</td>
      <td>0.33</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.20</td>
      <td>2</td>
      <td>0.50</td>
      <td>0.35</td>
      <td>0.50</td>
      <td>0</td>
      <td>1</td>
      <td>32879</td>
      <td>272</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



#### Analysing the relationship between different features of the data set and the amount of Winnings
Plotting different features against Winnings to identify patterns. There appears to be a strong positive linear relationship between BreakPointsFaced, BreakPointsOpportunities, ReturnGamesPlayed, ServiceGamesPlayed and wins.


```python
plt.figure(figsize = (20, 40))

features = ['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon',
       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted','BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon','ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
       'TotalPointsWon', 'TotalServicePointsWon']

for index in range(len(features)):
    ax = plt.subplot(6, 3, index + 1)
    title = features[index] + " versus Winnings"
    ax.title.set_text(title)
    plt.scatter(df["Winnings"], df[features[index]], alpha = 0.5, color = "#e65c00")

plt.show()

```


    
![png](/assets/img/tennis_linear_regression_files/tennis_linear_regression_6_0.png)
    


Calculating the Pearson Correlation between Winnings and all other features to identify those that have a string linear relationship with Winnings.


```python
features = ['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon',
       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted','BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon','ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
       'TotalPointsWon', 'TotalServicePointsWon']

for index in range(len(features)):
    corr, p = pearsonr(df["Winnings"], df[features[index]])
    if corr > 0.40 or corr< -0.40:
        print(features[index],": " , round(corr , 4))
```

    Aces :  0.7984
    BreakPointsFaced :  0.876
    BreakPointsOpportunities :  0.9004
    DoubleFaults :  0.8547
    ReturnGamesPlayed :  0.9126
    ServiceGamesPlayed :  0.913
    TotalPointsWon :  0.4611
    TotalServicePointsWon :  0.4077


#### Building a single feature linear regression model to predict Winnings

Based on the above, we've identified a strong linear relationship between Winnings and each of the following features: BreakPointsOpportunities, ReturnGamesPlayed and ServiceGamesPlayed.

We are going to build and train a single feature linear regression model using the BreakPointsOpportunities feature to predict Winnings and use sklearn‘s LinearRegression `.score()` method that returns the coefficient of determination R² of the prediction to assess the model's accuracy.


```python
x = df[['BreakPointsOpportunities']]
y = df['Winnings']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

singlefeature_lr = LinearRegression()
singlefeature_lr.fit(x_train, y_train)

print("The coefficient of determination R² of the single feature prediction:")
print(" - on the training data: ", singlefeature_lr.score(x_train, y_train))
print(" - on the test data: ", singlefeature_lr.score(x_test, y_test))
```

    The coefficient of determination R² of the single feature prediction:
     - on the training data:  0.8111412774695739
     - on the test data:  0.8081205523550063



```python
y_predict = singlefeature_lr.predict(x)

plt.scatter(df['BreakPointsOpportunities'], df['Winnings'], alpha = 0.5, color = '#e65c00')
plt.plot(x, y_predict, color = 'Blue')
plt.show()
```


    
![png](/assets/img/tennis_linear_regression_files/tennis_linear_regression_11_0.png)
    


#### Building a two-feature linear regression model to predict Winnings

Based on the above, we've identified a strong linear relationship between Winnings and each of the following features:
> BreakPointsOpportunities <br>
> ReturnGamesPlayed <br>
> ServiceGamesPlayed <br>

We are going to build and train a linear regression model using the ReturnGamesPlayed and the ServiceGamesPlayed features to predict Winnings.


```python
x = df[['BreakPointsOpportunities', 'ServiceGamesPlayed']]
y = df[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

two_features_lr = LinearRegression()
two_features_lr.fit(x_train, y_train)

print("The coefficient of determination R² of the two-feature prediction:")
print(" - on the training data: ", two_features_lr.score(x_train, y_train))
print(" - on the test data: ", two_features_lr.score(x_test, y_test))
```

    The coefficient of determination R² of the two-feature prediction:
     - on the training data:  0.8356666278203743
     - on the test data:  0.8303528220567031


#### Building a multiple-feature linear regression model to predict Winnings

Based on the above, we've identified a strong linear relationship (corr > 40) between Winnings and each of the following features: 

> Aces : corr = 0.7984<br>
> BreakPointsFaced : corr = 0.876<br>
> BreakPointsOpportunities : corr = 0.9004<br>
> DoubleFaults : corr = 0.8547<br>
> ReturnGamesPlayed : corr = 0.9126<br>
> ServiceGamesPlayed : corr = 0.913<br>
> TotalPointsWon : corr = 0.4611<br>
> TotalServicePointsWon : corr = 0.4077<br>



```python
x = df[['Aces', 'BreakPointsFaced', 'BreakPointsOpportunities', 'DoubleFaults', 'ReturnGamesPlayed', 'ServiceGamesPlayed', 'TotalPointsWon', 'TotalServicePointsWon']]
y = df[['Winnings']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mult_features_lr = LinearRegression()
mult_features_lr.fit(x_train, y_train)


print("The coefficient of determination R² of the multi-feature prediction:")
print(" - on the training data: ", mult_features_lr.score(x_train, y_train))
print(" - on the test data: ", mult_features_lr.score(x_test, y_test))
```

    The coefficient of determination R² of the multi-feature prediction:
     - on the training data:  0.8442889601539872
     - on the test data:  0.8290095725876743


### Conclusion

The above shows that we managed to improve the accuracy of the single feature model on the test dataset by adding a second feature. However, using 8 features in our linear regression model cause the accuracy on the test dataset to decrease due to overfitting the model to the training data.  

---

## Appendix
---

The different fields in the data set are described below:

> *Player*: name of the tennis player<br>
> *Year*: year data was recorded<br>

#### Service Game Columns (Offensive)

> *Aces*: number of serves by the player where the receiver does not touch the ball<br>
> *DoubleFaults*: number of times player missed both first and second serve attempts<br>
> *FirstServe*: % of first-serve attempts made<br>
> *FirstServePointsWon*: % of first-serve attempt points won by the player<br>
> *SecondServePointsWon*: % of second-serve attempt points won by the player<br>
> *BreakPointsFaced*: number of times where the receiver could have won service game of the player<br>
> *BreakPointsSaved*: % of the time the player was able to stop the receiver from winning service game when they had the chance<br>
> *ServiceGamesPlayed*: total number of games where the player served<br>
> *ServiceGamesWon*: total number of games where the player served and won<br>
> *TotalServicePointsWon*: % of points in games where the player served that they won<br>

#### Return Game Columns (Defensive)

> *FirstServeReturnPointsWon*: % of opponents first-serve points the player was able to win<br>
> *SecondServeReturnPointsWon*: % of opponents second-serve points the player was able to win<br>
> *BreakPointsOpportunities*: number of times where the player could have won the service game of the opponent<br>
> *BreakPointsConverted*: % of the time the player was able to win their opponent’s service game when they had the chance<br>
> *ReturnGamesPlayed*: total number of games where the player’s opponent served<br>
> *ReturnGamesWon*: total number of games where the player’s opponent served and the player won<br>
> *ReturnPointsWon*: total number of points where the player’s opponent served and the player won<br>
> *TotalPointsWon*: % of points won by the player<br>

#### Outcomes

> *Wins*: number of matches won in a year<br>
> *Losses*: number of matches lost in a year<br>
> *Winnings*: total winnings in USD($) in a year<br>
> *Ranking*: ranking at the end of year<br>