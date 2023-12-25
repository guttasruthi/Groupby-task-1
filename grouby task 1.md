```python
import pandas as pd,seaborn as sns,sklearn

```


```python
df = pd.read_csv(r"C:\Users\HP\Downloads\groupbyExcercise - groupbyExcercise.csv")
```


```python
df
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
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Product</th>
      <th>Revenue</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2023-01-01</td>
      <td>A</td>
      <td>221</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2023-01-01</td>
      <td>B</td>
      <td>111</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2023-01-02</td>
      <td>B</td>
      <td>171</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2023-01-02</td>
      <td>A</td>
      <td>141</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2023-02-01</td>
      <td>A</td>
      <td>75</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2023-02-01</td>
      <td>B</td>
      <td>90</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2023-02-02</td>
      <td>A</td>
      <td>96</td>
      <td>17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2023-02-02</td>
      <td>B</td>
      <td>170</td>
      <td>10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>2023-03-01</td>
      <td>B</td>
      <td>200</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>2023-03-01</td>
      <td>A</td>
      <td>120</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>2023-03-02</td>
      <td>A</td>
      <td>121</td>
      <td>12</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>2023-03-02</td>
      <td>B</td>
      <td>150</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12 entries, 0 to 11
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   Unnamed: 0  12 non-null     int64 
     1   Date        12 non-null     object
     2   Product     12 non-null     object
     3   Revenue     12 non-null     int64 
     4   Quantity    12 non-null     int64 
    dtypes: int64(3), object(2)
    memory usage: 612.0+ bytes
    


```python
df.describe(include='all').T
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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.5</td>
      <td>3.605551</td>
      <td>0.0</td>
      <td>2.75</td>
      <td>5.5</td>
      <td>8.25</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Date</th>
      <td>12</td>
      <td>6</td>
      <td>2023-01-01</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Product</th>
      <td>12</td>
      <td>2</td>
      <td>A</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Revenue</th>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>138.833333</td>
      <td>45.095522</td>
      <td>75.0</td>
      <td>107.25</td>
      <td>131.0</td>
      <td>170.25</td>
      <td>221.0</td>
    </tr>
    <tr>
      <th>Quantity</th>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>7.25</td>
      <td>10.5</td>
      <td>12.75</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



### 1) Arrange DataFrame df, focusing on distinct product categories. Explore the data           to reveal:
#### a)Overall revenue insights
#### b)Average revenue patterns
#### c)Peak quantity sold observations
#### d)Minimum quantity sold occurrences


#### a)Overall revenue insights


```python
df.groupby(['Product'])['Revenue'].sum()
```




    Product
    A    774
    B    892
    Name: Revenue, dtype: int64



#### b)Average revenue patterns


```python
df.groupby(['Product'])['Revenue'].mean()
```




    Product
    A    129.000000
    B    148.666667
    Name: Revenue, dtype: float64



#### c)Peak quantity sold observations


```python
df.groupby(['Product'])['Quantity'].max()
```




    Product
    A    17
    B    19
    Name: Quantity, dtype: int64



#### d)Minimum quantity sold occurrences


```python
df.groupby(['Product'])['Quantity'].min()
```




    Product
    A    3
    B    5
    Name: Quantity, dtype: int64



### 2)Aggregate DataFrame df by extracting the month and year from the 'Date' column. Calculate the collective revenue for each unique combination of month and year.



```python
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year
xx = df[['Date','Month','Year']]
```


```python
xx
```




    Month     Year
    February  2023    431
    January   2023    644
    March     2023    591
    Name: Revenue, dtype: int64




```python
x = df.groupby(['Month','Year'])['Revenue'].sum()
```


```python
x
```




    Month     Year
    February  2023    431
    January   2023    644
    March     2023    591
    Name: Revenue, dtype: int64



### 4) Group the DataFrame df by the Date and find products that have a total quantity sold greater than 10 and less than 15 .


```python
a = df.groupby(['Date','Product'])['Quantity'].sum()
```


```python
b = x[(x>10) & (x<15)]
```


```python
b
```




    Series([], Name: Revenue, dtype: int64)



## 5)Group the DataFrame df by the 'Product' column and add a new column 'Revenue Percentage' that shows the percentage of each product's revenue relative to the total revenue for that product.



```python
x = df.groupby('Product')['Revenue'].sum()
```


```python
percentage = x / x.sum() * 100
```


```python
df['Revenue Percentage'] = df['Product'].map(percentage)
```


```python
df
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
      <th>Unnamed: 0</th>
      <th>Date</th>
      <th>Product</th>
      <th>Revenue</th>
      <th>Quantity</th>
      <th>Month</th>
      <th>Year</th>
      <th>Revenue Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2023-01-01</td>
      <td>A</td>
      <td>221</td>
      <td>3</td>
      <td>January</td>
      <td>2023</td>
      <td>46.458583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2023-01-01</td>
      <td>B</td>
      <td>111</td>
      <td>8</td>
      <td>January</td>
      <td>2023</td>
      <td>53.541417</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2023-01-02</td>
      <td>B</td>
      <td>171</td>
      <td>11</td>
      <td>January</td>
      <td>2023</td>
      <td>53.541417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2023-01-02</td>
      <td>A</td>
      <td>141</td>
      <td>15</td>
      <td>January</td>
      <td>2023</td>
      <td>46.458583</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2023-02-01</td>
      <td>A</td>
      <td>75</td>
      <td>4</td>
      <td>February</td>
      <td>2023</td>
      <td>46.458583</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2023-02-01</td>
      <td>B</td>
      <td>90</td>
      <td>12</td>
      <td>February</td>
      <td>2023</td>
      <td>53.541417</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>2023-02-02</td>
      <td>A</td>
      <td>96</td>
      <td>17</td>
      <td>February</td>
      <td>2023</td>
      <td>46.458583</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2023-02-02</td>
      <td>B</td>
      <td>170</td>
      <td>10</td>
      <td>February</td>
      <td>2023</td>
      <td>53.541417</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>2023-03-01</td>
      <td>B</td>
      <td>200</td>
      <td>19</td>
      <td>March</td>
      <td>2023</td>
      <td>53.541417</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>2023-03-01</td>
      <td>A</td>
      <td>120</td>
      <td>10</td>
      <td>March</td>
      <td>2023</td>
      <td>46.458583</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>2023-03-02</td>
      <td>A</td>
      <td>121</td>
      <td>12</td>
      <td>March</td>
      <td>2023</td>
      <td>46.458583</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>2023-03-02</td>
      <td>B</td>
      <td>150</td>
      <td>5</td>
      <td>March</td>
      <td>2023</td>
      <td>53.541417</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
