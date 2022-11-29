# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Read the data set and find the number of null data.

3.Import KMeans from sklearn.clusters library package.

4.Find the y_pred .

5.Plot the clusters in graph.

## Program:
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Gayathri A
RegisterNumber:  212221230028
*/
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## Output:

![81](https://user-images.githubusercontent.com/94154854/204563278-d80e0e18-0dd6-4034-9484-31307cb27260.png)

![82](https://user-images.githubusercontent.com/94154854/204563306-68f4de0b-fdf1-4ccc-8fcf-d7046e56bd44.png)

![83](https://user-images.githubusercontent.com/94154854/204563777-345e424b-6cd8-42cf-8f9c-706cd0c559a5.png)

![84](https://user-images.githubusercontent.com/94154854/204563375-4baf336c-f82c-409e-840a-ad7ebb6888a9.png)

![85](https://user-images.githubusercontent.com/94154854/204563413-2d769ecb-3cfa-4a85-b55f-9c32247c1c40.png)

![86](https://user-images.githubusercontent.com/94154854/204563469-b43e5def-ed9c-4a3c-999c-c9bf46524ded.png)

![87](https://user-images.githubusercontent.com/94154854/204563938-a59d21cf-e2ae-4e5a-b277-255c446b7403.png)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
