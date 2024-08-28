import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {"x": [1, 2], "y": [3, 4], "z": [5, 6]}
df = pd.DataFrame(data)

# print(df)

first_point = df.iloc[0]
second_point = df.iloc[1]

distance_calculation = np.sqrt(np.sum((second_point-first_point)**2))
# print(distance_calculation)
# ==================================

x = np.array([[1,2],[1,4],[1,0],[12,2],[10,6],[10,0]])
print("x:", x)
print("ndim:",x.ndim)
model = KMeans(n_clusters=2, random_state=42)
model.fit(x)
print("cluster Center\n",model.cluster_centers_)
labels= model.labels_
print("labels",model.labels_)

cluster_naming= {0:"CLU1", 1:"CLU2"}

named_cluster = [cluster_naming[x] for x in model.labels_]
print(named_cluster)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(x[:,0],x[:,1])
plt.show()