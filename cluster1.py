import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = np.array([[25,50000],[30,60000],[35,70000],[40,80000],[50,90000],[55,100000],[60,110000]])
df = pd.DataFrame(data)

new_customer = np.array([24,49000])

model = KMeans(n_clusters=3,random_state=42)
model.fit(df)
cluster_center= model.cluster_centers_
label= model.labels_
print('\n',cluster_center,"********",label)
prediction = model.predict([new_customer])[0]
print(prediction)

for label, color in zip(np.unique(labels),['blue','orange']):
    plt.scatter(x[labels==label,0],x[labels==label,1],label=named_clusters[label],c=color)
    # plot the centroids
plt.scatter(centroids[:,0],centroids[:,1],s=200, c='red',marker=)