from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
import pandas
import numpy as np
import math

pothole = pandas.read_csv('/Users/snehamitta/Desktop/ML/MidTerm/ChicagoCompletedPotHole.csv', delimiter = ',')

data = pothole[["LATITUDE", "LONGITUDE"]]  
a = pothole["N_POTHOLES_FILLED_ON_BLOCK"].apply(lambda x: math.log(x))
b = pothole["N_DAYS_FOR_COMPLETION"].apply(lambda x: math.log(1 + x))
trainData = data.join([a,b])

#Q1.a) 

#To find elbow values from cluster 2 to cluster 15 

wk = []
K = range(2,16)
for i in range(2,16):
   kmeanModel = KMeans(n_clusters = i, random_state = 20181010).fit(trainData)
   wk.append((kmeanModel.inertia_)/trainData.shape[0])
   
print(wk)
plt.plot(K, wk, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow W')
plt.title('The elbow values for clusters 2 to 15')
plt.grid(True)
plt.show()

#To find Silhouette values from cluster 2 to cluster 15

s = []
K = range(2,16)
for i in K:
   kmeanModel = KMeans(n_clusters = i, random_state = 20181010).fit(trainData)
   labels = kmeanModel.fit_predict(trainData)
   s.append(metrics.silhouette_score(trainData, labels, metric = 'euclidean'))
   
print(s)
plt.plot(K, s, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhoutte Score')
plt.title('The silhouette values for clusters 2 to 15')
plt.grid(True)
plt.show()


#Q1.b) To create box-plot for asked variables, grouped by the cluster_id

kmeans = KMeans(n_clusters=4, random_state=20181010).fit(trainData)
print("Cluster Centroids = \n", kmeans.cluster_centers_)
ClusterResult = trainData
ClusterResult['ClusterLabel'] = kmeans.labels_

for i in range(4):
   print("Cluster Label = ", i)
   print(ClusterResult.loc[ClusterResult['ClusterLabel'] == i])
   
data1 = pothole.join(ClusterResult['ClusterLabel'])

data1.boxplot(column= 'N_POTHOLES_FILLED_ON_BLOCK', by='ClusterLabel', vert=False, figsize=(6,4))
plt.xlabel('N_POTHOLES_FILLED_ON_BLOCK')
plt.ylabel('ClusterLabel')
plt.grid(True)
plt.show()

data1.boxplot(column= 'N_DAYS_FOR_COMPLETION', by='ClusterLabel', vert=False, figsize=(6,4))
plt.title('BoxPlot Grouped by ClusterLabel')
plt.xlabel("N_DAYS_FOR_COMPLETION")
plt.ylabel("ClusterLabel")
plt.show()

data1.boxplot(column= 'LATITUDE', by='ClusterLabel', vert=False, figsize=(6,4))
plt.title('BoxPlot Grouped by ClusterLabel')
plt.xlabel("LATITUDE")
plt.ylabel("ClusterLabel")
plt.show()

data1.boxplot(column= 'LONGITUDE', by='ClusterLabel', vert=False, figsize=(6,4))
plt.title('BoxPlot Grouped by ClusterLabel')
plt.xlabel("LONGITUDE")
plt.ylabel("ClusterLabel")
plt.show()


#Q1.c) To create a Scatterplot for latitude agaisnt longitude

data = data.join(ClusterResult['ClusterLabel'])

fig = plt.figure()
ax = fig.add_subplot(111, aspect = 'equal')
plt.scatter(data['LONGITUDE'], data['LATITUDE'], s = 3, marker = '.', alpha = 0.4, c = data['ClusterLabel'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of latitude vs longitude')
plt.show()


#Q1.e) To build classification tree

data = data.join(pothole[['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION']])

input1 = data[['N_POTHOLES_FILLED_ON_BLOCK','N_DAYS_FOR_COMPLETION','LONGITUDE','LATITUDE']]
target = data['ClusterLabel']

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20181010)
potholeDT = classTree.fit(input1, target)
print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(input1, target)))
dot_data = tree.export_graphviz(potholeDT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['N_POTHOLES_FILLED_ON_BLOCK','N_DAYS_FOR_COMPLETION','LONGITUDE','LATITUDE'],
                                class_names = ['0', '1', '2','3'])

graph = graphviz.Source(dot_data)
print(graph)

graph.render('/Users/snehamitta/Desktop/pothole')


#Q1.f) To find the misclassification rate and RMS error

count = 0
pred = potholeDT.predict(input1)
pred_prob = potholeDT.predict_proba(input1)

for i in range(data.shape[0]):
    if pred[i] != data['ClusterLabel'].iloc[i]:
        count += 1

print('The misclassification rate of the classification tree is = ', count/data.shape[0])

r2 = []
t2 = []
cats = data['ClusterLabel'].unique()
for i in range (len(input1)):
    s2 = []
    for cat in cats:
        if(pred[i] == cat):
            s2.append((1-pred_prob[i][cat])**2)
        else:
            s2.append((0-pred_prob[i][cat])**2)
    r2.append(sum(s2))
t2 = sum(r2)
rase2 = np.sqrt(t2/(2*(len(input1))))
print('The RASE value for Classification Tree is', rase2)

