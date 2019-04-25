import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def lib(data):
	centroid=np.array([[0.10,0.60],[0.3,0.2]])
	kmeans=KMeans(n_clusters=2,init=centroid).fit(data)
	return kmeans.labels_,kmeans.cluster_centers_

def caldistance(data,centroid):
	dist1=[]
	dist2=[]
	
	for i in range(len(data)):
		dist1.append(np.sqrt((data.iloc[i][0]-centroid.iloc[0][0])**2+(data.iloc[i][1]-centroid.iloc[0][1])**2))
	for i in range(len(data)):
		dist2.append(np.sqrt((data.iloc[i][0]-centroid.iloc[1][0])**2+(data.iloc[i][1]-centroid.iloc[1][1])**2))
	return dist1,dist2

def formclusters(dist1,dist2,data):

	d1={'x':[],'y':[]}
	d2={'x':[],'y':[]}
	#print(dist2)
	for i in range(len(dist1)):
		if(dist1[i]<dist2[i]):
			d1['x'].append(data.iloc[i][0])
			d1['y'].append(data.iloc[i][1])
		else:
			d2['x'].append(data.iloc[i][0])
			d2['y'].append(data.iloc[i][1])


	cluster1=pd.DataFrame(data=d1,columns=['x','y'])
	cluster2=pd.DataFrame(data=d2,columns=['x','y'])

	return cluster1,cluster2

def calcentroid(cluster1,cluster2):
	d={'x':[],'y':[]}

	d['x'].append(np.sum(cluster1['x'])/len(cluster1))
	d['y'].append(np.sum(cluster1['y'])/len(cluster1))
	d['x'].append(np.sum(cluster2['x'])/len(cluster2))
	d['y'].append(np.sum(cluster2['y'])/len(cluster2))

	return pd.DataFrame(data=d,columns=['x','y'])


def kmeans(data):

	centroid=[[0.10,0.60],[0.3,0.2]]
	centroid=pd.DataFrame(centroid,columns=['x','y'])

	while(True):
		dist1,dist2=caldistance(data,centroid)
		cluster1,cluster2=formclusters(dist1,dist2,data)
		print("Current centroid:",centroid)
		newcentroid=calcentroid(cluster1,cluster2)
		if(newcentroid.equals(centroid)):
			break
		centroid=newcentroid
	print("cluster1:",cluster1)
	print("cluster2:",cluster2)
	print("centroid:",centroid)



def main():

	data=pd.read_csv("as4.csv")
	print(data)
	labels,centers=lib(data)
	print("labels:",labels)
	print("Cluster centers:",centers)

	print("using hard coded function:")
	kmeans(data)




if __name__ == '__main__':
	main()