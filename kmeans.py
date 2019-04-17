import pandas as pd
import numpy as np

## READ CSV FILE

data=pd.read_csv("as4.csv")

## INITIALIZE CENTROID AS DATAFRAME

## cluster1 has centroid(0.10,0.60) and cluster2 has centroid(0.30,0.20)
centroid=[[0.10,0.60],[0.30,0.20]]
centroid=pd.DataFrame(centroid,columns=['x','y'])

##FUNCTION TO FIND DISTANCE BETWEEN CENTROID AND DATA

def distance(data,centroid):
	'''
	INPUT:DATA(DATAFRAME) and CENTROID(DATAFRAME)
	RETURNS: TWO LISTS dist1(distance of data from first cluster) and dist2(distance of centroid from second cluster)
	'''
	dist1=[]
	dist2=[]
	for i in range(len(data)):
		dist1.append(np.sqrt(((data.iloc[i][0]-centroid.iloc[0][0])**2)+(data.iloc[i][1]-centroid.iloc[0][1])**2))
	for i in range(len(data)):
		dist2.append(np.sqrt(((data.iloc[i][0]-centroid.iloc[1][0])**2)+(data.iloc[i][1]-centroid.iloc[1][1])**2))
	return dist1,dist2

## FUNCTION TO FIND CLUSTERS

def formclusters(dist1,dist2,data):
    d1={'x':[],'y':[]}
    d2={'x':[],'y':[]}
    for i in range(len(dist1)):
        if(dist1[i]<=dist2[i]):
            
            d1['x'].append(data.iloc[i][0])
            d1['y'].append(data.iloc[i][1])
  
        else:

          
            d2['x'].append(data.iloc[i][0])
            d2['y'].append(data.iloc[i][1])
    cluster1=pd.DataFrame(data=d1,columns=['x','y'])
    cluster2=pd.DataFrame(data=d2,columns=['x','y'])
    return cluster1,cluster2

## FUNCTION TO FIND CENTROID OF THE NEWLY FORMED CLUSTERS

def calcentroid(cluster1,cluster2):

	'''
	INPUT:cluster1 and cluster2
	RETURNS: the centroids of the clusters
	'''
	a=0
	b=0
	d={'x':[],'y':[]}
	d['x'].append(cluster1['x'].sum()/len(cluster1))
	d['x'].append(cluster2['x'].sum()/len(cluster2))
	d['y'].append(cluster1['y'].sum()/len(cluster1))
	d['y'].append(cluster2['y'].sum()/len(cluster2))
	newcentroid=pd.DataFrame(data=d,columns=['x','y'])
	return newcentroid 

def main(data,centroid):
    dist1=[]
    dist2=[]
    cluster1=pd.DataFrame(columns=['x','y'])
    cluster2=pd.DataFrame(columns=['x','y'])
    while(True):
        dist1,dist2=distance(data,centroid)
        cluster1,cluster2=formclusters(dist1,dist2,data)
        newcentroid=calcentroid(cluster1,cluster2)
        if(centroid.equals(newcentroid)):
            break
        centroid=newcentroid
    return cluster1,cluster2,newcentroid 


cluster1,cluster2,newcentroid=main(data,centroid)
print("cluster1:{}".format(cluster1))
print("cluster2:{}".format(cluster2))
print("centroid:{}".format(newcentroid))
