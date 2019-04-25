import pandas as pd
import numpy as np
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def lib(X_train,X_test,y_train,y_test):
	classifier=KNeighborsClassifier(n_neighbors=5)
	classifier.fit(X_train,y_train.values.ravel())
	y_pred=classifier.predict(X_test)
	return y_pred

def score(y_test,y_pred):
	print("confusion_matrix:",confusion_matrix(y_test,y_pred))
	print("classification_report:",classification_report(y_test,y_pred))

def knn(X_train,X_test,y_train,y_test,k):

	y_pred=[]

	for i in range(len(X_test)):
		distance={}
		votes={}
		testx=X_test.iloc[i].values
		testy=y_test.iloc[i].values
		for j in range(len(X_train)): 
			distance[j]=euclidean(X_train.iloc[j].values,testx)
		sorted_d=sorted(distance.items(),key=operator.itemgetter(1))
		
		neighbors=[]

		for i in range(k):
			neighbors.append(sorted_d[i][0])

		for i in range(len(neighbors)):

			response=int(y_train.iloc[neighbors[i]].values)
			if response in votes:
				votes[response]+=1
			else:
				votes[response]=1

		sorted_votes=sorted(votes.items(),key=operator.itemgetter(1),reverse=True)

		y_pred.append(sorted_votes[0][0])

	return y_pred


def euclidean(a,b):
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


def main():

	data=pd.read_csv("as3.csv")
	# x=[[163,61]]
	# x=pd.DataFrame(x,columns=['Height','Weight'])
	# y=[[0]]
	# y=pd.DataFrame(y,columns=['Size'])
	# print(x)
	# print(y)

	lencoder=preprocessing.LabelEncoder()
	for i in data.columns.values:
		if data[i].dtype=="object":
			lencoder.fit(data[i])
			data[i]=lencoder.transform(data[i])

	X_train,X_test,y_train,y_test=train_test_split(data[['Height','Weight']],data[['Size']])
	print("Using library function")
	y_pred=lib(X_train,X_test,y_train,y_test)
	score(y_test,y_pred)
	print("y_pred:",y_pred)

	print("hardcoded function:")
	y_pred=knn(X_train,X_test,y_train,y_test,5)
	score(y_test,y_pred)
	print("y_pred:",np.array(y_pred))
if __name__ == '__main__':
	main()