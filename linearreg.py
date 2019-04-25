import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def cal_slope_intercept(X,y):
	sumx=np.sum(X)
	sumy=np.sum(y)
	sumxy=np.sum(X*y)
	sumx2=np.sum(X**2)
	n=len(X)
	slope=((n*sumxy)-(sumx*sumy))/((n*sumx2)-(sumx**2))
	intercept=(sumy-(slope*sumx))/n
	return slope,intercept

def score(y,y_pred):
	meany=np.mean(y)
	sstot=np.sum((y-meany)**2)
	ssres=np.sum((y-y_pred)**2)
	rsquare=1-(ssres/sstot)
	return rsquare

def lib(X,y):
	model=LinearRegression().fit(X,y)
	print("score:",model.score(X,y))
	y_pred=model.predict(X)
	return model.coef_,model.intercept_

def plot_regression_line(X,y,slope,intercept):
	y_pred=(slope*X)+intercept
	plt.scatter(X,y,color="r",marker="o")
	plt.plot(X,y_pred,color="b")
	plt.xlabel("year")
	plt.ylabel("sales")
	plt.show()
	return y_pred

def main():


	X=np.array([[2005],[2006],[2007],[2008],[2009]])
	y=np.array([[12],[19],[29],[37],[45]])

	# scaler=StandardScaler()
	# scaler.fit(X)
	# X=scaler.transform(X)
	#print(X)


	print("USING LIBRARY FUNCTION")
	slope,intercept=lib(X,y)
	print("slope:{}".format(slope))
	print("intercept:{}".format(intercept))
	y_pred=plot_regression_line(X,y,slope,intercept)
	print("predicted values:{}".format(y_pred))

	print("USING HARD CODED FUNCTION")
	slope1,intercept1=cal_slope_intercept(X,y)
	print("slope:{}".format(slope1))
	print("intercept:{}".format(intercept1))
	y_pred=plot_regression_line(X,y,slope1,intercept1)
	print("predicted values:{}".format(y_pred))
	print("Coefficient of determination(r squared):",score(y,y_pred))

	print("Sales in the year 2012:",slope*2012+intercept)
	print("Sales in the year 2012:",slope1*2012+intercept1)



if __name__ == '__main__':
	main()
