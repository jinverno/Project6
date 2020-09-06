import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import random

import warnings
import time

import datetime

df1=pd.read_csv("df_Ericeira_2008-11-22-2012-12-31.csv")
df2=pd.read_csv("df_Ericeira_2013-01-01-2016-12-31.csv")
df3=pd.read_csv("df_Ericeira_2017-01-01-2020-02-29.csv")

df12=pd.concat([df1,df2], axis=1,ignore_index=True)
df123=pd.concat([df12,df3], axis=1,ignore_index=True)

df123.drop(columns=[1503,1504,2966,2967],inplace=True)

df123.reset_index(inplace=True,drop=True)

df123.columns = df123.iloc[0]

df123.drop(index=[0],inplace=True)
new_names = df123.columns
new_names=list(new_names)

new_names[0] = 'metric name'
new_names[1] = 'time'

df123.columns = new_names

df123[df123["metric name"]=="Rain (mm/1h)"] =df123[df123["metric name"]=="Rain (mm/1h)"].fillna(value=0)

df123melt=pd.melt(df123, id_vars = ['metric name','time'])


df123melt["date_time"]=pd.to_datetime(df123melt["variable"]+" "+df123melt["time"], dayfirst=True)
df123melt.drop(columns=["time"],inplace=True) #["time","variable"]

df123melt=df123melt[['metric name', "variable", 'date_time','value']]

df_wave=df123melt[df123melt['metric name']=="Wave (m)"]
df_wave.reset_index(inplace=True,drop=True)
df_wave["value"]=df_wave["value"].replace("-",np.NaN).astype(float)
df_wave["value"]=df_wave["value"].interpolate(method ='linear', limit_direction ='forward')

#Plot waves H
plt.figure(figsize=(17,10))
plt.plot(df_wave["date_time"],df_wave["value"],linewidth=0.25,color='k')
plt.show()

#Series of 2 cycles time series with sm.tsa.seasonal_decompose
s_from="2016-01-01 00:00:00"
s_to="2017-12-31 00:00:00"
series_wave=pd.Series(df_wave[(df_wave["date_time"]>= s_from) & (df_wave["date_time"]<=s_to)]["value"])
series_wave.index=df_wave[(df_wave["date_time"]>=s_from) & (df_wave["date_time"]<=s_to)]["date_time"]

#Decomposition_waves 
rcParams['figure.figsize'] = 38,10
decomposition = sm.tsa.seasonal_decompose(series_wave, model='additive') ## model becomes multiplicative isntead of aditive 
fig = decomposition.plot()
plt.show()

#Group by Wave per day
df_wave_day=df_wave.groupby(by="variable", sort=True).mean().reset_index()
df_wave_day["variable"]=pd.to_datetime(df_wave_day["variable"],dayfirst=True)
df_wave_day.sort_values(by="variable", ascending=True,inplace=True)
df_wave_day.reset_index(inplace=True,drop=True)
df_wave_day.columns=["date_time","value"]

#Decomposition_waves BY DAY
"""plt.figure(figsize=(12,6))
plt.plot(df_wave_day["date_time"],df_wave_day["value"],linewidth=0.25,color='k')
plt.show()"""

s_from="2016-01-01 00:00:00"
s_to="2017-12-31 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]

rcParams['figure.figsize'] = [35, 15]
rcParams['figure.dpi'] = 80
rcParams['savefig.dpi'] = 1200

rcParams['font.size'] = 15
rcParams['legend.fontsize'] = 'large'
rcParams['figure.titlesize'] = 'medium'
decomposition = sm.tsa.seasonal_decompose(series_wave_day, model='additive') ## model becomes multiplicative isntead of aditive 
fig = decomposition.plot()
plt.show()


"""decomp = seasonal_decompose(series_wave_day)
decomp.plot()
plt.show()"""

#ARIMA

#The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters to use.

#p (seasonality): The number of lag observations included in the model, also called the lag order.

#d (trend): The number of times that the raw observations are differenced, also called the degree of differencing.

#q (noise): The size of the moving average window, also called the order of moving average.

t0 = time.time()


#WAVE TRAIN MODEL
#Train the model to get the optimum p d q
s_from="2016-01-01 00:00:00"
s_to="2017-12-31 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    
    
# Evaluate parameters
"""p_values = [1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series_wave_day.values, p_values, d_values, q_values)""" 

t1 = time.time()
total = t1-t0
print("Total time: "+str(total/60)+"min")


#Case 1: 25 April 2018 - WAVE HEIGHT
s_from="2016-01-01 00:00:00"
s_to="2017-12-31 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]

days = 3  # days to forecast
forecast_true=series_wave_day[-days:]
series_wave_day_train=series_wave_day[:-days]

arima = ARIMA(series_wave_day_train.values, order = (4, 0, 1))
arima = arima.fit()
#arima.summary()

N=len(series_wave_day_train)

plt.plot(series_wave_day.values, label="Archive")
plt.plot(arima.predict(1, N + days), color="red", label="ARIMA(4,0,1)")
plt.legend(loc="upper left")
plt.ylabel('H (m)')
plt.show()

#Plot for presentation
plt.plot(series_wave_day[:482],linewidth=2.0,label="dataset - Train")
plt.plot(series_wave_day[482:],  linewidth=2.0, color="green", label="dataset - Test")
plt.legend(loc="upper left")
plt.ylabel('H (m)')
plt.show()


print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)

#PLOTTAR
s_from="2018-04-13 00:00:00"
s_to="2018-04-25 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]
#Survey
#list=[str(series_wave_day.index[-days-1])[0:10], series_wave_day[-days-1],"2018-04-25",1.316666667]xxxx
#pd_list=pd.Series(list) xxxx
fig=plt.figure(figsize = (25,4))
#plt.plot(series_wave_day[:-days].index,series_wave_day[:-days],linewidth=2.0, label='Archive')
plt.plot(series_wave_day.index,series_wave_day,linewidth=2.0, label='Archive all')

plt.scatter(str(series_wave_day.index[-days-1])[0:10], series_wave_day[-days-1], c="m",s=100)
plt.scatter("2018-04-25",1.316666667, c="m",s=175, label='Survey')
#x1,x2,y1,y2 = plt.axis()
#plt.axis((y1,y2,-150,1850))
#plt.plot(merged["Date"],merged["difs_x"],'b', linewidth=2.0, label='ZOOM (USA)') #ZM
#plt.plot(merged["Date"],merged["difs_y"],'r', linewidth=2.0, label='ZOOM (IN)')
plt.ylabel('H (m)')

plt.plot(series_wave_day.index,arima.predict(1, N + days)[-len(series_wave_day.index):], color="red")
plt.scatter(series_wave_day.index[-days:],arima.predict(1, N + days)[-len(series_wave_day.index):][-days:], color="red",s=150, label="ARIMA")
#plt.legend(loc="upper left")

plt.legend(loc="upper right")
plt.show()


#Case 2: 28 August 2018 - WAVE HEIGHT
s_from="2016-01-01 00:00:00"
s_to="2018-08-28 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]

days = 3  # days to forecast
forecast_true=series_wave_day[-days:]
series_wave_day_train=series_wave_day[:-days]

arima = ARIMA(series_wave_day_train.values, order = (4, 0, 1))
arima = arima.fit()
#arima.summary()

N=len(series_wave_day_train)

plt.plot(series_wave_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)


#PLOTTAR
s_from="2018-08-17 00:00:00"
s_to="2018-08-28 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]
#Survey
fig=plt.figure(figsize = (25,4))
plt.plot(series_wave_day.index,series_wave_day,linewidth=2.0, label='Archive all')
#plt.plot(series_wave_day[:-days].index,series_wave_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_wave_day.index[-days-1])[0:10], series_wave_day[-days-1], c="m",s=100)
plt.scatter("2018-08-28",2.183333333, c="m",s=175, label='Survey')
plt.ylabel('H (m)')


plt.plot(series_wave_day.index,arima.predict(1, N + days)[-len(series_wave_day.index):], color="red")
plt.scatter(series_wave_day.index[-days:],arima.predict(1, N + days)[-len(series_wave_day.index):][-days:], color="red",s=150, label="ARIMA")
plt.legend(loc="upper left")
plt.show()



#Case 3: 26 December 2020 - WAVE HEIGHT
s_from="2016-01-01 00:00:00"
s_to="2018-12-26 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]

days = 3  # days to forecast
forecast_true=series_wave_day[-days:]
series_wave_day_train=series_wave_day[:-days]

arima = ARIMA(series_wave_day_train.values, order = (4, 0, 1))
arima = arima.fit()
#arima.summary()

N=len(series_wave_day_train)

plt.plot(series_wave_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)

s_from="2018-12-14 00:00:00"
s_to="2018-12-26 00:00:00"
series_wave_day=pd.Series(df_wave_day[(df_wave_day["date_time"]>= s_from) & (df_wave_day["date_time"]<=s_to)]["value"])
series_wave_day.index=df_wave_day[(df_wave_day["date_time"]>=s_from) & (df_wave_day["date_time"]<=s_to)]["date_time"]
#Survey
fig=plt.figure(figsize = (25,4))
plt.plot(series_wave_day.index,series_wave_day,linewidth=2.0, label='Archive all')
#plt.plot(series_wave_day[:-days].index,series_wave_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_wave_day.index[-days-1])[0:10], series_wave_day[-days-1], c="m",s=100)
plt.scatter("2018-12-26",2.433333333, c="m",s=175, label='Survey')
plt.ylabel('H (m)')

plt.plot(series_wave_day.index,arima.predict(1, N + days)[-len(series_wave_day.index):], color="red")
plt.scatter(series_wave_day.index[-days:],arima.predict(1, N + days)[-len(series_wave_day.index):][-days:], color="red",s=150, label="ARIMA")
#plt.legend(loc="upper left")
plt.legend(loc="upper right")
plt.show()




#- WAVE PERIOD

df_period=df123melt[df123melt['metric name']=="Wave period (s)"]
df_period.reset_index(inplace=True,drop=True)
df_period["value"]=df_period["value"].replace("-",np.NaN).astype(float)
df_period["value"]=df_period["value"].interpolate(method ='linear', limit_direction ='forward')

#Wave dir clean group day
df_period_day=df_period.groupby(by="variable", sort=True).mean().reset_index()
df_period_day["variable"]=pd.to_datetime(df_period_day["variable"],dayfirst=True)
df_period_day.sort_values(by="variable", ascending=True,inplace=True)
df_period_day.reset_index(inplace=True,drop=True)
df_period_day.columns=["date_time","value"]

df_period_day["value"]=df_period_day["value"].apply(round)

#Time period
s_from="2016-01-01 00:00:00"
s_to="2017-12-31 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]

"""plt.figure(figsize=(12,6))
plt.plot(series_period_day.index,series_period_day,linewidth=0.25,color='k')
plt.show()"""

decomp = seasonal_decompose(series_period_day)
decomp.plot()
plt.show()

# evaluate parameters
"""p_values = [1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series_period_day.values, p_values, d_values, q_values)""" 



#Case 1: 25 April 2018 
s_from="2016-01-01 00:00:00"
s_to="2018-04-25 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]

days = 3  # days to forecast
forecast_true=series_period_day[-days:]
series_period_day_train=series_period_day[:-days]

arima = ARIMA(series_period_day_train.values, order = (2,0,2))
arima = arima.fit()
#arima.summary()

N=len(series_period_day_train)

plt.plot(series_period_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)


#PLOTTAR
s_from="2018-04-13 00:00:00"
s_to="2018-04-25 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]
#Survey

fig=plt.figure(figsize = (25,4))
plt.plot(series_period_day.index,series_period_day,linewidth=2.0, label='Archive all')
#plt.plot(series_period_day[:-days].index,series_period_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_period_day.index[-days-1])[0:10], series_period_day[-days-1], c="m",s=100)
plt.scatter("2018-04-25",10.16666667, c="m",s=175, label='Survey')

plt.ylabel('T (s)')

plt.plot(series_period_day.index,arima.predict(1, N + days)[-len(series_period_day.index):], color="red")
plt.scatter(series_period_day.index[-days:],arima.predict(1, N + days)[-len(series_period_day.index):][-days:], color="red",s=150, label="ARIMA")
plt.legend(loc="upper left")
#plt.legend(loc="upper right")
plt.show()




#Case 2:28 August 2018  - WAVE PERIOD
s_from="2016-01-01 00:00:00"
s_to="2018-08-28 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]

"""plt.figure(figsize=(12,6))
plt.plot(series_period_day.index,series_period_day,linewidth=0.25,color='k')
plt.show()"""

"""decomp = seasonal_decompose(series_period_day)
decomp.plot()
plt.show()"""

# evaluate parameters
"""p_values = [1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series_wave_day.values, p_values, d_values, q_values)""" 

days = 3  # days to forecast
forecast_true=series_period_day[-days:]
series_period_day_train=series_period_day[:-days]

arima = ARIMA(series_period_day_train.values, order = (2,0,2))
arima = arima.fit()
#arima.summary()

N=len(series_period_day_train)

plt.plot(series_period_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)

#PlOTTAR
s_from="2018-08-17 00:00:00"
s_to="2018-08-28 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]
#Survey

fig=plt.figure(figsize = (25,4))
plt.plot(series_period_day.index,series_period_day,linewidth=2.0, label='Archive all')
#plt.plot(series_period_day[:-days].index,series_period_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_period_day.index[-days-1])[0:10], series_period_day[-days-1], c="m",s=100)
plt.scatter("2018-08-28",10.16666667, c="m",s=175, label='Survey')

plt.ylabel('T (s)')

plt.plot(series_period_day.index,arima.predict(1, N + days)[-len(series_period_day.index):], color="red")
plt.scatter(series_period_day.index[-days:],arima.predict(1, N + days)[-len(series_period_day.index):][-days:], color="red",s=150, label="ARIMA")
plt.legend(loc="upper left")
#plt.legend(loc="upper right")
plt.show()



#Case 3: 26 December 2020   - WAVE PERIOD
s_from="2016-01-01 00:00:00"
s_to="2018-12-26 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]

"""plt.figure(figsize=(12,6))
plt.plot(series_period_day.index,series_period_day,linewidth=0.25,color='k')
plt.show()"""

decomp = seasonal_decompose(series_period_day)
decomp.plot()
plt.show()

# evaluate parameters
"""p_values = [1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series_wave_day.values, p_values, d_values, q_values)""" 

days = 3  # days to forecast
forecast_true=series_period_day[-days:]
series_period_day_train=series_period_day[:-days]

arima = ARIMA(series_period_day_train.values, order = (2,0,2))
arima = arima.fit()
#arima.summary()

N=len(series_period_day_train)

plt.plot(series_period_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)

#PlOTTAR
s_from="2018-12-14 00:00:00"
s_to="2018-12-26 00:00:00"
series_period_day=pd.Series(df_period_day[(df_period_day["date_time"]>= s_from) & (df_period_day["date_time"]<=s_to)]["value"])
series_period_day.index=df_period_day[(df_period_day["date_time"]>=s_from) & (df_period_day["date_time"]<=s_to)]["date_time"]
#Survey

fig=plt.figure(figsize = (25,4))
plt.plot(series_period_day.index,series_period_day,linewidth=2.0, label='Archive all')
#plt.plot(series_period_day[:-days].index,series_period_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_period_day.index[-days-1])[0:10], series_period_day[-days-1], c="m",s=100)
plt.scatter("2018-12-26",13.16666667, c="m",s=175, label='Survey')

plt.ylabel('T (s)')

plt.plot(series_period_day.index,arima.predict(1, N + days)[-len(series_period_day.index):], color="red")
plt.scatter(series_period_day.index[-days:],arima.predict(1, N + days)[-len(series_period_day.index):][-days:], color="red",s=150, label="ARIMA")
plt.legend(loc="upper left")
plt.legend(loc="upper right")
plt.show()



##Wave dir clean
df_wave_dir=df123melt[df123melt['metric name']=="Wave direction"]
df_wave_dir.reset_index(inplace=True,drop=True)
df_wave_dir["value"]=df_wave_dir["value"].replace(["nothing","dir"],np.NaN).astype(float)
df_wave_dir["value"]=df_wave_dir["value"].interpolate(method ='linear', limit_direction ='forward')

df_wave_dir["value"]=(df_wave_dir["value"]-360+180)

#Wave dir clean group day
df_wave_dir_day=df_wave_dir.groupby(by="variable", sort=True).mean().reset_index()
df_wave_dir_day["variable"]=pd.to_datetime(df_wave_dir_day["variable"],dayfirst=True)
df_wave_dir_day.sort_values(by="variable", ascending=True,inplace=True)
df_wave_dir_day.reset_index(inplace=True,drop=True)
df_wave_dir_day.columns=["date_time","value"]

df_wave_dir_day["value"]=df_wave_dir_day["value"].apply(round)

s_from="2016-01-01 00:00:00"
s_to="2017-12-31 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

decomp = seasonal_decompose(series_wave_dir_day)
decomp.plot()
plt.show()

# evaluate parameters
"""p_values = [1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series_wave_dir_day.values, p_values, d_values, q_values)""" 


#Case 1: 25 April 2018 
s_from="2016-01-01 00:00:00"
s_to="2018-04-25 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

plt.figure(figsize=(12,6))
plt.plot(series_wave_dir_day.index,series_wave_dir_day,linewidth=0.25,color='k')
plt.show()

days = 3  # days to forecast
forecast_true=series_wave_dir_day[-days:]
series_period_day_train=series_wave_dir_day[:-days]

arima = ARIMA(series_wave_dir_day.values, order = (4, 0, 2))
arima = arima.fit()
#arima.summary()

N=len(series_wave_dir_day)

plt.plot(series_wave_dir_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)


#PLOTTAR
s_from="2018-04-13 00:00:00"
s_to="2018-04-25 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

#Survey

fig=plt.figure(figsize = (25,4))
plt.plot(series_wave_dir_day.index,series_wave_dir_day,linewidth=2.0, label='Archive all')
#plt.plot(series_wave_dir_day[:-days].index,series_wave_dir_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_wave_dir_day.index[-days-1])[0:10], series_wave_dir_day[-days-1], c="m",s=100)
plt.scatter("2018-04-25",300.8333333, c="m",s=175, label='Survey')

plt.ylabel('D (°)')

plt.plot(series_wave_dir_day.index,arima.predict(1, N + days)[-len(series_wave_dir_day.index):], color="red")
plt.scatter(series_wave_dir_day.index[-days:],arima.predict(1, N + days)[-len(series_wave_dir_day.index):][-days:], color="red",s=150, label="ARIMA")
plt.legend(loc="upper left")
#plt.legend(loc="upper right")
plt.show()



#Case 2:28 August 2018  - WAVE Dir
s_from="2016-01-01 00:00:00"
s_to="2018-08-28 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

plt.figure(figsize=(12,6))
plt.plot(series_wave_dir_day.index,series_wave_dir_day,linewidth=0.25,color='k')
plt.show()

days = 3  # days to forecast
forecast_true=series_wave_dir_day[-days:]
series_period_day_train=series_wave_dir_day[:-days]

arima = ARIMA(series_wave_dir_day.values, order = (4, 0, 2))
arima = arima.fit()
#arima.summary()

N=len(series_wave_dir_day)

plt.plot(series_wave_dir_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)



s_from="2018-08-17 00:00:00"
s_to="2018-08-28 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

#Survey

fig=plt.figure(figsize = (25,4))
plt.plot(series_wave_dir_day.index,series_wave_dir_day,linewidth=2.0, label='Archive all')
#plt.plot(series_wave_dir_day[:-days].index,series_wave_dir_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_wave_dir_day.index[-days-1])[0:10], series_wave_dir_day[-days-1], c="m",s=100)
plt.scatter("2018-08-28",325, c="m",s=175, label='Survey')

plt.ylabel('D (°)')

plt.plot(series_wave_dir_day.index,arima.predict(1, N + days)[-len(series_wave_dir_day.index):], color="red")
plt.scatter(series_wave_dir_day.index[-days:],arima.predict(1, N + days)[-len(series_wave_dir_day.index):][-days:], color="red",s=150, label="ARIMA")
#plt.legend(loc="upper left")
plt.legend(loc="upper right")
plt.show()


#Case 3: 26 December 2020   - WAVE PERIOD
s_from="2016-01-01 00:00:00"
s_to="2018-12-26 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

plt.figure(figsize=(12,6))
plt.plot(series_wave_dir_day.index,series_wave_dir_day,linewidth=0.25,color='k')
plt.show()

days = 3  # days to forecast
forecast_true=series_wave_dir_day[-days:]
series_period_day_train=series_wave_dir_day[:-days]

arima = ARIMA(series_wave_dir_day.values, order = (4, 0, 2))
arima = arima.fit()
#arima.summary()

N=len(series_wave_dir_day)

plt.plot(series_wave_dir_day.values)
plt.plot(arima.predict(1, N + days), color="red")
plt.show()

print("N", N)
print("forecast", arima.predict(1, N + days)[-days:])
print("true", forecast_true.values)
MSE=mean_squared_error(forecast_true.values,arima.predict(1, N + days)[-days:])
print("MSE",MSE)


#PLOTTAR
s_from="2018-12-14 00:00:00"
s_to="2018-12-26 00:00:00"
series_wave_dir_day=pd.Series(df_wave_dir_day[(df_wave_dir_day["date_time"]>= s_from) & (df_wave_dir_day["date_time"]<=s_to)]["value"])
series_wave_dir_day.index=df_wave_dir_day[(df_wave_dir_day["date_time"]>=s_from) & (df_wave_dir_day["date_time"]<=s_to)]["date_time"]

#Survey

fig=plt.figure(figsize = (25,4))
plt.plot(series_wave_dir_day.index,series_wave_dir_day,linewidth=2.0, label='Archive all')
#plt.plot(series_wave_dir_day[:-days].index,series_wave_dir_day[:-days],linewidth=2.0, label='Archive')
plt.scatter(str(series_wave_dir_day.index[-days-1])[0:10], series_wave_dir_day[-days-1], c="m",s=100)
plt.scatter("2018-12-26",293.8333333, c="m",s=175, label='Survey')

plt.ylabel('D (°)')

plt.plot(series_wave_dir_day.index,arima.predict(1, N + days)[-len(series_wave_dir_day.index):], color="red")
plt.scatter(series_wave_dir_day.index[-days:],arima.predict(1, N + days)[-len(series_wave_dir_day.index):][-days:], color="red",s=150, label="ARIMA")
plt.legend(loc="upper left")
#plt.legend(loc="upper right")
plt.show()


"""fig=plt.figure(figsize = (20,15))
x1,x2,y1,y2 = plt.axis()
plt.axis((y1,y2,-150,1850))
plt.plot(merged["Date"],merged["difs_x"],'b', linewidth=2.0, label='ZOOM (USA)') #ZM
plt.plot(merged["Date"],merged["difs_y"],'r', linewidth=2.0, label='ZOOM (IN)')
plt.ylabel('%')

plt.legend(loc="upper left")"""




#Other parameters not used

df_wind=df123melt[df123melt['metric name']=="Wind speed (knots)"]
df_wind.reset_index(inplace=True,drop=True)

df_temperature=df123melt[df123melt['metric name']=="Temperature (°C)"]
df_temperature.reset_index(inplace=True,drop=True)

df_rain=df123melt[df123melt['metric name']=="Rain (mm/1h)"]
df_rain.reset_index(inplace=True,drop=True)



