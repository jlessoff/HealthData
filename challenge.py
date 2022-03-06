from pytrends.request import TrendReq
import pandas as pd
import datetime
pd.set_option('display.max_columns', None)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std
from numpy import absolute

from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.dates as mdates

sns.set()
import numpy as np

import time
import random
from matplotlib import pyplot as plt
# for list of key words, interest over time in given period for regional level
# pytrends = TrendReq(hl='en-US', tz=360)
from datetime import date


# # GOOGLE DATA
# pytrends.build_payload(kw_list=['covid','coronavirus'], timeframe=f'2020-02-26 {date.today()}', geo='US')
# regiondf = pytrends.interest_by_region()
# df_ibr =pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
# related_queries = pytrends.related_queries()
# related_queries.values()
# df_rq = list(related_queries.values())[0]['rising']
# dfrising = pd.DataFrame(df_rq).head(20)
# keywords=dfrising['query'].tolist()
# kw_list=pd.read_csv('google.csv')
# kw_list=kw_list['0'].values.tolist()
# key_words=kw_list+keywords
# print(key_words)
# df_us_states=pd.read_csv('us_states.csv',header=None)
# df_us_states.rename(columns={0:'state'},inplace=True)
# us_states=df_us_states['state'].to_list()
# us_states=us_states
# print(us_states)
# tmp = pd.DataFrame()
# print(key_words)


# for j in us_states:
#     for i in key_words:
#         pytrends.build_payload(kw_list=[i],timeframe=f'2020-02-26 {date.today()}', geo=j)
#         pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
#         time_df = pytrends.interest_over_time()
#         time_df.rename(columns={i: 'number'}, inplace=True)
#         time_df['keyword']=i
#         time_df['state']=j
#         tmp = tmp.append(time_df)
#         time.sleep(20+ random.random())
# tmp.reset_index()
# tmp.to_csv('google_data.csv', index=True, header=True)
##COVID DATA USA CASES
# covid_daily = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
# dates = covid_daily.columns[11:]
# confirmed_df_long = covid_daily.melt(
#     id_vars=['Province_State', 'Country_Region','Admin2'],
#     value_vars=dates,
#     var_name='Date',
#     value_name='Confirmed'
# )
# us_covid_df=confirmed_df_long.groupby([ 'Province_State','Date']).sum().reset_index()
# us_covid_df['Date']=pd.to_datetime(us_covid_df['Date'])
# us_covid_df['week']=us_covid_df['Date'].dt.strftime('%Y-%U')
# us_covid_df=us_covid_df.groupby(by=['Province_State','week']).sum().reset_index()
# us_covid_df['Date_Time'] = pd.to_datetime(us_covid_df.week + '0', format='%Y-%W%w')
# us_covid_df.to_csv('covid_data.csv', index=True, header=True)


#covid data us deaths
# covid_daily = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
# dates = covid_daily.columns[13:]
# confirmed_df_long = covid_daily.melt(
#     id_vars=['Province_State', 'Country_Region','Admin2'],
#     value_vars=dates,
#     var_name='Date',
#     value_name='Death'
# )
# us_covid_df=confirmed_df_long.groupby([ 'Province_State','Date']).sum().reset_index()
# us_covid_df['Date']=pd.to_datetime(us_covid_df['Date'])
# us_covid_df['week']=us_covid_df['Date'].dt.strftime('%Y-%U')
# us_covid_df=us_covid_df.groupby(by=['Province_State','week']).sum().reset_index()
# us_covid_df['Date_Time'] = pd.to_datetime(us_covid_df.week + '0', format='%Y-%W%w')
# us_covid_df.to_csv('covid_data_death.csv', index=True, header=True)
#



google_data=pd.read_csv('google_data.csv')
covid_data=pd.read_csv('covid_data.csv')
covid_data_death=pd.read_csv('covid_data_death.csv')
df_us_states=pd.read_csv('us_states.csv')
mapped= google_data.merge(df_us_states, left_on='state', right_on='state_ab')
covid_data=covid_data.merge(covid_data_death, on=[ 'Date_Time','Province_State'])
df=mapped.merge(covid_data, left_on=['state_y','date'], right_on=['Province_State','Date_Time'])
print(df)
df = df[(df.isPartial == False)]
df=df[['date','number','keyword','Province_State','Confirmed','Death','week_x']]
df.to_csv('df.csv', index=True, header=True)
df=df[df.date!='2021-01-03']
df=df[df.date!='2022-01-02']

temp = df.groupby(['Province_State', 'date', ])['Confirmed', 'Death']
temp = temp.sum().diff().reset_index()
mask = temp['Province_State'] != temp['Province_State'].shift(1)
temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Death'] = np.nan
# renaming columns
temp.columns = ['Province_State', 'date', 'new_case', 'new_death']
df = pd.merge(df, temp, on=['Province_State', 'date'])
df = df.fillna(0)
cols = ['new_case', 'new_death']
df[cols] = df[cols].astype('int')
keyword=df['keyword'].unique().tolist()

state=df['Province_State'].unique().tolist()


# for i in state:
#     for j in keyword:
#         newdf = df[(df.keyword == j) ]
#         newdf = newdf[(newdf.Province_State == i) ]
#         print(newdf)
#         fig, ax = plt.subplots(figsize=(16, 10))
#         plt.plot_date(newdf.week_x, newdf.new_case /
#                       newdf.new_case.max(), fmt='-')
#         plt.plot_date(newdf.week_x, newdf.number /
#                       newdf.number.max(), fmt='-')
#         ax.xaxis.set_tick_params(rotation=90, labelsize=10)
#         plt.legend(['New Covid cases (normalized)', 'Google search for '+ str(j)+ '(normalized)'])
#         plt.savefig(str(i)+str(j)+'.png')
#
# df.to_csv('df.csv')
# print(df)
# print(covid_data[])

kwords=df['keyword'].unique().tolist()

states=df['Province_State'].unique().tolist()
#
#
# corrs = []
# corrsdeath=[]
# state=[]
# #
# keyword=[]
# for j in kwords:
#     for i in states:
#         newdf = df[(df.keyword == j) ]
#         newdf = newdf[(newdf.Province_State == i) ]
#         newdf = newdf.fillna(0)
#         case= newdf.new_case / newdf.new_case.max()
#         death= newdf.new_death / newdf.new_death.max()
#         num= newdf.number / newdf.number.max()
#         corr=num.corr(case, method='pearson')
#         corrd=num.corr(death, method='pearson')
#         corrsdeath.append(corrd)
#         corrs.append(corr)
#         state.append(i)
#         keyword.append(j)
# print(len(state))
# print(len(keyword))
# print(len(corrsdeath))
# print(len(corrs))
# correlation = pd.DataFrame(
#     {'state': state,
#      'keyword': keyword,
#      'corr_case':corrs,
#      'corr_death':corrsdeath
#      })
# correlation.to_csv('correlations.csv')

correlation=pd.read_csv('correlations.csv')
covid_data=df.merge(correlation, left_on=[ 'keyword','Province_State'],right_on=[ 'keyword','state'])
covid_data=covid_data
print(covid_data)
X= covid_data[['number','keyword','corr_case','new_case','new_death','Death','corr_death','Confirmed','date']]
print(X)
X=X.groupby(['keyword']).mean().reset_index()
print('HEA',X)
y=X[['new_case']]
X1=X[['number','corr_case','new_death','Death','corr_death']]
cat_cols=['keyword']
DF=pd.get_dummies(X[['keyword']])
X = pd.concat([X1, DF],axis=1)
print(X)
print('fewa')
features = X.columns


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
las = Lasso(alpha = 1.0, random_state = 0, max_iter = 10000)
alphas = 10.**np.arange(-7, 2)
tuned_parameters = [{'alpha': alphas}]


# search = GridSearchCV(reg,
#                       tuned_parameters,
#                       cv = 5, scoring="neg_mean_squared_error",verbose=3
#                       )
# model=search.fit(X_train,y_train)
# print(model.best_params_)
# print(model.best_score_)
# results = model.cv_results_
# for key,value in results.items():
#     print(key, value)
model=las.fit(X_train,y_train)

coefficients = model.coef_
print(coefficients)
importance = np.abs(coefficients)
print(importance)
print(np.array(features)[importance > 0])
print(np.array(features)[importance == 0])

# Training data
pred_train = model.predict(X_train)
from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train, pred_train)
print(mse_train)
mse_test =mean_squared_error(y_test, pred)

print(mse_test)

# Test data
pred = model.predict(X_test)
print('e',pred)
print('h',y_test)
pred.to_csv('prediction.csv', index=True, header=True)
y_test.to_csv('ytest.csv', index=True, header=True)

print(mse_test)
