import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import seaborn as sns
sns.set()
import numpy as np

google_data=pd.read_csv('google_data.csv')
covid_data=pd.read_csv('covid_data.csv')
covid_data_death=pd.read_csv('covid_data_death.csv')
df_us_states=pd.read_csv('us_states.csv')
mapped= google_data.merge(df_us_states, left_on='state', right_on='state_ab')
covid_data=covid_data.merge(covid_data_death, on=[ 'Date_Time','Province_State'])
df=mapped.merge(covid_data, left_on=['state_y','date'], right_on=['Province_State','Date_Time'])
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



correlation=pd.read_csv('correlations.csv')
covid_data=df.merge(correlation, left_on=[ 'keyword','Province_State'],right_on=[ 'keyword','state'])
covid_data=covid_data
X= covid_data[['number','keyword','corr_case','new_case','Province_State','new_death','Death','corr_death','Confirmed','date']]
X=X.set_index('Province_State')
shifted = X.groupby(level="Province_State").new_death.shift(1).abs().reset_index()
X['new_death_shift']= X['new_death']

X=X.set_index('date')
X=X.groupby(['keyword']).mean().reset_index()

y=X[['new_case']]
X1=X[['new_death']]
cat_cols=['keyword']
DF=pd.get_dummies(X[['keyword']])
X = pd.concat([X1, DF],axis=1)
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
las = Lasso(alpha = 10, random_state = 0,max_iter=100)
alphas = 10.**np.arange(-7, 2)
tuned_parameters = [{'alpha': alphas}]

search = GridSearchCV(las,
                      tuned_parameters,
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )
model=search.fit(X_train,y_train)
print('best param',model.best_params_)
# print(model.best_score_)
# results = model.cv_results_
# for key,value in results.items():
#     print(key, value)


model=las.fit(X_train,y_train)
coefficients = model.coef_


#features in coefficients
print('coefficients of features',list(zip(coefficients, features)))
importance = np.abs(coefficients)
#print important variables
print('important variables',np.array(features)[importance > 0])
print('dropped variables',np.array(features)[importance == 0])

# training mse, r2
pred_train = model.predict(X_train)
from sklearn.metrics import mean_squared_error, r2_score
mse_train = mean_squared_error(y_train, pred_train)
y_train=y_train.values.tolist()
r2_train= r2_score(y_train, pred_train)
print('train_R2',r2_train)
print('mse_train',mse_train)


# Test mse,r2
pred_test = model.predict(X_test)
mse_test =mean_squared_error(y_test, pred_test)
y_test=y_test.values.tolist()
r2_test= r2_score(y_test, pred_test)
print('r2 Test',r2_test)
print('mse Test',mse_test)


#prediction with whole data
prediction = model.predict(X)
mse = mean_squared_error(y, prediction)
r2= r2_score(y, prediction)
print('mse',mse)
print('r2',r2)