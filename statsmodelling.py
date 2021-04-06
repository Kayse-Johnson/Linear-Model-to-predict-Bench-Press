#%%
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
#%%
#read in lifting data
df = pd.read_csv('openpowerlifting.csv')
#%%
df.columns
# %%
#observe colums and drop the ones that will not be used in the model
df.columns
cleaned_df = df[['Sex','Event','Equipment','Age','BodyweightKg','Best3SquatKg','Best3BenchKg','Best3DeadliftKg','Tested']]
cleaned_df=cleaned_df[(cleaned_df['Event'] == 'SBD') & (cleaned_df['Tested'] == 'Yes')]
# %%
#Dropped event and tested column after filtering
cleaned_df.drop(['Event','Tested'], axis=1, inplace= True)
#%%
#Remove rows that have mising values
cleaned_df.dropna(inplace=True)
# %%
# Observe all the different options in this column
cleaned_df['Equipment'].unique()

#%%
#Ensure all the datatypes are in the right format
cleaned_df.info()
# %%
#Create an initial model
formula = 'Best3BenchKg ~ ' + ' + '.join(cleaned_df.columns.drop('Best3BenchKg'))


# %%
#Fit the initial model
model = smf.ols(formula,data = cleaned_df).fit()
# %%
#Get a summary of the model
model.summary()

# %%
#Create dummy variables for further analysis
dum = pd.get_dummies(cleaned_df, drop_first=True)
#%%
#Needed so that patchy works well in the formula line below
dum.columns = dum.columns.str.replace('-','_')
dum.columns
# %%
# Show all thE VIFs of all the variable to detect collinearity
vif_lst = []
for col in dum.columns:
    formula = f'{col} ~ ' + ' + '.join(dum.columns.drop(col))
    model = smf.ols(formula,data = dum).fit()
    dic = {col:1/(1-model.rsquared)}
    vif_lst.append(dic)
#%%
vif_lst
# %%
#Create a correlation heatmap
px.imshow(dum.corr(), title = 'Correlation heatmap of the lifting data')
# %%
#Split the data and drop the equipment_single variable
X = dum.copy()
X.drop('Equipment_Single_ply',axis=1, inplace= True)
print(X)
y = dum['Best3BenchKg']
print(y)
X_train, X_test, y_train, y_test=train_test_split(X,y, train_size = .7)
# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
#fit the linear model
formula = 'Best3BenchKg ~ ' + ' + '.join(X.columns.drop('Best3BenchKg'))
model = smf.ols(formula,data = X_train).fit()
model.summary()
#%%
for col in X_train.columns:
    predictions = model.predict(X_train)
    fig = px.scatter(x=X_train[col], y=y_train-predictions,labels={"x":col,"y":"residuals"})
    fig.show()
# %%
# Use the test data to calculate the performance of the model
predictions = model.predict(X_test)
# %%
1-((predictions- y_test)**2).sum() / ((y_test.mean()-y_test)**2).sum()
# %%
