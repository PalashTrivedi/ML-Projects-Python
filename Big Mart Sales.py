'''
The aim is to build a predictive model and find out the sales of each product at a particular store.
Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

We have train (8523) and test (5681) data set, train data set has both input and output variable(s). You need to predict the sales for test data set.

'''
#!/usr/bin/env python
# coding: utf-8

###importing basic modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

bms_train = pd.read_csv("C:\\Users\\Palash\\Downloads\\train_kOBLwZA.csv")
bms_test = pd.read_csv("C:\\Users\\Palash\\Downloads\\test_t02dQwI.csv")

###Checking the train and test files
bms_train.head(2)
bms_test.head(2)

bms_train.shape
bms_test.shape

##Exploratory data analysis
bms_train.hist(figsize= (15,12))

bms_train.info()
bms_train.isnull().sum()

##item_weight and outlet_size contain missing values
##correlation between target and features
'''

ITEM_MRP seems to have a good correlation with targeted 
ITEM_OUTLET_SALES and other columns are not very useful for prediction of target value

'''
corr_matrix = bms_train.corr()
corr_matrix['Item_Outlet_Sales']

##Lets start checking columns relation with target(Item_Outlet_Sales) with first one being
bms_train.Item_Identifier.value_counts()


##item identifier are categorical columns
###Next column is item_fat_content
bms_train.Item_Fat_Content.value_counts()

'''
LF, low fat belong to same category that is Low Fat and reg belong to Regular category so replacing LF, 
low fat and reg to thier category by

'''

bms_train.Item_Fat_Content = bms_train.Item_Fat_Content.replace('LF','Low Fat')
bms_train.Item_Fat_Content = bms_train.Item_Fat_Content.replace('reg','Regular')
bms_train.Item_Fat_Content = bms_train.Item_Fat_Content.replace('low fat','Low Fat')

bms_train.Item_Fat_Content.value_counts()


bms_train.dtypes


bms_train.Item_Identifier = bms_train.Item_Identifier.astype('category')
bms_train.Item_Fat_Content=bms_train.Item_Fat_Content.astype('category')
bms_train.Item_Type=bms_train.Item_Type.astype('category')
bms_train.Outlet_Identifier=bms_train.Outlet_Identifier.astype('category')
bms_train.Outlet_Establishment_Year=bms_train.Outlet_Establishment_Year.astype('int64')
bms_train.Outlet_Type=bms_train.Outlet_Type.astype('category')
bms_train.Outlet_Location_Type=bms_train.Outlet_Location_Type.astype('category')
bms_train.Outlet_Size=bms_train.Outlet_Size.astype('category')


bms_train.dtypes


##ITEM_MRP has a good strength in relation to target column
fig,axes = plt.subplots(1,1,figsize = (12,8))
sns.scatterplot(x = 'Item_MRP',y = 'Item_Outlet_Sales',hue = 'Item_Fat_Content',size = 'Item_Weight',data = bms_train)

bms_train.describe()

fig,axes = plt.subplots(1,1,figsize = (10,8))
sns.scatterplot(x = 'Item_MRP',y = 'Item_Outlet_Sales',hue = 'Item_Fat_Content',size = 'Item_Weight',data = bms_train)
plt.plot([69,69],[0,5000])
plt.plot([137,137],[0,5000])
plt.plot([203,203],[0,9000])



bms_train.head()



bms_train.Item_MRP = pd.cut(bms_train.Item_MRP,bins = [25,69,137,203,270],labels = ['a','b','c','d'],right = True)



bms_train.head()




fig,axes = plt.subplots(2,2,figsize =(15,12))
sns.boxplot(x = 'Outlet_Establishment_Year',y = "Item_Outlet_Sales",ax = axes[0,0],data = bms_train)
sns.boxplot(x = 'Outlet_Size',y = 'Item_Outlet_Sales',ax = axes[0,1],data = bms_train)
sns.boxplot(x = 'Outlet_Location_Type',y = 'Item_Outlet_Sales',ax = axes[1,0],data = bms_train)
sns.boxplot(x = 'Outlet_Type',y = 'Item_Outlet_Sales',ax = axes[1,1],data = bms_train)




##can drop item_visibility and item_weight as these columns have very low correlation with respect to the target column
##columns for model training will be
attributes=['Item_MRP','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year',
            'Outlet_Identifier','Item_Type','Item_Outlet_Sales']




fig,axes=plt.subplots(2,2,figsize=(15,12))
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,0],data=bms_train)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,1],data=bms_train)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,0],data=bms_train)
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,1],data=bms_train)




data = bms_train[attributes]

data.info()

fig,axes=plt.subplots(1,1,figsize=(8,6))
sns.boxplot(y='Item_Outlet_Sales',hue='Outlet_Type',x='Outlet_Location_Type',data=data)

data[data.Outlet_Size.isnull()]


'''
One thing to observe is when OUTLET_TYPE = supermarket type 1 and 
OUTLET_LOCATION_TYPE is Tier 2 then outlet size is null furthermore when OUTLET_TYPE = Grocery store 
and OUTLET_LOCATION_TYPE is Tier 3 then outlet size is always null
'''
data.groupby('Outlet_Type').get_group('Grocery Store')['Outlet_Location_Type'].value_counts()




##type of categories
print(data['Outlet_Location_Type'].unique())


data.groupby('Outlet_Type').get_group('Grocery Store')


data.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts()


(data.Outlet_Identifier =='OUT010').value_counts()



data.groupby('Outlet_Size').Outlet_Identifier.value_counts()


'''
Tier 1 have small and medium size shop. Tier 2 have small and (missing 1) type shop. 
Tier 3 have 2-medium and 1 high and (missing 2) shop
Tier 2 will have medium size shop in missing 1 and Tier 3 will be high or medium size shop
filling missing values according to the outlet identifier
'''
def func(x):
    if x.Outlet_Identifier == 'OUT010' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT045' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT017' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT013' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT046' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT035' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT019' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT027' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT049' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT018' :
        x.Outlet_Size == 'Medium'
    return(x)




data.Outlet_Size = data.apply(func,axis = 1)


data.head(5)


data.isnull().any()


##checking the outliers
sns.boxplot(x = 'Item_MRP',y = 'Item_Outlet_Sales',data = data)

##Finding maximum sales with respect to the item MRP
data[data.Item_MRP == 'b'].Item_Outlet_Sales.max()

##Finding the rows for which the item outlet sales is maximum
data[data.Item_Outlet_Sales == 7158.6816]

data = data.drop(index = 7796)
data.groupby('Item_MRP').get_group('b')['Item_Outlet_Sales'].max()


sns.boxplot(x = 'Outlet_Type',y = 'Item_Outlet_Sales',data = data)


sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=data)


data[data.Outlet_Location_Type == 'Tier 1'].Item_Outlet_Sales.max()

data[data['Item_Outlet_Sales']==9779.9362]

data = data.drop(index = 4289)

sns.boxplot(x = 'Outlet_Size',y = 'Item_Outlet_Sales',data = data)

sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',data=data)




data.Outlet_Establishment_Year = data.Outlet_Establishment_Year.astype('category')
data_label = data.Item_Outlet_Sales
data_dummy = pd.get_dummies(data.iloc[:,0:6])




data_dummy['Item_Outlet_Sales']= data_label


data_dummy.shape

data_dummy.head(3)

##Model_building
from sklearn.model_selection import train_test_split




train,test = train_test_split(data_dummy,test_size = 0.20,random_state = 2019)

train.shape
test.shape




train_label = train['Item_Outlet_Sales']
test_label = test['Item_Outlet_Sales']
del train['Item_Outlet_Sales']
del test['Item_Outlet_Sales']


##Applying linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


lr.fit(train,train_label)

from sklearn.metrics import mean_squared_error

predict_lr = lr.predict(test)

mse = mean_squared_error(test_label,predict_lr)
lr_score = np.sqrt(mse)
lr_score

##Cross Validation for Linear Regression
from sklearn.model_selection import cross_val_score
score = cross_val_score(lr,train,train_label,cv = 10,scoring = 'neg_mean_squared_error') 
lr_score_cross = np.sqrt(-score)

np.mean(lr_score_cross),np.std(lr_score_cross)

##Ridge regression
from sklearn.linear_model import Ridge

r = Ridge(alpha = 0.05,solver = 'cholesky')
r.fit(train,train_label)
predict_r  = r.predict(test)
mse = mean_squared_error(test_label,predict_r)
r_score = np.sqrt(mse)
r_score

##cross validating ridge
ridge = Ridge(alpha = 0.05,solver = 'cholesky')
score = cross_val_score(ridge,train,train_label,cv = 10,scoring = 'neg_mean_squared_error')
r_square_cross = np.sqrt(-score)
np.mean(r_square_cross),np.std(r_score_cross)

##LASSO
from sklearn.linear_model import Lasso
l = Lasso(alpha = 0.01)
l.fit(train,train_label)
predict_l = l.predict(test)
mse = mean_squared_error(test_label,predict_l)
l_score  = np.sqrt(mse)
l_score




##cross val LASSO
l = Lasso(alpha  = 0.01)
score = cross_val_score(l,train,train_label,cv = 10,scoring = 'neg_mean_squared_error')
l_score_cross = np.sqrt(-score)
np.mean(l_score_cross),np.std(l_score_cross)




##Elastic Net
from sklearn.linear_model import ElasticNet 




en = ElasticNet(alpha = 0.01,l1_ratio = 0.5)
en.fit(train,train_label)
predict_r = en.predict(test)
mse = mean_squared_error(test_label,predict_r)
en_score  = np.sqrt(mse)
en_score




##cross val Elastic
en=ElasticNet(alpha=0.01,l1_ratio=0.5)
score=cross_val_score(en,train,train_label,cv=10,scoring='neg_mean_squared_error')
en_score_cross=np.sqrt(-score)
np.mean(en_score_cross),np.std(en_score_cross)




##Stochastic Gradient
from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor(penalty='l2',max_iter=100,alpha=0.05)
sgd.fit(train,train_label)
predict_r=sgd.predict(test)
mse=mean_squared_error(test_label,predict_r)
sgd_score=np.sqrt(mse)
sgd_score

##Cross validate stochastic gradient
sgd=SGDRegressor(penalty='l2',max_iter=100,alpha=0.05)
score=cross_val_score(sgd,train,train_label,cv=10,scoring='neg_mean_squared_error')
sgd_score_cross=np.sqrt(-score)
np.mean(sgd_score_cross),np.std(sgd_score_cross)

###SVR
from sklearn.svm import SVR
svm = SVR(epsilon = 15,kernel = 'linear')
svm.fit(train,train_label)
predict_svm = svm.predict(test)
mse = mean_squared_error(test_label,predict_svm)
svm_score = np.sqrt(mse)
svm_score


##Cross Validate SVR
svm=SVR(epsilon=15,kernel='linear')
score=cross_val_score(svm,train,train_label,cv=10,scoring='neg_mean_squared_error')
svm_score_cross=np.sqrt(-score)
np.mean(svm_score_cross),np.std(svm_score_cross)

##Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train,train_label)
predict_dt = dtr.predict(test)
mse = mean_squared_error(test_label,predict_dt)
dtr_score = np.sqrt(mse)
dtr_score

##Cross Validate Decision Tree
dtr = DecisionTreeRegressor()
score = cross_val_score(dtr,train,train_label,cv = 10,scoring = 'neg_mean_squared_error')
dtr_score_cross = np.sqrt(-score)
np.mean(dtr_score_cross),np.std(dtr_score_cross)

##Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(train,train_label)
predict_rf = rf.predict(test)
mse = mean_squared_error(test_label,predict_r)
rf_score = np.sqrt(mse)
rf_score

##Cross Validate Random Forest
rf = RandomForestRegressor()
score = cross_val_score(rf,train,train_label,cv = 10,scoring = 'neg_mean_squared_error')
rf_score_cross = np.sqrt(-score)
np.mean(rf_score_cross),np.std(rf_score_cross)


##Bagging Regression
from sklearn.ensemble import BaggingRegressor

br = BaggingRegressor(max_samples = 100)
br.fit(train,train_label)
score = br.predict(test)
br_score = mean_squared_error(test_label,score)
br_score = np.sqrt(br_score)
br_score

##Cross Validating bagging
br = BaggingRegressor()
score = cross_val_score(br,train,train_label,cv = 10,scoring = 'neg_mean_squared_error')
br_score_cross = np.sqrt(-score)
np.mean(br_score_cross),np.std(br_score_cross)


from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor()
ada.fit(train,train_label)
g = ada.predict(test)
ada_score = mean_squared_error(test_label,g)
ada_score = np.sqrt(ada_score)
ada_score


##Cross Val for Ada Boost
ada=AdaBoostRegressor()
score=cross_val_score(ada,train,train_label,cv=10,scoring='neg_mean_squared_error')
ada_score_cross=np.sqrt(-score)
np.mean(ada_score_cross),np.std(ada_score_cross)




##Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(train,train_label)
gbr_predict = gbr.predict(test)
gb_score = mean_squared_error(test_label,gbr_predict)
gb_score = np.sqrt(gb_score)
gb_score




##Cross Validation for gb
gb=GradientBoostingRegressor()
score=cross_val_score(gb,train,train_label,cv=20,scoring='neg_mean_squared_error')
gb_score_cross=np.sqrt(-score)
np.mean(gb_score_cross),np.std(gb_score_cross)




name=['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',
     'Lasso Regression CV','Elastic Net Regression','Elastic Net Regression CV','SGD Regression','SGD Regression CV',
     'SVM','SVM CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost','Ada Boost CV',
     'Bagging','Bagging CV','Gradient Boost','Gradient Boost CV']




go = pd.DataFrame({'RMSE':[lr_score,lr_score_cross,r_score,r_score_cross,l_score,l_score_cross,en_score,en_score_cross,
                     sgd_score,sgd_score_cross,svm_score,svm_score_cross,dtr_score,dtr_score_cross,rf_score,rf_score_cross,
                     ada_score,ada_score_cross,br_score,br_score_cross,gb_score,gb_score_cross]},index=name)




##applymap function python
go['RMSE'] = go.applymap(lambda x: x.mean())


go.RMSE.sort_values()


fig = plt.figure(figsize = (10,6))
plt.scatter(np.arange(1,100,10),predict_r[0:100:10],color = 'blue')
plt.scatter(np.arange(1,100,10),gbr_predict[0:100:10],color='yellow')
plt.scatter(np.arange(1,100,10),test_label[0:100:10],color='black')
plt.legend(['Random_Forest','Gradient Boosting','Real Value'])




##Doing grid search for gradient boosting

from sklearn.model_selection import GridSearchCV
gb=GradientBoostingRegressor(max_depth=7,n_estimators=200,learning_rate=0.01)
param=[{'min_samples_split':[5,9,13],'max_leaf_nodes':[3,5,7,9],'max_features':[8,10,15,18]}]
gs=GridSearchCV(gb,param,cv=5,scoring='neg_mean_squared_error')
gs.fit(train,train_label)


train.columns
gs.best_estimator_

gb = gs.best_estimator_




##now training our model on the training data
total = pd.concat([train,test],axis = 0,ignore_index = True)
total_label = pd.concat([train_label,test_label],axis= = 0,ignore_index = True)




total_label.shape,total.shape
gb.fit(total,total_label)




##Importing test

bms_test.head()




bms_test.shape

##Test Data Preprocessing
attributes=['Item_MRP',
 'Outlet_Type',
 'Outlet_Size',
 'Outlet_Location_Type',
 'Outlet_Establishment_Year',
 'Outlet_Identifier',
 'Item_Type']

bms_test = bms_test[attributes]
bms_test.shape




bms_test.info()


bms_test.Item_MRP=pd.cut(bms_test.Item_MRP,bins=[25,75,140,205,270],labels=['a','b','c','d'],right=True)
bms_test.Item_Type=bms_test.Item_Type.astype('category')
bms_test.Outlet_Size=bms_test.Outlet_Size.astype('category')
bms_test.Outlet_Identifier=bms_test.Outlet_Identifier.astype('category')
bms_test.Outlet_Establishment_Year=bms_test.Outlet_Establishment_Year.astype('int64')
bms_test.Outlet_Type=bms_test.Outlet_Type.astype('category')
bms_test.Outlet_Location_Type=bms_test.Outlet_Location_Type.astype('category')




bms_test.Outlet_Establishment_Year=bms_test.Outlet_Establishment_Year.astype('category')
bms_test.info()




def func(x):
    if x.Outlet_Identifier == 'OUT010' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT045' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT017' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT013' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT046' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT035' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT019' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT027' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT049' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT018' :
        x.Outlet_Size == 'Medium'
    return(x)


bms_test.Outlet_Size=bms_test.apply(func,axis=1)




bms_test.columns




bms_test_dummy=pd.get_dummies(bms_test.iloc[:,0:6])




bms_test_dummy.columns




##Now price of our test data wih our model

predict=gb.predict(bms_test_dummy)




predict.shape




sample = pd.read_csv('C:\\Users\\Palash\\Downloads\\SampleSubmission_TmnO39y.txt')
sample.head()




del sample['Item_Outlet_Sales']




sample.head()




df = pd.DataFrame({'Item_Outlet_Sales':predict})
corr_ans = pd.concat([sample,df],axis = 1)
corr_ans




corr_ans.to_csv('big_mart_sales_submission.csv',index = True)

