#!/usr/bin/env python
# coding: utf-8

#Importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
loan_df = pd.read_csv("C:\\Users\\Palash\\Downloads\\train_ctrUa4k.csv")


#checking the dataframe
loan_df.head()


#basic info
loan_df.info()
'''
object: Object format means variables are categorical. 
Categorical variables in our dataset are: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Property_Area, 
Loan_Status.
int64: It represents the integer variables. ApplicantIncome is of this format.
float64: It represents the variable which have some decimal values involved. 
They are also numerical variables. 
Numerical variables in our dataset are: CoapplicantIncome, LoanAmount, Loan_Amount_Term, and Credit_History.

'''


'''
For categorical features we can use frequency table or bar plots which will calculate the number of each category in a particular variable.
For numerical features, probability density plots can be used to look at the distribution of the variable.
For the target variable lets look at the frequency table,precentage distribution and barplots
Frequency table of a variable will give us the count of each category in that variable.
'''

#train["Loan_Status"].size
loan_df["Loan_Status"].count()


##for categorical variables like credit history,employed,married,gender we can take the counts,here i have taken Loan_Status
loan_df['Loan_Status'].value_counts()


#Normalize can be set to True to print proportions instead of number
loan_df['Loan_Status'].value_counts(normalize = True)*100


#Plotting the barplot after normalizing
loan_df['Loan_Status'].value_counts(normalize = True).plot.bar(title = 'Loan_Status')



##Analysis of ordinal dependent variables
loan_df['Dependents'].count()


#Frequencies for dependent variables
loan_df['Dependents'].value_counts()


#Normalizing the dependents
loan_df['Dependents'].value_counts(normalize=True)*100


#Plotting them in bar plot
loan_df['Dependents'].value_counts(normalize = True).plot.bar(title = 'Dependents')



##In the same way we can do analysis on education variable,property area


##Independent Variable (Numerical)
##Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,Applicant income)
'''The data is skewed towards left so it is not normally distributed,boxplot conforming the presence of lot of outliers,
income disparity in the society segregate the applicant income level by education'''
plt.figure(1)
plt.subplot(121)
sns.distplot(loan_df['ApplicantIncome']);

plt.subplot(122)
loan_df['ApplicantIncome'].plot.box(figsize = (16,5))
plt.show()


##higher number of graduates with high number of income
loan_df.boxplot(column = 'ApplicantIncome',by = 'Education')
plt.suptitle(" ")
plt.show()


##Coapplicant income
plt.figure(1)
plt.subplot(121)
sns.distplot(loan_df["CoapplicantIncome"]);

plt.subplot(122)
loan_df["CoapplicantIncome"].plot.box(figsize=(16,5))
plt.show()


##Lets look at the loan amount variable
##Distribution is normal but there are outliers
plt.figure(1)
plt.subplot(121)
df=loan_df.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
loan_df['LoanAmount'].plot.box(figsize=(16,5))

plt.show()


##Loan amount term variable
##outliers present but the distribution is normal
plt.figure(1)
plt.subplot(121)
df = loan_df.dropna()
sns.distplot(df["Loan_Amount_Term"]);

plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16,5))
plt.show()


##Bivariate Analysis
'''
i)Applicants with high income should have more chances of loan approval.

ii)Applicants who have repaid their previous debts should have higher chances of loan approval.

iii)Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.

iv)Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.

'''
##categorical independent variable vs target variable
'''
First of all we will find the relation between target variable and categorical independent variables. 
Let us look at the stacked bar plot now which will give us the proportion of approved and unapproved loans.
'''
##Relation between loan_status and gender

print(pd.crosstab(loan_df['Gender'],loan_df['Loan_Status']))
Gender = pd.crosstab(loan_df['Gender'],loan_df['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis= 0).plot(kind = 'bar',stacked = True,figsize = (4,4))

plt.xlabel('Gender')
plt.ylabel('Perentage')
plt.show()

## from above plot we see mail applicants higher for the approved loans

##Relation between loan status and married
print(pd.crosstab(loan_df["Married"],loan_df["Loan_Status"]))
Married=pd.crosstab(loan_df["Married"],loan_df["Loan_Status"])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Married")
plt.ylabel("Percentage")
plt.show()

##From above plot we see married applicants getting higher number of loan approvals


##Relation between "Loan_Status" and "Dependents"
print(pd.crosstab(loan_df['Dependents'],loan_df["Loan_Status"]))
Dependents = pd.crosstab(loan_df['Dependents'],loan_df["Loan_Status"])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Dependents")
plt.ylabel("Percentage")
plt.show()

#From above plot the Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.


##Relation between "Loan_Status" and "Education"
print(pd.crosstab(loan_df["Education"],loan_df["Loan_Status"]))
Education = pd.crosstab(loan_df["Education"],loan_df["Loan_Status"])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Education")
plt.ylabel("Percentage")
plt.show()

##From the graph we see the proportion of Graduate applicants is higher for the approved loans.


##Relation between "Loan_Status" and "Self_Employed"

print(pd.crosstab(loan_df["Self_Employed"],loan_df["Loan_Status"]))
SelfEmployed = pd.crosstab(loan_df["Self_Employed"],loan_df["Loan_Status"])
SelfEmployed.div(SelfEmployed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Self_Employed")
plt.ylabel("Percentage")
plt.show

##From above there is nothing significant we can infer from Self_Employed vs Loan_Status plot.


##Relation between "Loan_Status" and "Credit_History"

print(pd.crosstab(loan_df["Credit_History"],loan_df["Loan_Status"]))
CreditHistory = pd.crosstab(loan_df["Credit_History"],loan_df["Loan_Status"])
CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Credit_History")
plt.ylabel("Percentage")
plt.show()

##From above plot it seems people with credit history as 1 are more likely to get their loans approved.

##Relation between "Loan_Status" and "Property_Area"

PropertyArea = pd.crosstab(loan_df["Property_Area"],loan_df["Loan_Status"])
PropertyArea.div(PropertyArea.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Property_Area")
plt.ylabel("Loan_Status")
plt.show()

##The proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.

##Numerical Independent variable vs target variable
##Relation between loan_status and income
##finding the mean income of people for which the loan has been approved vs mean for those whose loan has not had been approved
    
loan_df.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


##Above Here the y-axis represents the mean applicant income. We don’t see any change in the mean income. So, let’s 
##make bins for the applicant income variable based on the values in it and analyze the corresponding loan status for each bin.

bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High', 'Very high']
loan_df['Income_bin'] = pd.cut(df['ApplicantIncome'],bins,labels = group)


print(pd.crosstab(loan_df['Income_bin'],loan_df['Loan_Status']))
Income_bin = pd.crosstab(loan_df["Income_bin"],loan_df["Loan_Status"])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("ApplicantIncome")
plt.ylabel("Percentage")
plt.show()
'''
It can be inferred that Applicant income does not affect the chances of loan approval which contradicts our hypothesis in which we assumed 
that if the applicant income is high the chances of loan approval will also be high.

'''


##We will analyze the coapplicant income and loan amount variable in similar way.

bins =  [0,1000,3000,42000]
group = ['Low','Average','High']
loan_df['CoapplicantIncome_bin'] = pd.cut(df['CoapplicantIncome'],bins,labels = group)

print(pd.crosstab(loan_df["CoapplicantIncome_bin"],loan_df["Loan_Status"]))
CoapplicantIncome_Bin = pd.crosstab(loan_df["CoapplicantIncome_bin"],loan_df["Loan_Status"])
CoapplicantIncome_Bin.div(CoapplicantIncome_Bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.xlabel("CoapplicantIncome")
plt.ylabel("Percentage")
plt.show()
'''
It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. 
The possible reason behind this may be that most of the applicants don’t 
have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it.

'''


##Making a new variable to visualize the combined effect of applicants and coapplicant income
loan_df['Total_income'] = loan_df['ApplicantIncome']+loan_df['CoapplicantIncome'] 


bins =[0,2500,4000,6000,81000]
group=['Low','Average','High','Very High']
loan_df["TotalIncome_bin"]=pd.cut(loan_df["Total_income"],bins,labels=group)


print(pd.crosstab(loan_df["TotalIncome_bin"],loan_df["Loan_Status"]))
TotalIncome = pd.crosstab(loan_df["TotalIncome_bin"],loan_df["Loan_Status"])
TotalIncome.div(TotalIncome.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(2,2))
plt.xlabel("TotalIncome")
plt.ylabel("Percentage")
plt.show()
'''
We can see that Proportion of loans getting approved for applicants having low Total_Income is very 
less as compared to that of applicants with Average, High and Very High Income.

'''


##Relation between Loan_Status and Loan amount
bins = [0,100,200,700]
group = ['Low','Average','High']
loan_df['LoanAmount_bin'] = pd.cut(df['LoanAmount'],bins,labels = group)

print(pd.crosstab(loan_df["LoanAmount_bin"],loan_df["Loan_Status"]))
LoanAmount=pd.crosstab(loan_df["LoanAmount_bin"],loan_df["Loan_Status"])
LoanAmount.div(LoanAmount.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.xlabel("LoanAmount")
plt.ylabel("Percentage")
plt.show()
'''
It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount as 
compared to that of High Loan Amount which supports 
our hypothesis in which we considered that the chances of loan approval will be high when the loan amount is less.

'''

'''
Let’s drop the bins which we created for the exploration part. 
We will change the 3+ in dependents variable to 3 to make it a numerical variable

'''

loan_df = loan_df.drop(["Income_bin","CoapplicantIncome_bin","LoanAmount_bin","Total_income","TotalIncome_bin"],axis=1)

loan_df.head(3)


testloan_df = pd.read_csv("C:\\Users\\Palash\\Downloads\\test_lAUu6dG.csv")


loan_df['Dependents'].replace('3+',3,inplace = True)
testloan_df['Dependents'].replace('3+',3,inplace = True)
loan_df['Loan_Status'].replace('N', 0,inplace=True)
loan_df['Loan_Status'].replace('Y', 1,inplace=True)



'''
Now lets look at the correlation between all the numerical variables. 
We will use the heat map to visualize the correlation. 
Heatmaps visualize data through variations in coloring. 
The variables with darker color means their correlation is more.

'''

matrix = loan_df.corr()
f,ax = plt.subplots(figsize = (10,12))
sns.heatmap(matrix,vmax = .8,square = True,cmap = 'BuPu',annot = True)
###Fro above plot we see that the most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History - Loan_Status).

##Missing value and outlier treatment
loan_df.isnull().sum()

'''
We can consider these methods to fill the missing values:

For numerical variables: imputation using mean or median

For categorical variables: imputation using mode

There are very less missing values in Gender, Married, Dependents, Credit_History 
and Self_Employed features so we can fill them using the mode of the features

'''
loan_df['Gender'].fillna(loan_df['Gender'].mode()[0],inplace = True)
loan_df["Married"].fillna(loan_df["Married"].mode()[0],inplace=True)
loan_df['Dependents'].fillna(loan_df["Dependents"].mode()[0],inplace=True)
loan_df["Self_Employed"].fillna(loan_df["Self_Employed"].mode()[0],inplace=True)
loan_df["Credit_History"].fillna(loan_df["Credit_History"].mode()[0],inplace=True)

'''
Now let’s try to find a way to fill the missing values in Loan_Amount_Term. 
We will look at the value count of the Loan amount term variable.

'''
loan_df["Loan_Amount_Term"].value_counts()


'''
It can be seen that in loan amount term variable, the value of 360 is repeating the most. 
So we will replace the missing values in this variable using the mode of this variable.

'''

loan_df["Loan_Amount_Term"].fillna(loan_df["Loan_Amount_Term"].mode()[0],inplace=True)
loan_df["Loan_Amount_Term"].value_counts()

##Loan amount variable
'''
Now we will see the LoanAmount variable. 
As it is a numerical variable, we can use mean or median to impute the missing values.

We will use median to fill the null values as earlier we saw that loan amount have outliers 
so the mean will not be the proper approach as it is highly affected by the presence of outliers.
'''
loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].median(),inplace = True)


##Check whether all the missing values are filled
loan_df.isnull().sum()
testloan_df.isnull().sum()


testloan_df["Gender"].fillna(testloan_df["Gender"].mode()[0],inplace=True)
testloan_df['Dependents'].fillna(testloan_df["Dependents"].mode()[0],inplace=True)
testloan_df["Self_Employed"].fillna(testloan_df["Self_Employed"].mode()[0],inplace=True)
testloan_df["Loan_Amount_Term"].fillna(testloan_df["Loan_Amount_Term"].mode()[0],inplace=True)
testloan_df["Credit_History"].fillna(testloan_df["Credit_History"].mode()[0],inplace=True)
testloan_df["LoanAmount"].fillna(testloan_df["LoanAmount"].median(),inplace=True)


testloan_df.isnull().sum()
sns.distplot(loan_df['LoanAmount'])


loan_df['LoanAmount'].hist(bins = 20)
##from above plot there is right skewness due to outliers

'''
One way to remove the skewness is by doing the log transformation. 
As we take the log transformation, it does not affect the smaller values much, but reduces the larger values.
One way to remove the skewness is by doing the log transformation. As we take the log transformation, 
it does not affect the smaller values much, but reduces the larger values.
we get a distribution similar to normal
'''

loan_df['LoanAmount_log']  = np.log(loan_df['LoanAmount'])
loan_df['LoanAmount_log'].hist(bins = 20)


sns.distplot(loan_df["LoanAmount_log"])

testloan_df["LoanAmount_log"]=np.log(testloan_df["LoanAmount"])
testloan_df['LoanAmount_log'].hist(bins=20)


sns.distplot(testloan_df["LoanAmount_log"])

##Feature Engineering:Total Income,EMI,Balance Income

'''
Total Income - As discussed during bivariate analysis we will combine the Applicant Income and Coapplicant Income. 
If the total income is high, chances of loan approval might also be high.

EMI - EMI is the monthly amount to be paid by the applicant to repay the loan. 
Idea behind making this variable is that people who have high EMI’s might find it difficult to pay back the loan. 
We can calculate the EMI by taking the ratio of loan amount with respect to loan amount term.

Balance Income - This is the income left after the EMI has been paid. 
Idea behind creating this variable is that if this value is high, 
the chances are high that a person will repay the loan and hence increasing the chances of loan approval.


'''
loan_df["TotalIncome"]=loan_df["ApplicantIncome"]+loan_df["CoapplicantIncome"]

loan_df[['TotalIncome']].head()

testloan_df["TotalIncome"]=testloan_df["ApplicantIncome"]+testloan_df["CoapplicantIncome"]

testloan_df[['TotalIncome']].head()

sns.distplot(loan_df['TotalIncome'])
##We can see it is shifted towards left, i.e., the distribution is right skewed. 
##So, let’s take the log transformation to make the distribution normal.


loan_df['TotalIncome_log'] =  np.log(loan_df['TotalIncome'])
sns.distplot(loan_df['TotalIncome_log'])
##Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided.
sns.distplot(testloan_df["TotalIncome"])

testloan_df["TotalIncome_log"] = np.log(testloan_df["TotalIncome"])
sns.distplot(testloan_df["TotalIncome_log"])

##creating EMI feature
loan_df["EMI"]=loan_df["LoanAmount"]/loan_df["Loan_Amount_Term"]
testloan_df["EMI"]=testloan_df["LoanAmount"]/testloan_df["Loan_Amount_Term"]

loan_df[["EMI"]].head()

testloan_df[["EMI"]].head()

sns.distplot(loan_df["EMI"])

sns.distplot(testloan_df["EMI"])

##creating balance income
loan_df['Balance_Income'] = loan_df['TotalIncome']-loan_df['EMI']*1000 ##To make the units equal we multiply with 1000

testloan_df['Balance_Income'] = testloan_df['TotalIncome']-testloan_df['EMI']*1000


# In[81]:


loan_df[["Balance_Income"]].head()


# In[82]:


testloan_df[["Balance_Income"]].head()


# In[83]:


'''
Let us now drop the variables which we used to create these new features. 
Reason for doing this is, the correlation between those old features and 
these new features will be very high and logistic regression assumes that the variables are not highly correlated. 
We also wants to remove the noise from the dataset, so removing correlated features will help in reducing the noise too.

'''
loan_df=loan_df.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)
loan_df.head(2)

testloan_df = testloan_df.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)
testloan_df.head(2)

testloan_df=testloan_df.drop("Loan_ID",axis=1)
testloan_df.head(2)

X =  loan_df.drop('Loan_Status',1)
X.head(2)


Y = loan_df[['Loan_Status']]
Y.head(2)

##Making dummy variables
##Gender_male(1) and Gender_female(0)
loan_df = pd.get_dummies(loan_df)
testloan_df = pd.get_dummies(testloan_df)


X = loan_df.drop('Loan_Status',1)
X.dtypes

Y = loan_df[['Loan_Status']]
Y.head(2)

testloan_df.head(2)

'''
Now we will train the model on training dataset and make predictions for the test dataset. 
But can we validate these predictions? One way of doing this is we can divide our train dataset into 
two parts:train and validation. We can train the model on this train part and using that make predictions 
for the validation part. In this way we can 
validate our predictions as we have the true predictions for the validation part (which we do not have for the test dataset).

'''
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv = train_test_split(X,Y,test_size = 0.3,random_state = 1)


x_train.dtypes

##Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logistic_model = LogisticRegression(random_state = 1)
logistic_model.fit(x_train,y_train)

##Let's predict the loan status for the validation set
pred_cv_logistic = logistic_model.predict(x_cv)

##Now calculate how accurate our predictions are by calculating the accuracy.
score_logistic = accuracy_score(pred_cv_logistic,y_cv)*100
score_logistic

##making predictions for the test dataset
pred_test_logistic = logistic_model.predict(testloan_df)

##Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=1)


tree_model.fit(x_train,y_train)
pred_cv_tree = tree_model.predict(x_cv)

score_tree = accuracy_score(pred_cv_tree,y_cv)*100
score_tree

pred_test_tree = tree_model.predict(testloan_df)    


##Random Forest
from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(random_state = 1,max_depth = 10,n_estimators = 50)

forest_model.fit(x_train,y_train)

pred_cv_forest = forest_model.predict(x_cv)
score_forest = accuracy_score(pred_cv_forest,y_cv)*100
score_forest


pred_test_forest=forest_model.predict(testloan_df)


##RandomForest Grid-Search
##max_depth and n_estimators(maximum depth and number of trees that will be used in the random forest model)
##Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators.
from sklearn.model_selection import GridSearchCV

paramgrid = {'max_depth': list(range(1,20,2)),'n_estimators': list(range(1,200,20))}


grid_search = GridSearchCV(RandomForestClassifier(random_state = 1),paramgrid)


##Fitting the grid search model
grid_search.fit(x_train,y_train)


##Find out the optimized value
grid_search.best_estimator_


'''
The optimized value for the max_depth variable is 3 and for n_estimator is 101,random_state = 1. 
Now let’s build the model using these optimized values.

'''
grid_forest_model = RandomForestClassifier(random_state=1,max_depth=3,n_estimators=101)

grid_forest_model.fit(x_train,y_train)

pred_grid_forest = grid_forest_model.predict(x_cv)

score_grid_forest = accuracy_score(pred_grid_forest,y_cv)*100

score_grid_forest

###Xgboost algorithm

from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators = 50,max_depth = 4)
xgb_model.fit(x_train,y_train)


pred_xgb = xgb_model.predict(x_cv)


score_xgb = accuracy_score(pred_xgb,y_cv)*100


score_xgb


'''
Logistic Regression model gives : 79% prediction accuracy

Decision Tree model gives : 71% prediction accuracy

Random Forest model gives : 78% prediction accuracy

Random Forest with Grid Search model gives : 77% prediction accuracy

XGBClassifier model gives : 78% prediction accuracy

Finding the important feature

Let us find the feature importance now, i.e. which features are most important for this problem. We will use feature_importances_attribute of sklearn to do it.

As 'LogisticRegression' object has no attribute 'featureimportances' so we choose next high accuracy predictive model. Random Forest model is 2nd highest model.

Using Random Forest model we can find out most important feature among the features.

'''
(pd.Series(forest_model.feature_importances_, index=X.columns)
   .nlargest(10)
   .plot(kind='barh')) 