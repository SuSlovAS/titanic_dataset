import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt
import seaborn as sns


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Test'] = 0
test['Test'] = 1

#Exploring data
print(train.head())
print(train.tail())
print(train.isnull().sum())
print(train.describe())
print(train.info())

fig_1 = plt.figure(figsize=(10,8))
ax_1 = sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
plt.show()

fig_2 = plt.figure(figsize=(10,8))
ax_2 = sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')
plt.show()

fig_3 = plt.figure(figsize=(10,8))
ax_3 = sns.countplot(x='Survived',data=train,hue='Pclass',palette='rocket')
plt.show()

fig_4 = plt.figure(figsize=(10,8))
ax_4 = sns.distplot(train['Age'].dropna(),kde=False,bins=30)
plt.show()

fig_5 = plt.figure(figsize=(10,8))
ax_5 = sns.boxplot(x='Pclass',y='Age',data=train)
plt.show()

y_train = train['Survived'].drop(train.index[train['Embarked'].isna() == True].tolist(),axis=0)
train.drop(['Survived'],axis=1,inplace=True)
#print(train.index[train['Embarked'].isna() == True].tolist())
df_comb = pd.concat([train,test],axis=0,ignore_index=True)

mean_age_class = {1:np.around(df_comb[df_comb['Pclass'] == 1]['Age'].mean()),
                  2:np.around(df_comb[df_comb['Pclass'] == 2]['Age'].mean()),
                  3:np.around(df_comb[df_comb['Pclass'] == 3]['Age'].mean())}
print(mean_age_class)

#Clearing data
def impute_age(cols):
    
    if pd.isnull(cols[0]):
        if cols[1] == 1:
            return 38#mean_age_class[1]
        elif cols[1] == 2:
            return 30#mean_age_class[2]
        else:
            return 25#mean_age_class[3]
    else:
        return cols[0]

df_comb['Age'] = df_comb[['Age','Pclass']].apply(impute_age,axis=1)          
df_comb['Age'] = df_comb[['Age','Pclass']].apply(impute_age,axis=1) 

fig_6 = plt.figure(figsize=(10,8))
ax_6 = sns.heatmap(df_comb.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
plt.show()

df_comb.drop('Cabin',axis=1,inplace=True)
print(df_comb.info())
df_comb.dropna(inplace=True)

sex = pd.get_dummies(df_comb['Sex'],drop_first=True)
embark = pd.get_dummies(df_comb['Embarked'],drop_first=True)
parch = pd.get_dummies(df_comb['Parch'],drop_first=True)
parch.rename(columns={1:'P1',2:'P2',3:'P3',4:'P4',5:'P5',6:'P6',9:'P9'},inplace=True)
sibsp = pd.get_dummies(df_comb['SibSp'],drop_first=True)
sibsp.rename(columns={1:'S1',2:'S2',3:'S3',4:'S4',5:'S5',8:'S6'},inplace=True)
df_comb = pd.concat([df_comb,sex,embark,parch,sibsp],axis=1)
df_comb.drop(['Sex','Embarked','Name','Ticket','PassengerId','Parch','SibSp'],axis=1,inplace=True)

print(df_comb.head(10))
print(df_comb.tail(10))

X_train = df_comb[df_comb['Test'] == 0].drop('Test',axis=1)
X_test = df_comb[df_comb['Test'] == 1].drop('Test',axis=1)

#Logistic Regression Model
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train,y_train)
predictions = log_model.predict(X_test)

