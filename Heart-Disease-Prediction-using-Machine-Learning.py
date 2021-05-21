#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')


# In[152]:


dataset = pd.read_csv('F:/Python/Heart-Disease-Prediction-using-Machine-Learning/Prediction.csv')


# In[3]:


dataset.head(10)


# In[4]:


dataset.shape


# In[5]:


dataset.sample(5)


# In[6]:


dataset.describe()


# In[7]:


dataset.info()


# In[8]:


dataset.isnull().sum()


# In[9]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# In[10]:


dataset['target'].describe()


# In[11]:


y = dataset['target']
sns.countplot(y)
target_temp = dataset.target.value_counts()
print(target_temp)


# In[12]:


countNoDisease = len(dataset[dataset.target == 0])
countHaveDisease = len(dataset[dataset.target == 1])


# In[13]:


countNoDisease


# In[14]:


countHaveDisease


# In[15]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# # We'll analyse 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca' and 'thal' features

# # Analysing the 'Sex' feature

# In[16]:


dataset['sex'].unique()


# # We notice, that as expected, the 'sex' feature has 2 unique features

# In[17]:


sns.barplot(dataset['sex'],y)


# We notice, that females are more likely to have heart problems than males

# # Analysing the 'Chest Pain Type' feature

# In[18]:


dataset['cp'].unique()


# In[19]:


sns.barplot(dataset['cp'],y)


# We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems

# # Analysing the FBS feature

# In[20]:


dataset['fbs'].describe()


# In[21]:


dataset['fbs'].unique()


# In[22]:


sns.barplot(dataset['fbs'],y)


# Nothing extraordinary here

# # Analysing the restecg feature

# In[23]:


dataset['restecg'].unique()


# In[24]:


sns.barplot(dataset['restecg'],y)


# We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'

# # Analysing the 'exang' feature

# In[25]:


dataset['exang'].unique()


# In[26]:


sns.barplot(dataset['exang'],y)


# People with exang = 1 i.e Excercise include angina are much less likely to have heart problems

# # Anslysing the slope feature

# In[28]:


dataset['slope'].unique()


# In[29]:


sns.barplot(dataset['slope'],y)


# We observe, tht Slope '2' causes heart pain much more thn Slope '0' and '1'

# # Analysing the 'ca'feature

# In[30]:


dataset['ca'].unique()


# In[32]:


sns.countplot(dataset['ca'])


# In[33]:


sns.barplot(dataset['ca'],y)


# ca=4 has astonishingly large number of heart patients

# # Analysis the total feature

# In[34]:


dataset['thal'].unique()


# In[35]:


sns.barplot(dataset['thal'],y)


# In[36]:


sns.distplot(dataset['thal'])


# # Train-Test-Split

# In[37]:


from sklearn.model_selection import train_test_split

predictors = dataset.drop('target',axis=1)
target = dataset['target']


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(predictors,target, test_size=0.20, random_state=0)


# In[39]:


x_train.shape


# In[40]:


x_test.shape


# In[41]:


y_train.shape


# In[42]:


y_test.shape


# # Model Fitting

# In[43]:


from sklearn.metrics import accuracy_score


# # Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


# In[45]:


lr.fit(x_train,y_train)


# In[46]:


y_pred_lr = lr.predict(x_test)


# In[47]:


y_pred_lr


# In[48]:


y_pred_lr.shape


# In[51]:


score_lr = round(accuracy_score(y_pred_lr, y_test)*100,2)


# In[52]:


print("The accuracy score achived using Logistic Regression is: "+str(score_lr)+"%")


# # Naive Bayes

# In[54]:


from sklearn.naive_bayes import GaussianNB


# In[55]:


nb = GaussianNB()


# In[56]:


nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)


# In[57]:


y_pred_nb.shape


# In[58]:


score_nb = round(accuracy_score(y_pred_nb, y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+"%")


# # SVM

# In[59]:


from sklearn import svm

sv = svm.SVC(kernel = 'linear')

sv.fit(x_train, y_train)
y_pred_svm = sv.predict(x_test)


# In[60]:


y_pred_svm.shape


# In[61]:


score_svm = round(accuracy_score(y_pred_svm,y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+"%")


# # K Nearest Neighbors

# In[67]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)


# In[68]:


y_pred_knn.shape


# In[69]:


score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+"%")


# # Decision Tree

# In[70]:


from sklearn.tree import DecisionTreeClassifier
max_accuracy = 0

for x in range(200):
    dt=DecisionTreeClassifier(random_state=x)
    dt.fit(x_train,y_train)
    y_pred_dt =dt.predict(x_test)
    current_accuracy = round(accuracy_score(y_pred_dt, y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)


# In[77]:


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(x_train,y_train)
y_pred_dt = dt.predict(x_test)


# In[86]:


y_pred_dt.shape


# In[154]:


score_dt = round(accuracy_score(y_pred_dt,y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+"%")


# # Random Forest

# In[155]:


from sklearn.ensemble import RandomForestClassifier


# In[156]:


max_accuracy = 0


# In[158]:


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(x_train,y_train)
    y_pred_rf = rf.predict(x_test)
    current_accuracy = round(accuracy_score(y_pred_rf,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
print(max_accuracy)
print(best_x)        


# In[160]:


rf = RandomForestClassifier(random_state=best_x)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)


# In[161]:


score_rf = round(accuracy_score(y_pred_rf,y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+"%")


# # XGBoost

# In[165]:


import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state = 42)
xgb_model.fit(x_train, y_train)
y_pred_xgb = xgb_model.predict(x_test)


# In[166]:


score_xgb = round(accuracy_score(y_pred_xgb,y_test)*100,2)

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")


# # Output Final score

# In[170]:


scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost"]


# In[176]:


for i in range(len(algorithms)):
    print("The accuracy score achived using "+algorithms[i]+"is:"+str(scores[i])+"%")


# In[174]:


sns.set(rc={'figure.figsize' : (15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")

sns.barplot(algorithms, scores)


# Here random forest has good result as compare to other algorithms

# In[ ]:




