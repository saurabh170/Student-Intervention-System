import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings('ignore')
import matplotlib
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#Read csv files
d1 = pd.read_csv('./student-mat1.csv')
d2 = pd.read_csv("./student-por1.csv")

print(d1.head())
print(d2.head())

#Uncomment below two lines to take column G1 and G2
d2.drop(['G1','G2'],axis=1,inplace=True)
d1.drop(['G1','G2'],axis=1,inplace=True)

for i in range(len(d2)):
    if d2['G3'].iloc[i] > 9:  #If the grade is greater than 9 then the student is passed otherwise he/she is fail.
        d2['G3'].iloc[i] = "yes"
    else:
        d2['G3'].iloc[i] = "no"


for i in range(len(d1)):
    if d1['G3'].iloc[i] > 9:
        d1['G3'].iloc[i] = "yes"
    else:
        d1['G3'].iloc[i] = "no"

#Rename label column as passed
d1.rename(index=str,columns={'G3':'passed'},inplace=True)
d2.rename(index=str,columns={'G3':'passed'},inplace=True)

student_data = pd.concat([d1,d2]) # Merge d1 and d2 row wise
print(student_data.head())

print(student_data.info())
print(student_data.isnull().sum())

pk.dump(lr,open('Logistic Regression with G1 and G2','wb'))
pk.dump(sv,open('Support Vector Classifier without G1 and G2','wb'))


print(student_data['school'].value_counts())
print(student_data['age'].value_counts())


#Count histogram
columns=student_data.columns[:]
plt.subplots(figsize=(18,55))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    student_data[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()

#Relation of each column with every other column (Pairwise Plot)
sns.pairplot(data=student_data,hue='passed',diag_kind='kde')
plt.show()
df_small = student_data[[
               'passed',
               'Dalc',
               'goout',
               'Medu',
               'Fedu','failures','Walc']]
sns.pairplot(df_small)


# Heatmap
plt.figure(figsize=(15,15))
sns.heatmap(student_data.corr(),annot = True,fmt = ".2f",cbar = True)
plt.xticks(rotation=90)
plt.yticks(rotation = 0)
plt.show()

#Relation between Workday Alcohol Consumption and final result
matplotlib.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.swarmplot(x='Dalc',y='passed',hue='sex', data=student_data,dodge=True)
ax.set_xlabel("Workday Alcohol Consumption", fontsize=24, labelpad = 40)
ax.set_ylabel("Final Grade", fontsize=24, labelpad = 20)
ax.set_xticklabels(['Very Low','Low','Moderate','High','Very High'],rotation=0)
ax.set_title('Alcohol Consumption and School Performance\n')
ax.legend(ncol=2,loc='upper right')
fig.show()

#Relation between Weekend Alcohol Consumption and final result
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.swarmplot(x='Walc',y='passed',hue='sex', data=student_data,dodge=True)
ax.set_xlabel("Weekend Alcohol Consumption", fontsize=24, labelpad = 20)
ax.set_ylabel("Final Grade", fontsize=24, labelpad = 20)
ax.set_xticklabels(['Very Low','Low','Moderate','High','Very High'],rotation=0)
ax.set_title('Alcohol Consumption and School Performance\n')
ax.legend(ncol=2,loc='upper right')
fig.tight_layout()



# Analysis
#Calculate number of students
n_students = len(student_data.index)
# Calculate number of features, excluding the label column
n_features = len(student_data.columns) - 1
# Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'])
# Calculate failing students.
n_failed = len(student_data[student_data['passed'] == 'no'])
# Print the results
print ("Total number of students: {}".format(n_students))
print ("Number of features: {}".format(n_features))
print ("Number of students who passed: {}".format(n_passed))
print ("Number of students who failed: {}".format(n_failed))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[student_data.columns[:-1]]
y_all = student_data[student_data.columns[-1]]

obj_df = student_data.select_dtypes(include=['object']).copy() # Make dataframe consists of only object
obj_df.drop(['passed'],axis=1,inplace=True)
#Label encoding
lb = LabelEncoder()
for i in obj_df.columns:
    obj_df[i] = lb.fit_transform(obj_df[i])

flt_df = student_data.select_dtypes(include=['int','float']).copy() #Make dataframe consists of int and float value
#One Hot encoding
on = OneHotEncoder()
obj_df = on.fit_transform(obj_df).toarray()
flt_arr = np.array(flt_df) #Convert df to array

lb = LabelEncoder() # Label encoding on label
student_data['passed'] = lb.fit_transform(student_data['passed'])

X = np.concatenate((flt_arr,obj_df),axis=1)
y = np.array([student_data['passed']]).T
print("Shape of y: ",y.shape)
print("Shape of X: ",X.shape)



# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=2,stratify=y_all)
# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

#Logistic Regression
lr = LogisticRegression().fit(X_train,y_train)
y_pred = lr.predict(X_train)
print("F1 score of Logistic Regression on trainig set: {}".format(f1_score(y_train, y_pred, pos_label=1))) #F1 score of training set
y_pred = lr.predict(X_test)
print("F1 score of Logistic Regression on test set: {}".format(f1_score(y_test, y_pred, pos_label=1))) #F1 score of test set

# Support Vector classifier
sv = SVC().fit(X_train,y_train)
y_pred = sv.predict(X_train)
print("F1 score of Support Vector Classifier on trainig set: {}".format(f1_score(y_train, y_pred, pos_label=1))) #F1 score of training set
y_pred = sv.predict(X_test)
print("F1 score of Support Vector Classifier on test set: {}".format(f1_score(y_test, y_pred, pos_label=1))) #F1 score of test set

#Navie Bayes
gb = GaussianNB().fit(X_train,y_train)
y_pred = gb.predict(X_train)
print("F1 score of Navie Bayes on trainig set: {}".format(f1_score(y_train, y_pred, pos_label=1))) #F1 score of training set
y_pred = gb.predict(X_test)
print("F1 score of Navie Bayes on test set: {}".format(f1_score(y_test, y_pred, pos_label=1))) #F1 score of test set

# Decision Tree Classifier
dt = DecisionTreeClassifier().fit(X_train,y_train)
y_pred = dt.predict(X_train) # F1 score of trainig set
print("F1 score of Decision tree on trainig set: {}".format(f1_score(y_train, y_pred, pos_label=1)))
y_pred = dt.predict(X_test) # F1 score of test set
print("F1 score of Decision tree on test set: {}".format(f1_score(y_test, y_pred, pos_label=1)))

#Random Forest Classifier
rm = RandomForestClassifier(n_estimators=50).fit(X_train,y_train)
y_pred = rm.predict(X_train)
print("F1 score of Random Forest on training set: {}".format(f1_score(y_train, y_pred, pos_label=1))) # F1 score of trainig set
y_pred = rm.predict(X_test)
print("F1 score of Random Forest on test set: {}".format(f1_score(y_test, y_pred, pos_label=1))) # F1 score of test set
