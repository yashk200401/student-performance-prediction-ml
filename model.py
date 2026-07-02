import pickle

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 

df = pd.read_csv("Student_project.csv")
df_clean = df.dropna()
df_clean = df_clean.drop(columns=["student_id","timestamp"])


#preprocessing 

x = df_clean.drop(columns=["performance_risk_level"])
y = df_clean["performance_risk_level"]


#splitting into 80/20
train_x,test_x,train_y,test_y = train_test_split(x,y,train_size=0.8,random_state=20)

# apply scaling 
scaled = StandardScaler()
train_x_scaled = scaled.fit_transform(train_x)

#apply ML algorithm 
model = LogisticRegression()
model.fit(train_x_scaled,train_y)
print("Model Trained successfully")

#Saving the training model 
with open("stu_model.data","wb") as file :
    pickle.dump(model,file)
    print("Trained model saved")