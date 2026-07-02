import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

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
test_x_scaled = scaled.fit_transform(test_x)

# Load trained model and test accurucy 
with open("stu_model.data","rb") as file :
    model = pickle.load(file)        #ye vhi object hai jisme trained model save hai 
    pred_y = model.predict(test_x_scaled)
    accuracy = accuracy_score(test_y,pred_y)
    print("Accuracy of Testing =",accuracy * 100)