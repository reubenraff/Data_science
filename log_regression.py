import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")

# Update sex column to numerical
passengers["Sex"] = passengers.Sex.map({"male":1, "female":0})

# Fill the nan values in the age column
passengers.Age = passengers.Age.fillna(passengers.Age.mean())

#print(passengers.Age.isna().any())

# Create a first class column
passengers["FirstClass"] = passengers.Pclass.apply(lambda x: 1 if x == 1 else 0)

#print(passengers["FirstClass"])

# Create a second class column
passengers["SecondClass"] = passengers.Pclass.apply(lambda x: 1 if x == 2 else 0)


# Select the desired features
features = passengers[["Sex","Age","FirstClass","SecondClass"]]

survival = passengers["Survived"]


# Perform train, test, split
X_train, X_test, y_train, y_test = train_test_split(features, survival,test_size=0.3)






# Scale the feature data so it has mean = 0 and standard deviation = 1

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test =scaler.transform(X_test)
# Create and train the model
model = LogisticRegression()
model.fit(X_train,y_train)

# Score the model on the train data
print(model.score(X_train,y_train))
# 0.7800963081861958
# Score the model on the test data
print(model.score(X_test,y_test))
# 0.8507462686567164
# Analyze the coefficients
print(model.coef_)

#[[-1.25793065 -0.45638647  1.04830804  0.51766022]]

# First class is the highest coef_

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([1.0,18.0,1.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack,Rose,You])
#print("pre transform", sample_passengers)
# Scale the sample passenger features
scaler_2 = StandardScaler()

sample_passengers = scaler_2.fit_transform(sample_passengers)

#print("transform",sample_passengers)
# Make survival predictions!
print(model.predict(sample_passengers))

# [0 0 0] die die die

print(model.predict_proba(sample_passengers))
