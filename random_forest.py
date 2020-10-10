import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

'''
age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income
'''

income_data = pd.read_csv("income.csv",delimiter=", ",header=0)

income_data.head()

income_data.iloc[0]

income_data['native-country']


labels = income_data[["income"]]

income_data["not_married"] = income_data["marital-status"].apply(lambda x: 1 if x == "Never-married" else 0)

income_data["married"] = income_data["marital-status"].apply(lambda x: 1 if x == "Married-civ-spouse" else 0)


income_data["sex-int"] = income_data["sex"].apply(lambda x: 0 if x == "Male" else 1)

income_data["country-int"] = income_data['native-country'].apply(lambda x: 0 if x == "United-States" else 1)


income_data["Masters-int"] = income_data['education'].apply(lambda x: 1 if x == "Masters" else 0)

income_data["Dr-int"] = income_data["education"].apply(lambda x: 1 if x == "Doctorate" else 0)

income_data["bachelor-int"] = income_data['education'].apply(lambda x: 1 if x == "Bachelors" else 0)


data = income_data[["age","capital-gain","hours-per-week",'country-int',"Dr-int","married"]]


train_data, test_data, train_labels, test_labels = train_test_split(data, labels,random_state=1)


cv = KFold(n_splits=8,random_state=1,shuffle=True)
forest = RandomForestClassifier(random_state=1)


forest.fit(train_data,train_labels)

scores = cross_val_score(forest,train_data,train_labels,scoring="accuracy",cv=cv,n_jobs=1)

print(scores)

print(forest.score(test_data,test_labels))



'''
HS-grad         10501
Some-college     7291
Bachelors        5355
Masters          1723
Assoc-voc        1382
11th             1175
Assoc-acdm       1067
10th              933
7th-8th           646
Prof-school       576
9th               514
12th              433
Doctorate         413
5th-6th           333
1st-4th           168
Preschool          51
'''
