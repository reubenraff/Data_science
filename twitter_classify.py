import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

new_york_tweets = pd.read_json("new_york.json", lines=True)
new_york_tweets

(new_york_tweets.columns)
(new_york_tweets.loc[12]["text"])

london_tweets = pd.read_json("london.json",lines=True)

paris_tweets = pd.read_json("paris.json",lines=True)

new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

train_data, test_data, train_labels, test_labels = train_test_split(all_tweets,labels,test_size=0.2,random_state=1)

counter = CountVectorizer()

counter.fit(train_data)

train_counts = counter.transform(train_data)

test_counts = counter.transform(test_data)

nb_classifier = MultinomialNB()

nb_classifier.fit(train_counts,train_labels)

nb_classifier.score(test_counts,test_labels)

nb_predictions = nb_classifier.predict(test_counts)
