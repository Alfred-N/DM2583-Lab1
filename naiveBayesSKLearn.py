from re import VERBOSE
from naive_bayes_classifier import naiveBayesClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes_classifier import naiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB

train = pd.read_csv("train.csv", dtype={"score":np.int32,"text":str})
test = pd.read_csv("test.csv", dtype={"score":np.int32,"text":str})
eval = pd.read_csv("evaluation.csv", dtype={"score":np.int32,"text":str})

#Using SKlearn's data processing
# train["text"] = train["text"].str.strip().str.lower()
# test["text"] = test["text"].str.strip().str.lower()
# vec = CountVectorizer(stop_words='english')

#Using my data processing:
train["text"] = naiveBayesClassifier.processStrings(train).str.join(" ")
test["text"] = naiveBayesClassifier.processStrings(test).str.join(" ")
vec = CountVectorizer()

x_train = vec.fit_transform(train["text"]).toarray()
x_test = vec.transform(test["text"]).toarray()

model = MultinomialNB()
model.fit(x_train, train["score"].values)

acc_train = model.score(x_train, train["score"].values)
acc_test = model.score(x_test, test["score"].values)

print(acc_train)
print(acc_train)

