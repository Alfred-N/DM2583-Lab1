import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes_classifier import NaiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB

# read data sets
train = pd.read_csv("train.csv", dtype={"score": np.int32, "text": str})
test = pd.read_csv("test.csv", dtype={"score": np.int32, "text": str})
evaluation = pd.read_csv("evaluation.csv", dtype={"score": np.int32, "text": str})

vec = CountVectorizer()

# choose type of data processing
algorithm = input("Please choose the type of data processing. Choices are 'sklearn' or 'own'.\n").lower()
if algorithm == "own":
    train["text"] = NaiveBayesClassifier.process_strings(train)
    test["text"] = NaiveBayesClassifier.process_strings(test)
    evaluation["text"] = NaiveBayesClassifier.process_strings(evaluation)
    vec = CountVectorizer()
elif algorithm == "sklearn":
    train["text"] = train["text"].str.strip().str.lower()
    test["text"] = test["text"].str.strip().str.lower()
    evaluation["text"] = evaluation["text"].str.strip().str.lower()
    vec = CountVectorizer(stop_words='english')

# transform data
print("Transforming data...")
x_train = vec.fit_transform(train["text"]).toarray()
x_test = vec.transform(test["text"]).toarray()
x_eval = vec.transform(evaluation["text"]).toarray()

# train model
print("Training model...")
model = MultinomialNB()
model.fit(x_train, train["score"].values)

# measure accuracies of models
print("Scoring model...")
acc_train = model.score(x_train, train["score"].values)
acc_test = model.score(x_test, test["score"].values)
acc_eval = model.score(x_eval, evaluation["score"].values)

print(f"Accuracy on training data = {round(acc_train, 3)} %")
print(f"Accuracy on test data = {round(acc_test, 3)} %")
print(f"Accuracy on evaluation data = {round(acc_eval, 3)} %")
