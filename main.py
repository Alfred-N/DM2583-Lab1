import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
else:
    print("Invalid choice")
    exit(1)

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

# calculate how many predictions were off
count0 = np.size(np.where(test["score"] == 0))
count1 = np.size(np.where(test["score"] == 1))
correct0 = np.size(np.where(np.logical_and(model.predict(x_test) == test["score"].values, test["score"].values == 0)))
correct1 = np.size(np.where(np.logical_and(model.predict(x_test) == test["score"].values, test["score"].values == 1)))

# confusion matrix for sentiment
fig, ax = plt.subplots()
ax.imshow([[count1 - correct1, correct1], [correct0, count0 - correct0]], cmap="PuBu")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["0", "1"])
ax.set_yticklabels(["1", "0"])
plt.xlabel("true sentiment")
plt.ylabel("predicted sentiment")
plt.title("Confusion matrix of true to predicted sentiment")
plt.show()
