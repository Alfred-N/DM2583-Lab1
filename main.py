from naive_bayes_classifier import naiveBayesClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("train.csv", dtype={"score":np.int32,"text":str})
test = pd.read_csv("test.csv", dtype={"score":np.int32,"text":str})
eval = pd.read_csv("evaluation.csv", dtype={"score":np.int32,"text":str})

classifier = naiveBayesClassifier(train)
classifier.train()

test_set = train
predictions = classifier.predict(test_set)
accuracy = np.size(np.where(predictions==test_set["score"].values))/(test_set.index.values[-1] + 1)

print(f"Accuracy = {accuracy} %")