import pandas as pd
import numpy as np
import pickle


class NaiveBayesClassifier:
    """
    Own implementation of naive bayes classifier
    """

    def __init__(self, data):
        self.data = data
        self.prior = self.calculate_prior()
        self.conditional = None

    def predict(self, test_data):
        """
        predict using the given test data
        :param test_data: for use in the prediction
        :return: predictions
        """
        test_data["text"] = self.process_strings(test_data)
        n_samples = test_data["text"].index.values[-1] + 1
        classes = 2
        prob = np.zeros((classes, n_samples))
        prob[0, :] += np.log(self.prior[0])
        prob[1, :] += np.log(self.prior[1])
        eps = np.finfo(float).eps
        not_in_vocab = 0
        in_vocab = 0
        for i, text_arr in enumerate(test_data["text"]):
            for word in text_arr:
                for c in range(classes):
                    if word not in self.conditional[str(c)].keys():
                        # prob[c][i] += eps
                        not_in_vocab += 1
                    else:
                        prob[c][i] += np.log(self.conditional[str(c)][word])
                        in_vocab += 1

        predictions = np.argmax(prob, axis=0)

        print("Fractions of words that were not in vocab: " + str(
            round(not_in_vocab / (in_vocab + not_in_vocab), 2) * 100) + "%")
        return predictions

    def calculate_prior(self):
        """
        count frequency of classes 0 and 1
        :return: prior
        """
        prior = [0.5, 0.5]
        prior[1] = self.data["score"].sum() / self.data["score"].index.values[-1]
        prior[0] = 1 - prior[1]
        return prior

    def train(self):
        """
        train the model
        :return:
        """
        file_name = 'conditional_train_full.pickle'
        try:
            with open(file_name, 'rb') as handle:
                self.conditional = pickle.load(handle)
            print(f"Loaded conditional {file_name}. No training needed")
        except IOError:
            print(f"loading conditional {file_name} failed. Starting training")
            self.conditional = self.calculate_conditional()

            try:
                with open(file_name, 'wb') as handle:
                    pickle.dump(self.conditional, handle)
            except IOError:
                print(f"Failed to save conditional to {file_name}. Exiting")

    def calculate_conditional(self):
        """
        extract vocabulary, and frequency of each word given the class
        :return: vocabulary
        """
        vocab = {"0": {}, "1": {}}
        self.data["text"] = self.process_strings(self.data)

        c0_counter = 0
        c1_counter = 0
        for i, text_arr in enumerate(self.data["text"]):
            for word in text_arr:
                if word not in vocab[str(self.data.loc[i, "score"])]:
                    vocab[str(self.data.loc[i, "score"])][word] = 1
                else:
                    vocab[str(self.data.loc[i, "score"])][word] += 1

                if self.data.loc[i, "score"] == 0:
                    c0_counter += 1
                else:
                    c1_counter += 1
            if i % 100 == 0:
                print("Calculating freqs. Progress: " + str(round((i / self.data.index.values[-1]) * 100, 2)) + "%")

        print({k: vocab["0"][k] for k in list(vocab["0"])[:10]})

        check_sum = 0
        check_sum1 = 0
        for key in vocab["0"].keys():
            vocab["0"][key] = vocab["0"][key] / c0_counter
            check_sum += vocab["0"][key]
        for key in vocab["1"].keys():
            vocab["1"][key] = vocab["1"][key] / c1_counter
            check_sum1 += vocab["1"][key]

        print("Check-sum class 0: " + str(check_sum))
        print("Check-sum class 1: " + str(check_sum1))

        return vocab

    @staticmethod
    def process_strings(data):
        """
        prepare the string by cleaning and normalizing it
        :param data: data to be pre processed
        :return: processed data
        """
        series = pd.Series(data["text"], dtype="string")

        # remove html tags
        series = series.str.replace("[<][a-zA-Z]+ [/][>]+", "", case=False, regex=True)

        # normalize string
        series = series.str.lower()
        series = series.str.findall("[a-zA-Z]+")
        # TODO: add emoticons

        return series
