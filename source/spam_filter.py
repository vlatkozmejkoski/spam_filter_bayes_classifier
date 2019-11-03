import math


def read_data(file_path):
    """Reads the data set file line by line
    :param file_path
    :returns list of strings (lines)"""
    file = open(file_path, "r").readlines()
    return file


def format_sentences(data_set):
    """
    Formats the list of lines such that we separate the target values from the features (attributes)
    :param data_set:
    :returns list of tuples, each tuple is a vector in the data set (target, attribute):
    """
    result = []
    for i, sentence in enumerate(data_set):
        vector = sentence.split('\t')
        result.insert(i, (vector[0], vector[1].strip()))
    return result


def confusion_matrix(actual, predicted):
    """
    Confusion matrix for computing TruePositives, TrueNegatives, FalsePositives, FalseNegatives
    Where positive = ham, negative = spam
    :param actual:
    :param predicted:
    :return string of TP,TN,FP,FN:
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(actual)):
        if actual[i][0] == 'spam' and predicted[i] == 'spam':
            TN += 1
        elif actual[i][0] == 'spam' and predicted[i] == 'ham':
            FP += 1
        elif actual[i][0] == 'ham' and predicted[i] == 'ham':
            TP += 1
        elif actual[i][0] == 'ham' and predicted[i] == 'spam':
            FN += 1
    print("TP: %d\t TN: %d\t FP: %d\t FN: %d" % (TP, TN, FP, FN))


class MultinomialNBSpamFilter:
    """
    Class representing the multinomial naive bayes classifier, consists of all the methods and class variables for
    training the model and predicting target values of vectors
    """
    _punctuations = [',', '.', '!', '?']

    def __init__(self, training_set_file):
        self.p_of_ham = 0
        self.p_of_spam = 0
        self.total_spam= 0
        self.total_ham = 0
        self.vocab_freq = {}
        self.training_set_file = training_set_file

    def word_tokenize(self, document):
        """
        Tokenize each text, such that we get each word of the sentence
        While iterating each word, it removes all punctuations at the and of the word
        :param document:
        :return tuple of the target value and the list of words of the sentence (text):
        """
        result = []
        words = document[1].split()
        for word in words:
            list_of_chars = list(word.lower())
            for char in reversed(list_of_chars):
                if char in self._punctuations:
                    list_of_chars.pop()
                else:
                    break
            result.append(''.join(list_of_chars))
        return document[0], result

    def text_pre_processing(self, data):
        """
        Text pre-processing of the data set
        Calls word_tokenize for each record in the data set
        :param data:
        :return refined list of records:
        """
        refined_set = []
        for doc in data:
            tokenized_doc = self.word_tokenize(doc)
            refined_set.append(tokenized_doc)
        return refined_set

    def term_frequency(self, vectors):
        """
        This method calculates the frequency of words in a document (sms text)
        :param vectors:
        :return: List of tuples with the target value and a dictionary of each word with the frequency
        """
        refined_set = []
        for (target, list_of_words) in vectors:
            term_frequencies = {}
            for word in list_of_words:
                if word in term_frequencies.keys():
                    term_frequencies[word] += 1
                else:
                    term_frequencies[word] = 1

            refined_set.append((target, term_frequencies))
        return refined_set

    def build_model(self):
        """
        We are building the model with first reading the data set. Then we format the set such that we get a list
        of tuples representing a vector in the set, contained of the sms text and target class it belongs to.
        We also apply some pre-processing of the text, like removing dots at the end of the word or any punctuations,
        and the words are to lower to avoid one word appearing as two distinct words.
        For results to be more accurate, stopwords should also be removed, but no external libraries are allowed
        for the stopwords set in english.
        We compute log probabilities of the target classes and store them in the class, used later in prediction
        We compute the frequencies of each word appearing in both target values, used to compute the probabilities as
        as specified in the 'classify' function below
        :return vocab_freq:
        """
        data = read_data(self.training_set_file)
        formatted_data = self.text_pre_processing(format_sentences(data))
        targets = [tar[0] for tar in formatted_data]
        self.total_spam = len([t for t in targets if t == 'spam'])
        self.total_ham = len([t for t in targets if t == 'ham'])
        self.p_of_ham = math.log(self.total_ham / len(formatted_data))
        self.p_of_spam = math.log(self.total_spam / len(formatted_data))
        refined_set = self.term_frequency(formatted_data)
        self.vocab_freq['spam'] = {}
        self.vocab_freq['ham'] = {}
        for (target, freq) in refined_set:
            for (k, v) in freq.items():
                if k not in self.vocab_freq[target]:
                    self.vocab_freq[target][k] = v
                else:
                    self.vocab_freq[target][k] += v
        return self.vocab_freq

    def classify(self, test_set):
        """
        Following bayesian theorem P(c|sms) = (P(sms|c)P(c)) / P(sms), where c = 'ham' or c = 'spam'
        In this case we have to extend the formula, because our 'sms' consists of n words, so we are combining individual
        probabilities :
        P(c|(w1,...,wn)) = P((w1,...,wn)|c)(Pc) / P(w1,...,wn)
        Taking into consideration the naivety where each word is independent of one another
        P(c|(w1,...,wn)) = P(c)* П p(wi|c) / P(w1,...,wn), i = 1, ... n;
        We can also take down the denominator since it only scales the numerator.
        For avoiding floating point underflow, we log both sides of the equation
        log(P(c|(w1,...,wn)) = log(P(c)) + Sum(log(P(wi|c)))
        p = p_of_c + p_of_words_c

        :param test_set:
        :return predictions:
        """
        refined_set = self.term_frequency(self.text_pre_processing(test_set))
        predictions = []
        for (target, word_frequencies) in refined_set:
            result_spam = 0
            result_ham = 0
            for word, freq in word_frequencies.items():
                if word in self.vocab_freq['spam'] or word in self.vocab_freq['ham']:
                    ts_word_freq = self.vocab_freq['spam'].get(word, 0)
                    # This part of avoiding log of 0, i am not sure if i am doing it right, since i am adding +1 to the
                    # numerator
                    p_of_word_spam = math.log((ts_word_freq + 1)/ self.total_spam)
                    ts_word_freq = self.vocab_freq['ham'].get(word, 0)
                    p_of_word_ham = math.log((ts_word_freq + 1)/ self.total_ham)
                else:
                    continue
                result_ham += p_of_word_ham
                result_spam += p_of_word_spam

            result_spam += self.p_of_spam
            result_ham += self.p_of_ham

            if result_spam > result_ham:
                predictions.append('spam')
            else:
                predictions.append('ham')
        return predictions


def main():
    sf = MultinomailNBSpamFilter("../data/SMSSpamTrain.txt")
    sf.build_model()
    test_data = read_data("../data/SMSSpamTest.txt")
    formatted_data = format_sentences(test_data)
    predictions = sf.classify(formatted_data)
    extract_text = [data[1] for data in formatted_data]
    predictions_with_text = list(zip(predictions, extract_text))
    print(predictions_with_text)
    confusion_matrix(formatted_data, predictions)




if __name__ == "__main__":
    main()
