import nltk
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Classifier:
    def __init__(self):
        self.clf = None
        self.train = self.load_raw_data('datasets/founta')
        self.test = self.load_raw_data('datasets/davidson')

    def load_raw_data(self, path):
        dataset = load_files(path, shuffle=True, encoding='ISO-8859-1')
        return dataset

    def extract_features(self):
        vector = CountVectorizer(
            min_df=2,
            tokenizer=nltk.TweetTokenizer(False).tokenize,
            encoding='ISO-8859-1',
            stop_words=nltk.corpus.stopwords.words('english')
        )
        train_counts = vector.fit_transform(self.train.data)
        test_counts = vector.transform(self.test.data)
        tfidf_transformer = TfidfTransformer()
        train_tfidf = tfidf_transformer.fit_transform(train_counts)
        test_tfidf = tfidf_transformer.transform(test_counts)
        return train_counts, test_counts

    def mnb(self):
        train_tfidf, test_tfidf = self.extract_features()
        train_tfidf, validation_tfidf, train_tfidf_target, validation_tfidf_target = train_test_split(
            train_tfidf, self.train.target, test_size=0.10
        )
        clf = MultinomialNB()
        clf.fit(train_tfidf, train_tfidf_target)
        val_predicted = clf.predict(validation_tfidf)
        test_predicted = clf.predict(test_tfidf)
        print("--- Validation metrics ---")
        print("Accuarcy score:", accuracy_score(validation_tfidf_target, val_predicted))
        print("Confusion matrix:\n", confusion_matrix(validation_tfidf_target, val_predicted))
        print("Classification report:\n", classification_report(validation_tfidf_target, val_predicted))
        print("--- Test metrics ---")
        print("Accuarcy score:", accuracy_score(self.test.target, test_predicted))
        print("Confusion matrix:\n", confusion_matrix(self.test.target, test_predicted))
        print("Classification report:\n", classification_report(self.test.target, test_predicted))

    def svm(self):
        train_tfidf, test_tfidf = self.extract_features()
        train_tfidf, validation_tfidf, train_tfidf_target, validation_tfidf_target = train_test_split(
            train_tfidf, self.train.target, test_size=0.10
        )
        scaler = Normalizer()
        train_tfidf = scaler.transform(train_tfidf)
        validation_tfidf = scaler.transform(validation_tfidf)
        test_tfidf = scaler.transform(test_tfidf)
        clf = SGDClassifier()
        clf.fit(train_tfidf, train_tfidf_target)
        val_predicted = clf.predict(validation_tfidf)
        test_predicted = clf.predict(test_tfidf)
        print("--- Validation metrics ---")
        print("Accuarcy score:", accuracy_score(validation_tfidf_target, val_predicted))
        print("Confusion matrix:\n", confusion_matrix(validation_tfidf_target, val_predicted))
        print("Classification report:\n", classification_report(validation_tfidf_target, val_predicted))
        print("--- Test metrics ---")
        print("Accuarcy score:", accuracy_score(self.test.target, test_predicted))
        print("Confusion matrix:\n", confusion_matrix(self.test.target, test_predicted))
        print("Classification report:\n", classification_report(self.test.target, test_predicted))

    def lr(self):
        train_tfidf, test_tfidf = self.extract_features()
        train_tfidf, validation_tfidf, train_tfidf_target, validation_tfidf_target = train_test_split(
            train_tfidf, self.train.target, test_size=0.10
        )
        clf = LogisticRegression()
        clf.fit(train_tfidf, train_tfidf_target)
        val_predicted = clf.predict(validation_tfidf)
        test_predicted = clf.predict(test_tfidf)
        print("--- Validation metrics ---")
        print("Accuarcy score:", accuracy_score(validation_tfidf_target, val_predicted))
        print("Confusion matrix:\n", confusion_matrix(validation_tfidf_target, val_predicted))
        print("Classification report:\n", classification_report(validation_tfidf_target, val_predicted))
        print("--- Test metrics ---")
        print("Accuarcy score:", accuracy_score(self.test.target, test_predicted))
        print("Confusion matrix:\n", confusion_matrix(self.test.target, test_predicted))
        print("Classification report:\n", classification_report(self.test.target, test_predicted))


clf = Classifier()
print("--- Multinomial Naive Bayes classification ---")
clf.mnb()
print("--- Linear Support Vector Machine classification ---")
clf.svm()
print("--- Logistic Regression classification ---")
clf.lr()
