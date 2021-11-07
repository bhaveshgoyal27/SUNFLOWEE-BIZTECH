from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
api = Api(app)

class NLPModel(object):

    def __init__(self):
        #self.clf = MultinomialNB()
        self.clf = LogisticRegression()
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='TFIDFVectorizer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='SentimentClassifier.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

model=NLPModel()

clf_path = 'classifier_logistic.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        args = parser.parse_args()
        user_query = args['query']

        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)
        # Output 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'
            
        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output


api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
