from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

app = Flask(__name__)
api = Api(app)

clf_path = 'lr2.pkl'
model = SentenceTransformer('bert-base-nli-mean-tokens')
with open(clf_path, 'rb') as f:
    model1 = pickle.load(f)

parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        args = parser.parse_args()
        user_query = args['query']

        x = model.encode([user_query])
        y = model1.predict(x)
        
        if y[0]==0:
        	a="negative"
        else:
        	a="positive"
        output = {'prediction': a}
        
        return output

api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
        