from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.externals import joblib


def load_classfication_pred_res(data):
    text=[]
    text.append(data)
    classifier = joblib.load('model.pkl')
    labels, texts = [], []
    with open("newdataset.txt") as fp:
        for cnt, line in enumerate(fp):
            a,b = line.split(' ', 1)
            labels.append(a)
            texts.append(b)
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
     # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    classificationtext =  tfidf_vect.transform(text)
    predictions = classifier.predict(classificationtext)
    if predictions[0] == 0:
        return "invoice"
    elif  predictions[0] == 1:
        return "receipt"
    elif predictions[0] == 2:
        return "letter"
    else:
        return "memo"   
