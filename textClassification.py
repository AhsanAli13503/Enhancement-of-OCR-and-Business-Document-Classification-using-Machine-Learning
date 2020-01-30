from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.externals import joblib

def train_model(valid_y,classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    joblib.dump(classifier, 'model.pkl') 
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

def model_builder():
    labels, texts = [], []
    with open("newdataset.txt") as fp:
        for cnt, line in enumerate(fp):
            a,b = line.split(' ', 1)
            labels.append(a)
            texts.append(b)
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
    print(valid_x)
# label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)    
    valid_y = encoder.fit_transform(valid_y)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(valid_y,naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
                          
    print ("NB, WordLevel TF-IDF: ", accuracy)
    


        
model_builder()