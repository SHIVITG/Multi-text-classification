import csv
import sys
import pandas as pd
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,log_loss
import warnings
warnings.filterwarnings("ignore")

# other local imports
from utlis.nlp_utlis import *
from utlis.text_analysis import FeatureEng

'''Global Parameters'''
features = FeatureEng
LE = LabelEncoder()
MNB_classifier = MultinomialNB(alpha=0.5)
xgBoost = XGBClassifier()
rest_classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.001,penalty='l2'), n_jobs=-1)
rf = RandomForestClassifier(random_state=3)

model_dict = {'XGBoost Classifier' : xgBoost,
              'Multinomial Naive Bayes' : MNB_classifier,
              'OneVsRest Classifier': rest_classifier,
              'Random Forest': rf,
              'AdaBoost': AdaBoostClassifier(random_state=3),
              'K Nearest Neighbor': KNeighborsClassifier(),
              'Stochastic Gradient Descent' : SGDClassifier(random_state=3, loss='log')}

class AuthPredict():
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True)
        
    def preprocessing(data):
        '''Input: Takes a file location (csv file)
           Returns: A preprocessed/cleaned dataframe required for further process'''
        df = pd.read_csv(data,encoding="utf-8")
        df['cleaned_content'] = df.content.apply(lambda x: features.clean_text(x))
        df['cleaned_content'] = df.content.apply(lambda x: features.remove_stopwords(x))
        df['author_id'] = LE.fit_transform(df['author'])
        return df
    
    def get_author_map(df):
        author_map = {}
        list_of_author = list(df.author.unique())
        for i in list_of_author:
            value =df.loc[df['author'] == i, 'author_id'].iloc[0]
            author_map[i] = value
            print(value)
        return author_map
    
    def split_data(df):
        X_train, X_test, y_train, y_test = train_test_split(df['cleaned_content'],
                                                    df['author_id'], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    #Converting Text into features
    def tfidf_features(self,X_train, X_test):
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        return X_train_vectorized,X_test_vectorized
    
    def model_classifiers_score(model_dict,X_train_vectorized,X_test_vectorized,y_train, y_test):
        model_name, ac_score_list, p_score_list, r_score_list, f1_score_list,log_loss_list = [],[], [], [], [], []
        for k,v in model_dict.items(): 
            model_name.append(k)
            v.fit(X_train_vectorized,y_train)
            predictions = v.predict(X_test_vectorized)
            pred = v.predict_proba(X_test_vectorized)
            ac_score_list.append(accuracy_score(y_test, predictions))
            p_score_list.append(precision_score(y_test, predictions, average='macro'))
            r_score_list.append(recall_score(y_test, predictions, average='macro'))
            f1_score_list.append(f1_score(y_test, predictions, average='macro'))
            log_loss_list.append(log_loss(y_test, pred))
        print("Classifiers used in training: ",model_name)
        model_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list,log_loss_list]).T
        model_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score','log_loss']
        model_df = model_df.sort_values(by='f1_score', ascending=False)
        return model_df
    
    def training(self,train_file):
        print("--------------------Training Started---------------------")
        print("------------------Preprocessing Started------------------")
        df =  AuthPredict.preprocessing(train_file)
        print(df.info())
        print(" Shape of dataframe passed:" ,df.shape)
        print("-------------Preprocessing Ended--------------")
        X_train, X_test, y_train, y_test =  AuthPredict.split_data(df)
        print("X_train shape: ",X_train.shape)
        print("X_test shape: ",X_test.shape)
        print("y_train shape: ",y_train.shape)
        print("y_test shape: ",y_test.shape)
        print("Converting text into vectors..........................")
        X_train_vectorized,X_test_vectorized = AuthPredict.tfidf_features(self,X_train, X_test)
        model_df = AuthPredict.model_classifiers_score(model_dict,X_train_vectorized,X_test_vectorized,y_train, y_test)
        print(model_df.to_markdown())
        print("---------------------Training Complete !!-------------------")
        return model_df
    
    def make_inference(self,content):
        # author_map hard coded but it can be fetched using the get_author_map function
        author_map = {'The Quint': 4,'PTI': 1,'FP Staff': 0,'Press Trust of India': 2,'Scroll Staff': 3}
        query_vector = self.vectorizer.transform([content])
        predicted_author = xgBoost.predict(query_vector)
        author= list(author_map.keys())[list(author_map.values()).index(predicted_author)] 
        return author
    
    def prediction(self,test_file):
        df =  AuthPredict.preprocessing('data/content_author_assignment_train.csv')
        X_train, X_test, y_train, y_test =  AuthPredict.split_data(df)
        X_train_vectorized,X_test_vectorized = AuthPredict.tfidf_features(self,X_train, X_test)
        model_df = AuthPredict.model_classifiers_score(model_dict,X_train_vectorized,X_test_vectorized,y_train, y_test)
        print("--------------------Testing on unseen data-------------------")
        unseen_data = pd.read_csv(test_file)
        unseen_data_raw = unseen_data.copy()
        # Preparing unseen data for inference
        unseen_data['cleaned_content'] = unseen_data.content.apply(lambda x: features.clean_text(x))
        print("Fetching all the predictions..............")
        predicted_author = []
        for content in unseen_data['cleaned_content']:
            pred = AuthPredict.make_inference(self,content)
            predicted_author.append(pred)
        author_predictions = pd.DataFrame({'Content text':unseen_data['content'],
                                           'Author':unseen_data['author'],'Predicted Author':predicted_author})
        print(" Shape of author_prediction:" ,author_predictions.shape)
        author_predictions.to_csv('author_predictions.csv',index=False)
        print("Saved the prediction file to current directory")
        print(" Prediction completed.....Check file for results")
        return author_predictions
    
if  __name__  ==  "__main__" :
    author_predict = AuthPredict()
    train_file = "data/content_author_assignment_train.csv"
    test_file = "data/content_author_assignment_test.csv"
    author_predict.training(train_file)
    author_predict.prediction(train_file)
