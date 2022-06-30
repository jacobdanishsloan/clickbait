import csv
import re
import nltk
from nltk import FreqDist
from nltk import punkt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
stopwords = stopwords.words("english")
porter = nltk.PorterStemmer()
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import gensim
import random
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("clickbait.csv")
df = df.reset_index(drop=True)

#add column for presence / absence of "!" symbol
substr = "!"
df["Exclaim"] = 0
for i in range(0,len(df)):
    if substr in str(df["Headline"][i]):
        df['Exclaim'][i] = 1

df.loc[df.Clickbait=="No","Clickbait"] = "Non-Clickbait"
df.loc[df.Clickbait=="Yes","Clickbait"] = "Clickbait"


#convert headlines to features
remove_non_alphabets = lambda x: re.sub(r'[^a-zA-Z]',' ',x)
# token alphabets-only list
tokenize = lambda x: word_tokenize(x)
#lemmatize
lemmatizer = WordNetLemmatizer()
lemmtizer = lambda x: [ lemmatizer.lemmatize(word) for word in x ]
remove_stopwords = lambda x : [w for w in x if w not in stopwords]

df["Headline"] = df["Headline"].apply(remove_non_alphabets)
df["Headline"] = df["Headline"].apply(tokenize)
df["Headline"] = df["Headline"].apply(remove_stopwords)
df["Headline"] = df["Headline"].apply(lemmtizer)
df["Headline"] = df["Headline"].apply(lambda x: ' '.join(x))
for i in range(0,len(df)):
    if df["Exclaim"][i] == 1:
        df["Headline"][i] = df["Headline"][i] + " exclamation"

# 70%-30% train-test split
train_corpus, test_corpus, train_labels, test_labels = train_test_split(df["Headline"],
                                                                        df["Clickbait"],
                                                                        test_size=0.3)

#obtain features
#first, we use a simple bag-of-words approach
#we use single-word terms that appear in 5+ headlines(but less than 50% of them)
bow_vectorizer=CountVectorizer(min_df=5, max_df=0.50,max_features=5000,ngram_range=(1,1))
bow_train_features = bow_vectorizer.fit_transform(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)

#generate bag-of-words term-document matrix
count_array = bow_train_features.toarray()
df_bow = pd.DataFrame(data=count_array,columns = bow_vectorizer.get_feature_names())

#next, binary approach; rather than term-document counts, either 0 (absent) or 1 (present)
bin_train_features = bow_train_features
bin_train_features[bin_train_features > 0] = 1
bin_test_features = bow_test_features
bin_test_features[bin_test_features > 0] = 1

#next, we will utilize TF-IDF:
# build tfidf features' vectorizer and get features
#terms must appear in at least 5 documents, less than 50% of documents
tfidf_vectorizer=TfidfVectorizer(min_df=5, 
                                 max_df=0.50,
                                 max_features=5000,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=(1,1))
tfidf_train_features = tfidf_vectorizer.fit_transform(train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)  
count_array1 = tfidf_test_features.toarray()
df_tfidf = pd.DataFrame(data=count_array1,columns = tfidf_vectorizer.get_feature_names())


#import classifying algorithms
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

#Use KNN algorithm and bag of words features
#k=3 found to work well
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(bow_train_features,train_labels)
bow_knn_pred = knn.predict(bow_test_features)
bow_knn_confusion = pd.crosstab(test_labels, bow_knn_pred, rownames = ["KNN BOW Actual"],colnames=["KNN BOW Predicted"])
bow_knn_confusion

#Use KNN algorithm and TF-IDF features
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(tfidf_train_features,train_labels)
tfidf_knn_pred = knn.predict(tfidf_test_features)
tfidf_knn_confusion = pd.crosstab(test_labels, tfidf_knn_pred, rownames = ["KNN TFIDF Actual"],
                                 colnames=["KNN TFIDF Predicted"])
tfidf_knn_confusion

#Use KNN algorithm and binary features
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(bin_train_features,train_labels)
bin_knn_pred = knn.predict(bin_test_features)
bin_knn_confusion = pd.crosstab(test_labels, bin_knn_pred, rownames = ["KNN Bin Actual"],
                               colnames=["KNN Bin Predicted"])
bin_knn_confusion


#Use logistic regression and bag of words features
logr = LogisticRegression()
logr.fit(bow_train_features,train_labels)
bow_logr_pred = logr.predict(bow_test_features)
bow_logr_confusion = pd.crosstab(test_labels, bow_logr_pred, rownames = ["LogReg BOW Actual"], 
                                 colnames = ["LogReg BOW Predicted"])
bow_logr_confusion


#Use logistic regression and TF-IDF features
logr.fit(tfidf_train_features,train_labels)
tfidf_logr_pred = logr.predict(tfidf_test_features)
tfidf_logr_confusion = pd.crosstab(test_labels, tfidf_logr_pred, rownames = ["LogReg TFIDF Actual"],
                                colnames = ["LogReg  P"])
tfidf_logr_confusion

#Use logistic regression and binary features
logr.fit(bin_train_features,train_labels)
bin_logr_pred = logr.predict(bin_test_features)
bin_logr_confusion = pd.crosstab(test_labels, bin_logr_pred, rownames = ["LogReg Bin Actual"],
                                colnames = ["LogReg Bin Predicted"])
bin_logr_confusion

#Use Random Forest with bag of words features
#about 35 estimators worked well
rf = RandomForestClassifier(n_estimators=35)
rf.fit(bow_train_features, train_labels)
bow_rf_pred = rf.predict(bow_test_features)
bow_rf_confusion = pd.crosstab(test_labels, bow_rf_pred, rownames = ["RForest BOW Actual"],
                                colnames = ["RForest BOW Predicted"])
bow_rf_confusion


#Use Random Forest with TF-IDF features
rf = RandomForestClassifier(n_estimators=35)
rf.fit(tfidf_train_features, train_labels)
tfidf_rf_pred = rf.predict(tfidf_test_features)
tfidf_rf_confusion = pd.crosstab(test_labels, tfidf_rf_pred, rownames = ["RForest TFIDF Actual"],
                                colnames = ["RForest TFIDF Predicted"])
tfidf_rf_confusion


#Use Random Forest with binary features
rf = RandomForestClassifier(n_estimators=35)
rf.fit(bin_train_features, train_labels)
bin_rf_pred = rf.predict(bin_test_features)
bin_rf_confusion = pd.crosstab(test_labels, bin_rf_pred, rownames = ["RForest Bin Actual"],
                                colnames = ["RForest Bin Predicted"])
bin_rf_confusion


#Use SVM with bag of words features
#optimized parameters were determined using GridSearchCV()
svc_bow = SVC(C=10, gamma=0.1, kernel='rbf')
svc_bow.fit(bow_train_features, train_labels)
bow_svc_pred = svc_bow.predict(bow_test_features)
bow_svc_confusion = pd.crosstab(test_labels, bow_svc_pred, rownames = ["Support Vector BOW Actual"],
                                colnames = ["Support Vector BOW Predicted"])
bow_svc_confusion


#Use SVM with TF-IDF features
#optimized parameters were determined using GridSearchCV()
svc_tfidf = SVC(C=10, gamma=1, kernel='rbf')
svc_tfidf.fit(tfidf_train_features, train_labels)
tfidf_svc_pred = svc_tfidf.predict(tfidf_test_features)
tfidf_svc_confusion = pd.crosstab(test_labels, tfidf_svc_pred, rownames = ["Support Vector TFIDF Actual"],
                                colnames = ["Support Vector TFIDF Predicted"])
tfidf_svc_confusion

#Use SVM with binary features
#optimized parameters were determined using GridSearchCV()
svc_bin = SVC(C=10, gamma=0.1, kernel='rbf')
svc_bin.fit(bin_train_features, train_labels)
bin_svc_pred = svc_bin.predict(bin_test_features)
bin_svc_confusion = pd.crosstab(test_labels, bin_svc_pred, rownames = ["Support Vector Binary Actual"],
                                colnames = ["Support Vector Binary Predicted"])
bin_svc_confusion

#Use Naive Bayes with bag of words features
nb = MultinomialNB()
nb.fit(bow_train_features,train_labels)
bow_nb_pred = nb.predict(bow_test_features)
bow_nb_confusion = pd.crosstab(test_labels, bow_nb_pred, rownames = ["Naive Bayes BOW Actual"],
                                colnames = ["Naive Bayes BOW Predicted"])
bow_nb_confusion

#Use Naive Bayes with TF-IDF features
nb.fit(tfidf_train_features,train_labels)
tfidf_nb_pred = nb.predict(tfidf_test_features)
tfidf_nb_confusion = pd.crosstab(test_labels, tfidf_nb_pred, rownames = ["Naive Bayes TFIDF Actual"],
                                colnames = ["Naive Bayes TFIDF Predicted"])
tfidf_nb_confusion


#Use Naive Bayes with binary features
nb.fit(bin_train_features,train_labels)
bin_nb_pred = nb.predict(bin_test_features)
bin_nb_confusion = pd.crosstab(test_labels, bin_nb_pred, rownames = ["Naive Bayes Binary Actual"],
                                colnames = ["Naive Bayes Binary Predicted"])
bin_nb_confusion


#10-fold cross validation to evaluate models' accuracy

#generate features from ALL observations 
bow_vectorizer=CountVectorizer(min_df=5, max_df=0.50,max_features=6000,ngram_range=(1,1))
bow_features = bow_vectorizer.fit_transform(df["Headline"])
bin_features = bow_features
bin_features[bin_features > 0] = 1
tfidf_vectorizer=TfidfVectorizer(min_df=5, max_df=0.50,max_features=6000,norm='l2', smooth_idf=True,
                                 use_idf=True, ngram_range=(1,1))
tfidf_features = tfidf_vectorizer.fit_transform(df["Headline"])  
labels = df["Clickbait"]

#cross validation
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold
kf=KFold(n_splits=10)
bow_knn_score = cross_val_score(knn,bow_features,labels,cv=kf)
tfidf_knn_score = cross_val_score(knn,tfidf_features,labels,cv=kf)
bin_knn_score = cross_val_score(knn,bin_features,labels,cv=kf)
bow_nb_score = cross_val_score(nb,bow_features,labels,cv=kf)
tfidf_nb_score = cross_val_score(nb,tfidf_features,labels,cv=kf)
bin_nb_score = cross_val_score(nb,bin_features,labels,cv=kf)
bow_logr_score = cross_val_score(logr,bow_features,labels,cv=kf)
tfidf_logr_score = cross_val_score(logr,tfidf_features,labels,cv=kf)
bin_logr_score = cross_val_score(logr,bin_features,labels,cv=kf)
bow_rf_score = cross_val_score(rf,bow_features,labels,cv=kf)
tfidf_rf_score = cross_val_score(rf,tfidf_features,labels,cv=kf)
bin_rf_score = cross_val_score(rf,bin_features,labels,cv=kf)
bow_svc_score = cross_val_score(svc_bow,bow_features,labels,cv=kf)
tfidf_svc_score = cross_val_score(svc_tfidf,tfidf_features,labels,cv=kf)
bin_svc_score = cross_val_score(svc_bin,bin_features,labels,cv=kf)

#display average scores (average of ten possible "fold" arrangements) as table
eval_data4 = [[bow_knn_score.mean(),tfidf_knn_score.mean(),bin_knn_score.mean()],
              [bow_nb_score.mean(),tfidf_nb_score.mean(),bin_nb_score.mean()],
              [bow_logr_score.mean(),tfidf_logr_score.mean(),bin_logr_score.mean()], 
              [bow_rf_score.mean(),tfidf_rf_score.mean(),bin_rf_score.mean()],
              [bow_svc_score.mean(),tfidf_svc_score.mean(),bin_svc_score.mean()]]
performance_chart4 = pd.DataFrame(data=eval_data4, index=["K Nearest Neighbors","Naive Bayes",
                                "Logistic Regression","Random Forest", "Support Vector Classifier"], 
                                 columns = ["Bag of Words","TFIDF","Binary"])
#highlight maximum value (best performance) in each column
performance_chart4.style.highlight_max(color = 'lightblue', axis=0)


#evaluate performance for all feature/algorithm combos
bow_knn_precision_pos = (bow_knn_confusion["Yes"]["Yes"]/(bow_knn_confusion["Yes"]["Yes"]+bow_knn_confusion["Yes"]["No"]))
bow_knn_precision_neg = (bow_knn_confusion["No"]["No"]/(bow_knn_confusion["No"]["No"]+bow_knn_confusion["No"]["Yes"]))
bow_knn_recall_pos = bow_knn_confusion["Yes"]["Yes"]/(bow_knn_confusion["Yes"]["Yes"] + bow_knn_confusion["No"]["Yes"])
bow_knn_recall_neg = bow_knn_confusion["No"]["No"]/(bow_knn_confusion["No"]["No"] + bow_knn_confusion["Yes"]["No"])
bow_knn_f1_pos = 2*(bow_knn_precision_pos*bow_knn_recall_pos)/(bow_knn_precision_pos+bow_knn_recall_pos)
bow_knn_f1_neg = 2*(bow_knn_precision_neg*bow_knn_recall_neg)/(bow_knn_precision_neg+bow_knn_recall_neg)

tfidf_knn_precision_pos = (tfidf_knn_confusion["Yes"]["Yes"]/(tfidf_knn_confusion["Yes"]["Yes"]+tfidf_knn_confusion["Yes"]["No"]))
tfidf_knn_precision_neg = (tfidf_knn_confusion["No"]["No"]/(tfidf_knn_confusion["No"]["No"]+tfidf_knn_confusion["No"]["Yes"]))
tfidf_knn_recall_pos = tfidf_knn_confusion["Yes"]["Yes"]/(tfidf_knn_confusion["Yes"]["Yes"] + tfidf_knn_confusion["No"]["Yes"])
tfidf_knn_recall_neg = tfidf_knn_confusion["No"]["No"]/(tfidf_knn_confusion["No"]["No"] + tfidf_knn_confusion["Yes"]["No"])
tfidf_knn_f1_pos = 2*(tfidf_knn_precision_pos*tfidf_knn_recall_pos)/(tfidf_knn_precision_pos+tfidf_knn_recall_pos)
tfidf_knn_f1_neg = 2*(tfidf_knn_precision_neg*tfidf_knn_recall_neg)/(tfidf_knn_precision_neg+tfidf_knn_recall_neg)

bin_knn_precision_pos = (bin_knn_confusion["Yes"]["Yes"]/(bin_knn_confusion["Yes"]["Yes"]+bin_knn_confusion["Yes"]["No"]))
bin_knn_precision_neg = (bin_knn_confusion["No"]["No"]/(bin_knn_confusion["No"]["No"]+bin_knn_confusion["No"]["Yes"]))
bin_knn_recall_pos = bin_knn_confusion["Yes"]["Yes"]/(bin_knn_confusion["Yes"]["Yes"] + bin_knn_confusion["No"]["Yes"])
bin_knn_recall_neg = bin_knn_confusion["No"]["No"]/(bin_knn_confusion["No"]["No"] + bin_knn_confusion["Yes"]["No"])
bin_knn_f1_pos = 2*(bin_knn_precision_pos*bin_knn_recall_pos)/(bin_knn_precision_pos+bin_knn_recall_pos)
bin_knn_f1_neg = 2*(bin_knn_precision_neg*bin_knn_recall_neg)/(bin_knn_precision_neg+bin_knn_recall_neg)

bow_nb_precision_pos = (bow_nb_confusion["Yes"]["Yes"]/(bow_nb_confusion["Yes"]["Yes"]+bow_nb_confusion["Yes"]["No"]))
bow_nb_precision_neg = (bow_nb_confusion["No"]["No"]/(bow_nb_confusion["No"]["No"]+bow_nb_confusion["No"]["Yes"]))
bow_nb_recall_pos = bow_nb_confusion["Yes"]["Yes"]/(bow_nb_confusion["Yes"]["Yes"] + bow_nb_confusion["No"]["Yes"])
bow_nb_recall_neg = bow_nb_confusion["No"]["No"]/(bow_nb_confusion["No"]["No"] + bow_nb_confusion["Yes"]["No"])
bow_nb_f1_pos = 2*(bow_nb_precision_pos*bow_nb_recall_pos)/(bow_nb_precision_pos+bow_nb_recall_pos)
bow_nb_f1_neg = 2*(bow_nb_precision_neg*bow_nb_recall_neg)/(bow_nb_precision_neg+bow_nb_recall_neg)

tfidf_nb_precision_pos = (tfidf_nb_confusion["Yes"]["Yes"]/(tfidf_nb_confusion["Yes"]["Yes"]+tfidf_nb_confusion["Yes"]["No"]))
tfidf_nb_precision_neg = (tfidf_nb_confusion["No"]["No"]/(tfidf_nb_confusion["No"]["No"]+tfidf_nb_confusion["No"]["Yes"]))
tfidf_nb_recall_pos = tfidf_nb_confusion["Yes"]["Yes"]/(tfidf_nb_confusion["Yes"]["Yes"] + tfidf_nb_confusion["No"]["Yes"])
tfidf_nb_recall_neg = tfidf_nb_confusion["No"]["No"]/(tfidf_nb_confusion["No"]["No"] + tfidf_nb_confusion["Yes"]["No"])
tfidf_nb_f1_pos = 2*(tfidf_nb_precision_pos*tfidf_nb_recall_pos)/(tfidf_nb_precision_pos+tfidf_nb_recall_pos)
tfidf_nb_f1_neg = 2*(tfidf_nb_precision_neg*tfidf_nb_recall_neg)/(tfidf_nb_precision_neg+tfidf_nb_recall_neg)

bin_nb_precision_pos = (bin_nb_confusion["Yes"]["Yes"]/(bin_nb_confusion["Yes"]["Yes"]+bin_nb_confusion["Yes"]["No"]))
bin_nb_precision_neg = (bin_nb_confusion["No"]["No"]/(bin_nb_confusion["No"]["No"]+bin_nb_confusion["No"]["Yes"]))
bin_nb_recall_pos = bin_nb_confusion["Yes"]["Yes"]/(bin_nb_confusion["Yes"]["Yes"] + bin_nb_confusion["No"]["Yes"])
bin_nb_recall_neg = bin_nb_confusion["No"]["No"]/(bin_nb_confusion["No"]["No"] + bin_nb_confusion["Yes"]["No"])
bin_nb_f1_pos = 2*(bin_nb_precision_pos*bin_nb_recall_pos)/(bin_nb_precision_pos+bin_nb_recall_pos)
bin_nb_f1_neg = 2*(bin_nb_precision_neg*bin_nb_recall_neg)/(bin_nb_precision_neg+bin_nb_recall_neg)

bow_logr_precision_pos = (bow_logr_confusion["Yes"]["Yes"]/(bow_logr_confusion["Yes"]["Yes"]+bow_logr_confusion["Yes"]["No"]))
bow_logr_precision_neg = (bow_logr_confusion["No"]["No"]/(bow_logr_confusion["No"]["No"]+bow_logr_confusion["No"]["Yes"]))
bow_logr_recall_pos = bow_logr_confusion["Yes"]["Yes"]/(bow_logr_confusion["Yes"]["Yes"] + bow_logr_confusion["No"]["Yes"])
bow_logr_recall_neg = bow_logr_confusion["No"]["No"]/(bow_logr_confusion["No"]["No"] + bow_logr_confusion["Yes"]["No"])
bow_logr_f1_pos = 2*(bow_logr_precision_pos*bow_logr_recall_pos)/(bow_logr_precision_pos+bow_logr_recall_pos)
bow_logr_f1_neg = 2*(bow_logr_precision_neg*bow_logr_recall_neg)/(bow_logr_precision_neg+bow_logr_recall_neg)

tfidf_logr_precision_pos = (tfidf_logr_confusion["Yes"]["Yes"]/(tfidf_logr_confusion["Yes"]["Yes"]+tfidf_logr_confusion["Yes"]["No"]))
tfidf_logr_precision_neg = (tfidf_logr_confusion["No"]["No"]/(tfidf_logr_confusion["No"]["No"]+tfidf_logr_confusion["No"]["Yes"]))
tfidf_logr_recall_pos = tfidf_logr_confusion["Yes"]["Yes"]/(tfidf_logr_confusion["Yes"]["Yes"] + tfidf_logr_confusion["No"]["Yes"])
tfidf_logr_recall_neg = tfidf_logr_confusion["No"]["No"]/(tfidf_logr_confusion["No"]["No"] + tfidf_logr_confusion["Yes"]["No"])
tfidf_logr_f1_pos = 2*(tfidf_logr_precision_pos*tfidf_logr_recall_pos)/(tfidf_logr_precision_pos+tfidf_logr_recall_pos)
tfidf_logr_f1_neg = 2*(tfidf_logr_precision_neg*tfidf_logr_recall_neg)/(tfidf_logr_precision_neg+tfidf_logr_recall_neg)

bin_logr_precision_pos = (bin_logr_confusion["Yes"]["Yes"]/(bin_logr_confusion["Yes"]["Yes"]+bin_logr_confusion["Yes"]["No"]))
bin_logr_precision_neg = (bin_logr_confusion["No"]["No"]/(bin_logr_confusion["No"]["No"]+bin_logr_confusion["No"]["Yes"]))
bin_logr_recall_pos = bin_logr_confusion["Yes"]["Yes"]/(bin_logr_confusion["Yes"]["Yes"] + bin_logr_confusion["No"]["Yes"])
bin_logr_recall_neg = bin_logr_confusion["No"]["No"]/(bin_logr_confusion["No"]["No"] + bin_logr_confusion["Yes"]["No"])
bin_logr_f1_pos = 2*(bin_logr_precision_pos*bin_logr_recall_pos)/(bin_logr_precision_pos+bin_logr_recall_pos)
bin_logr_f1_neg = 2*(bin_logr_precision_neg*bin_logr_recall_neg)/(bin_logr_precision_neg+bin_logr_recall_neg)

bow_rf_precision_pos = (bow_rf_confusion["Yes"]["Yes"]/(bow_rf_confusion["Yes"]["Yes"]+bow_rf_confusion["Yes"]["No"]))
bow_rf_precision_neg = (bow_rf_confusion["No"]["No"]/(bow_rf_confusion["No"]["No"]+bow_rf_confusion["No"]["Yes"]))
bow_rf_recall_pos = bow_rf_confusion["Yes"]["Yes"]/(bow_rf_confusion["Yes"]["Yes"] + bow_rf_confusion["No"]["Yes"])
bow_rf_recall_neg = bow_rf_confusion["No"]["No"]/(bow_rf_confusion["No"]["No"] + bow_rf_confusion["Yes"]["No"])
bow_rf_f1_pos = 2*(bow_rf_precision_pos*bow_rf_recall_pos)/(bow_rf_precision_pos+bow_rf_recall_pos)
bow_rf_f1_neg = 2*(bow_rf_precision_neg*bow_rf_recall_neg)/(bow_rf_precision_neg+bow_rf_recall_neg)

tfidf_rf_precision_pos = (tfidf_rf_confusion["Yes"]["Yes"]/(tfidf_rf_confusion["Yes"]["Yes"]+tfidf_rf_confusion["Yes"]["No"]))
tfidf_rf_precision_neg = (tfidf_rf_confusion["No"]["No"]/(tfidf_rf_confusion["No"]["No"]+tfidf_rf_confusion["No"]["Yes"]))
tfidf_rf_recall_pos = tfidf_rf_confusion["Yes"]["Yes"]/(tfidf_rf_confusion["Yes"]["Yes"] + tfidf_rf_confusion["No"]["Yes"])
tfidf_rf_recall_neg = tfidf_rf_confusion["No"]["No"]/(tfidf_rf_confusion["No"]["No"] + tfidf_rf_confusion["Yes"]["No"])
tfidf_rf_f1_pos = 2*(tfidf_rf_precision_pos*tfidf_rf_recall_pos)/(tfidf_rf_precision_pos+tfidf_rf_recall_pos)
tfidf_rf_f1_neg = 2*(tfidf_rf_precision_neg*tfidf_rf_recall_neg)/(tfidf_rf_precision_neg+tfidf_rf_recall_neg)

bin_rf_precision_pos = (bin_rf_confusion["Yes"]["Yes"]/(bin_rf_confusion["Yes"]["Yes"]+bin_rf_confusion["Yes"]["No"]))
bin_rf_precision_neg = (bin_rf_confusion["No"]["No"]/(bin_rf_confusion["No"]["No"]+bin_rf_confusion["No"]["Yes"]))
bin_rf_recall_pos = bin_rf_confusion["Yes"]["Yes"]/(bin_rf_confusion["Yes"]["Yes"] + bin_rf_confusion["No"]["Yes"])
bin_rf_recall_neg = bin_rf_confusion["No"]["No"]/(bin_rf_confusion["No"]["No"] + bin_rf_confusion["Yes"]["No"])
bin_rf_f1_pos = 2*(bin_rf_precision_pos*bin_rf_recall_pos)/(bin_rf_precision_pos+bin_rf_recall_pos)
bin_rf_f1_neg = 2*(bin_rf_precision_neg*bin_rf_recall_neg)/(bin_rf_precision_neg+bin_rf_recall_neg)

bin_svc_precision_pos = (bin_svc_confusion["Yes"]["Yes"]/(bin_svc_confusion["Yes"]["Yes"]+bin_svc_confusion["Yes"]["No"]))
bin_svc_precision_neg = (bin_svc_confusion["No"]["No"]/(bin_svc_confusion["No"]["No"]+bin_svc_confusion["No"]["Yes"]))
bin_svc_recall_pos = bin_svc_confusion["Yes"]["Yes"]/(bin_svc_confusion["Yes"]["Yes"] + bin_svc_confusion["No"]["Yes"])
bin_svc_recall_neg = bin_svc_confusion["No"]["No"]/(bin_svc_confusion["No"]["No"] + bin_svc_confusion["Yes"]["No"])
bin_svc_f1_pos = 2*(bin_svc_precision_pos*bin_svc_recall_pos)/(bin_rf_precision_pos+bin_svc_recall_pos)
bin_svc_f1_neg = 2*(bin_svc_precision_neg*bin_svc_recall_neg)/(bin_rf_precision_neg+bin_svc_recall_neg)

tfidf_svc_precision_pos = (tfidf_svc_confusion["Yes"]["Yes"]/(tfidf_svc_confusion["Yes"]["Yes"]+tfidf_svc_confusion["Yes"]["No"]))
tfidf_svc_precision_neg = (tfidf_svc_confusion["No"]["No"]/(tfidf_svc_confusion["No"]["No"]+tfidf_svc_confusion["No"]["Yes"]))
tfidf_svc_recall_pos = tfidf_svc_confusion["Yes"]["Yes"]/(tfidf_svc_confusion["Yes"]["Yes"] + tfidf_svc_confusion["No"]["Yes"])
tfidf_svc_recall_neg = tfidf_svc_confusion["No"]["No"]/(tfidf_svc_confusion["No"]["No"] + tfidf_svc_confusion["Yes"]["No"])
tfidf_svc_f1_pos = 2*(tfidf_svc_precision_pos*tfidf_svc_recall_pos)/(tfidf_rf_precision_pos+tfidf_svc_recall_pos)
tfidf_svc_f1_neg = 2*(tfidf_svc_precision_neg*tfidf_svc_recall_neg)/(tfidf_rf_precision_neg+tfidf_svc_recall_neg)

bow_svc_precision_pos = (bow_svc_confusion["Yes"]["Yes"]/(bow_svc_confusion["Yes"]["Yes"]+bow_svc_confusion["Yes"]["No"]))
bow_svc_precision_neg = (bow_svc_confusion["No"]["No"]/(bow_svc_confusion["No"]["No"]+bow_svc_confusion["No"]["Yes"]))
bow_svc_recall_pos = bow_svc_confusion["Yes"]["Yes"]/(bow_svc_confusion["Yes"]["Yes"] + bow_svc_confusion["No"]["Yes"])
bow_svc_recall_neg = bow_svc_confusion["No"]["No"]/(bow_svc_confusion["No"]["No"] + bow_svc_confusion["Yes"]["No"])
bow_svc_f1_pos = 2*(bow_svc_precision_pos*bow_svc_recall_pos)/(bow_rf_precision_pos+bow_svc_recall_pos)
bow_svc_f1_neg = 2*(bow_svc_precision_neg*bow_svc_recall_neg)/(bow_rf_precision_neg+bow_svc_recall_neg)

#display average scores (average of ten possible "fold" arrangements) as table
eval_data4 = [[bow_knn_precision_pos,bow_knn_precision_neg,bow_knn_recall_pos,bow_knn_recall_neg,
               bow_knn_f1_pos,bow_knn_f1_neg,"B.O.W.","KNN"],
              [tfidf_knn_precision_pos,tfidf_knn_precision_neg,tfidf_knn_recall_pos,tfidf_knn_recall_neg,
               tfidf_knn_f1_pos,tfidf_knn_f1_neg,"TF-IDF","KNN"],
              [bin_knn_precision_pos,bin_knn_precision_neg,bin_knn_recall_pos,bin_knn_recall_neg,
               bin_knn_f1_pos,bin_knn_f1_neg,"Binary","KNN"],          
              [bow_nb_precision_pos,bow_nb_precision_neg,bow_nb_recall_pos,bow_nb_recall_neg,
               bow_nb_f1_pos,bow_nb_f1_neg,"B.O.W.","Naive Bayes"],
              [tfidf_nb_precision_pos,tfidf_nb_precision_neg,tfidf_nb_recall_pos,tfidf_nb_recall_neg,
               tfidf_nb_f1_pos,tfidf_nb_f1_neg,"TF-IDF","Naive Bayes"],
              [bin_nb_precision_pos,bin_nb_precision_neg,bin_nb_recall_pos,bin_nb_recall_neg,
               bin_nb_f1_pos,bin_nb_f1_neg,"Binary","Naive Bayes"],
             [bow_logr_precision_pos,bow_logr_precision_neg,bow_logr_recall_pos,bow_logr_recall_neg,
               bow_logr_f1_pos,bow_logr_f1_neg,"B.O.W.","Log. Regression"],
             [tfidf_logr_precision_pos,tfidf_logr_precision_neg,tfidf_logr_recall_pos,tfidf_logr_recall_neg,
               tfidf_logr_f1_pos,tfidf_logr_f1_neg,"TF-IDF","Log. Regression"],
             [bin_logr_precision_pos,bin_logr_precision_neg,bin_logr_recall_pos,bin_logr_recall_neg,
               bin_logr_f1_pos,bin_logr_f1_neg,"Binary","Log. Regression"],
            [bow_rf_precision_pos,bow_rf_precision_neg,bow_rf_recall_pos,bow_rf_recall_neg,
               bow_rf_f1_pos,bow_rf_f1_neg,"B.O.W.","Random Forest"],
             [tfidf_rf_precision_pos,tfidf_rf_precision_neg,tfidf_rf_recall_pos,tfidf_rf_recall_neg,
               tfidf_rf_f1_pos,tfidf_rf_f1_neg,"TF-IDF","Random Forest"],
            [bin_rf_precision_pos,bin_rf_precision_neg,bin_rf_recall_pos,bin_rf_recall_neg,
               bin_rf_f1_pos,bin_rf_f1_neg,"Binary","Random Forest"], 
            [bow_svc_precision_pos,bow_svc_precision_neg,bow_svc_recall_pos,bow_svc_recall_neg,
               bow_svc_f1_pos,bow_svc_f1_neg,"B.O.W.","SVC"],   
             [tfidf_svc_precision_pos,tfidf_svc_precision_neg,tfidf_svc_recall_pos,tfidf_svc_recall_neg,
               tfidf_svc_f1_pos,tfidf_svc_f1_neg,"TF-IDF","SVC"],
             [bin_svc_precision_pos,bin_svc_precision_neg,bin_svc_recall_pos,bin_svc_recall_neg,
               bin_svc_f1_pos,bin_svc_f1_neg,"Binary","SVC"],]
        
    
performance_chart5 = pd.DataFrame(data=eval_data4, index=["K Nearest Neighbors (BOW)","K Nearest Neighbors (TF-IDF)",
                                "K Nearest Neighbors (Bin)", "Naive Bayes (BOW)", "Naive Bayes (TF-IDF)",
                                "Naive Bayes (Bin)", "Logistic Regression (BOW)", "Logistic Regression (TF-IDF)",
                                "Logistic Regression (Bin)", "Random Forest (BOW)", "Random Forest (TF-IDF)",
                                "Random Forest (Bin)","Support Vector Classifier (BOW)",
                                "Support Vector Classifier (TF-IDF)","Support Vector Classifier (BOW)"], 
                                 columns = ["Precision Pos.","Precision Neg.","Recall Pos.","Recall Neg.",
                                           "F1 Score Pos.","F1 Score Neg.","Features","Algorithm"])
performance_chart6 = performance_chart5.iloc[:,0:6]
#highlight maximum value (best performance) in each column
performance_chart6
#performance_chart6.style.highlight_max(color = 'lightblue', axis=0)



##########################
#What portion of legitimate outlets' stories are clickbait-y?
mylist = [0,2]
df.iloc[:,mylist]
train_corpus, test_corpus, train_labels, test_labels = train_test_split(df.iloc[:,mylist],
                                                                        df["Clickbait"],
                                                                        test_size=0.3)
test_corpus = test_corpus[test_labels=="Non-Clickbait"]
tfidf_vectorizer=TfidfVectorizer(min_df=5, 
                                 max_df=0.50,
                                 max_features=5000,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=(1,1))
tfidf_train_features = tfidf_vectorizer.fit_transform(train_corpus.iloc[:,0])  
tfidf_test_features = tfidf_vectorizer.transform(test_corpus.iloc[:,0])  
count_array1 = tfidf_test_features.toarray()
df_tfidf = pd.DataFrame(data=count_array1,columns = tfidf_vectorizer.get_feature_names())
svc_tfidf.fit(tfidf_train_features, train_labels)
tfidf_svc_pred = svc_tfidf.predict(tfidf_test_features)

predictions = list(tfidf_svc_pred)
outlets = list(test_corpus.iloc[:,1])
indices = list(range(0,len(predictions)))

data = {'Outlet':outlets,
        'Prediction':predictions}
legit_df = pd.DataFrame(data)
pd.DataFrame(legit_df.groupby(["Prediction","Outlet"])["Prediction"].count())
data = [["Associated Press",506/(506+22)],["Reuters",134/(134+7)],["The Atlantic",99/(99+39)],
       ["The Boston Globe",153/(153+9)],["The Economist",424/(424+7)],["The Guardian",271/(271+19)],
       ["The New York Times",263/(263+99)],["The San Francisco Chronicle",205/(205+7)],
       ["The Washington Post",205/(205+7)]]
clickbaity = pd.DataFrame(data,columns=["Outlet","Percentage Predicted as Non-Clickbait"])
clickbaity = clickbaity.sort_values("Percentage Predicted as Non-Clickbait", ascending=False)

sns.set_theme(style="whitegrid", font_scale=1.25)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 6))

# Load the example car crash dataset
#barplot = sns.load_dataset(clickbaity)

# Plot the total crashes
#sns.set_color_codes("pastel")
sns.barplot(x="Percentage Predicted as Non-Clickbait", y="Outlet", data=clickbaity, color="lightblue")

# Add a legend and informative axis label
ax.set(xlim=(0, 1), ylabel="",
       xlabel="Proportion Classified as Resembling Non-Clickbait")
sns.despine(left=True, bottom=True)
sns.cubehelix_palette(as_cmap=True)