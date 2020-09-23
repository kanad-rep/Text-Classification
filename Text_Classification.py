#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import collections

#Loading the training and the test data
train_data = load_files("C:/Users/Kanad Das/Desktop/training/", load_content = True, encoding = 'utf-8', shuffle = False)
train_df = pd.DataFrame(data = train_data.data, columns = ['Data'])
train_df['Class'] = train_data.target
l1 = list(train_df["Class"])
ctr1 = collections.Counter(l1)

test_data = load_files("C:/Users/Kanad Das/Desktop/test/", load_content = True, encoding = 'utf-8', shuffle = False)
test_df = pd.DataFrame(data = test_data.data, columns = ['Data'])
test_df['Class'] = test_data.target
l2 = list(test_df["Class"])
ctr2 = collections.Counter(l2)

#Removing punctuations and stopwprds
def punc(text):
    exclude = set(string.punctuation)
    text = ''.join(i for i in text if i not in exclude) 
    text = ''.join(i for i in text.lower() if i not in stopwords.words('english'))
    return text

train_df['Data'] = train_df['Data'].apply(punc)
test_df['Data'] = test_df['Data'].apply(punc)

#Stemming
snow = SnowballStemmer('english')
def stemming(text):
    words = text.split()
    words = [snow.stem(i) for i in words]
    return words

#Labelling the columns
trn_data = train_df['Data']
tst_data = test_df['Data']
trn_class_label =  train_df['Class']
tst_class_label =  test_df['Class']

#Vocalbulary
C = CountVectorizer()
C1 = C.fit(trn_data)
len(C1.vocabulary_)
C2 = C.fit(tst_data)
len(C2.vocabulary_)

#BernoulliNB
pipe = Pipeline([("vect",CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',BernoulliNB())])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,2),(2,3)),'clf__alpha':(0.0001,0.001,0.01)}]
grid = GridSearchCV(pipe,parameters,cv=5,verbose=2)
grid.fit(trn_data,trn_class_label)
cf= grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(tst_data)

print('The confusion matrix for Bernoulli Naive Bayes is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report for Bernoulli Naive Bayes is as follows:')
print(classification_report(tst_class_label,predicted_class_label))

#MultinomialNB
pipe = Pipeline([('vect',CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',MultinomialNB())])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,1),(2,3)),'clf__alpha':(0.0001,0.001,0.01,0.1)}]
grid = GridSearchCV(pipe,parameters,cv=5, verbose=2)
grid.fit(train_df['Data'],train_df['Class'])
cf= grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(test_df['Data'])


print('The confusion matrix for Multinomial Naive Bayes is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report Multinomial Naive Bayes is as follows:')
print(classification_report(tst_class_label,predicted_class_label))

#knn algorithm
k = []
for i in range(1,30):
    k.append(i)

pipe = Pipeline([('vect',CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',KNeighborsClassifier())])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,1),(1,2)),'clf__n_neighbors':k,'clf__metric':['cosine','euclidean','minkowski']}]
grid = GridSearchCV(pipe,parameters,cv = 5, verbose = 2)
grid.fit(trn_data,trn_class_label)
cf = grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(tst_data)


print('The confusion matrix for knn algorithm is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report for knn algorithm is as follows:')
print(classification_report(tst_class_label,predicted_class_label))

#Logistic Regression
pipe = Pipeline([('vect',CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',LogisticRegression())])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,1),(1,2)),'clf__C':np.logspace(-5,5,10)}]
grid = GridSearchCV(pipe,parameters,cv = 5, verbose = 2)
grid.fit(trn_data,trn_class_label)
cf = grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(tst_data)


print('The confusion matrix for logistic regression is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report for logistic regression is as follows:')
print(classification_report(tst_class_label,predicted_class_label))

#Support Vector Machine without kernel
pipe = Pipeline([('vect',CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',LinearSVC())])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,1),(1,2)),'clf__C':np.logspace(-5,5,10),'clf__tol':(0.0001,0.001,0.01,1.0)}]
grid = GridSearchCV(pipe,parameters,cv = 5, verbose=2)
grid.fit(trn_data,trn_class_label)
cf = grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(tst_data)


print('The confusion matrix for support vector machine is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report for support vector machine is as follows:')
print(classification_report(tst_class_label,predicted_class_label))

#Support Vector Machine using kernel

#Linear kernel
pipe = Pipeline([('vect',CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',SVC(kernel = 'linear'))])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,1),(1,2)),'clf__C':np.logspace(-5,5,10),'clf__tol':(0.0001,0.001,0.01,1.0)}]
grid = GridSearchCV(pipe,parameters,cv = 5, verbose=1)
grid.fit(trn_data,trn_class_label)
cf = grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(tst_data)


print('The confusion matrix for support vector machine is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report for support vector machine is as follows:')
print(classification_report(tst_class_label,predicted_class_label))

#Non-linear kernel
pipe = Pipeline([('vect',CountVectorizer(tokenizer = stemming)),('tfidf',TfidfTransformer()),('clf',SVC(gamma='scale'))])
parameters = [{'vect__max_df':(0.48,0.52,0.54,0.56),'vect__ngram_range':((1,1),(1,2)),'clf__C':np.logspace(-5,5,10),'clf__kernel':('rbf','sigmoid'),'clf__tol':(0.0001,0.001,0.01,1.0)}]
grid = GridSearchCV(pipe,parameters,cv = 5, verbose=1)
grid.fit(trn_data,trn_class_label)
cf = grid.best_estimator_
print ('\n The best grid is as follows: \n')
print (grid.best_params_)
predicted_class_label = cf.predict(tst_data)


print('The confusion matrix for support vector machine is as follows:')
print(confusion_matrix(tst_class_label,predicted_class_label))
print('\n')
print('The classicfication report for support vector machine is as follows:')
print(classification_report(tst_class_label,predicted_class_label))