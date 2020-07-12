########################
#Importing libraries
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as ac,precision_score,recall_score,f1_score as f1
########################
#Feature Extraction
data = pd.read_csv('polarity.csv').iloc[:1500,:]
y = np.array(data['spam'])
X = np.array(data['review'])
#Vectorization
tfidf = TfidfVectorizer(ngram_range = (1,2))
tfidf1 = TfidfVectorizer(ngram_range = (1,3))
x=X
X = tfidf.fit_transform(X)
X1 = tfidf1.fit_transform(x)#Data for SVM
polarity = sparse.csr_matrix(data['polarity'])
X_final = sparse.hstack([X,polarity.reshape(-1,1)], format='csr')#Data for random forest
#Initializing model
clf1=SVC()
clf1.fit(X1,y)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_final, y)
#Training regressor
train_regressor = pd.read_csv('polarity.csv').iloc[1400:1500,:]
y_regg = np.array(train_regressor['spam'])
X_regg = np.array(train_regressor['review'])
x_regg=X_regg
X_regg = tfidf.transform(X_regg)
polarity_regressor = sparse.csr_matrix(train_regressor['polarity'])
X_regg = sparse.hstack([X_regg,polarity_regressor.reshape(-1,1)], format='csr')
X_regg1=tfidf1.transform(x_regg)
val1=clf.predict(X_regg)
val2=clf1.predict(X_regg1)
val3=pd.concat([pd.DataFrame(val1),pd.DataFrame(val2)],axis=1).values
classifier = LinearRegression().fit(val3, y_regg)
#Testing the hybrid model together
final = pd.read_csv('polarity.csv').iloc[1500:,:]
y_test = np.array(final['spam'])
X_test = np.array(final['review'])
x_test=X_test
X_test = tfidf.transform(X_test)
polarity1 = sparse.csr_matrix(final['polarity'])
X_test = sparse.hstack([X_test,polarity1.reshape(-1,1)], format='csr')
X_test1=tfidf1.transform(x_test)
val1=clf.predict(X_test)
val2=clf1.predict(X_test1)
val3=pd.concat([pd.DataFrame(val1),pd.DataFrame(val2)],axis=1).values
predRF=classifier.predict(val3)
predRF=predRF>0.5
print('Accuracy score: {}'.format(ac(y_test, predRF)))
print('F1 score: {}'.format(f1(y_test, predRF)))
