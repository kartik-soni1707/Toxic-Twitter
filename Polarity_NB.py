from statistics import mode as md
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB as clf1 

df=pd.read_csv("ab1.csv",encoding='latin-1')
vect=CountVectorizer()
x=df.iloc[:,1]
y=df.iloc[:,2:-1]
X=vect.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
polarity_train=pd.DataFrame(y_train.iloc[:,0])
polarity_train.reset_index(inplace=True)
polarity_train=polarity_train.iloc[:,1:]
polarity_test=pd.DataFrame(y_test.iloc[:,0])
polarity_test.reset_index(inplace=True)
polarity_test=polarity_test.iloc[:,1:]
y_train=y_train.iloc[:,1]
y_test=y_test.iloc[:,1]
clf=clf1()
clf.fit(X_train,y_train)
predRF1=clf.predict(X_train)
inpt=pd.concat([pd.DataFrame(predRF1),(polarity_train)],axis=1).values
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', 
                     input_dim = 2))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(inpt, y_train, batch_size = 5, epochs = 20)
prediction=pd.DataFrame(clf.predict(X_test))
inpt2=pd.concat([(prediction),(polarity_test)],axis=1).values

predRF=classifier.predict(inpt2)
predRF=predRF>0.5
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score as f1
print('F1 score: {}'.format(f1(y_test, predRF)))
print('Acc score: {}'.format(accuracy_score(y_test, predRF)))
