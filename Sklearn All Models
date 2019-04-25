# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 08:50:40 2019

@author: K
"""

import matplotlib.pyplot
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
#plt.style.use('dark_background')
k = 5
N = 11

def isInSphere(x,y): #program does nothing as written pow(x-0.5,2) +  pow(y-0.5,2) <= 0.01) | 
    return (pow(x,2) +  pow(y,2) <= 0.49) | (pow(x-1,2) +  pow(y-1,2) <= 0.01)  | (0.1*pow(x-0.2,2) +  0.2*pow(y-0.9,2) <= 0.01) 
def isInQuartdeCercle(x,y): #program does nothing as written
    return (pow(x,2) +  pow(y,2) <= 0.25) 
def isInBoutDePlan(x,y): #program does nothing as written
    return (x - y <= -0.5)
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html
#http://stackoverflow.com/a/2257449/2901002
#a = [i for i in np.arange(0,1.01,0.01)]
df = pd.DataFrame({ 'x' : [],
    'y' : [],
    'valeur' : [],
    'color': [] })
nb_rangees = 20
for i in np.arange(0,1.01,1/nb_rangees):
    for j in np.arange(0,1.01,1/nb_rangees):
        val = 0
        color = 'blue'
        if(isInSphere(i,j)):
            val = 1
            color = 'red'
        df = df.append({'x': i, 'y': j, 'valeur': val, 'color': color}, ignore_index=True)
from sklearn.utils import shuffle
df = df.reindex(np.random.permutation(df.index))
x_train, x_test, y_train, y_test = train_test_split(df[['x','y']], df['valeur'], test_size=0.25)


algo = pd.DataFrame()

print("tree.DecisionTreeClassifier() training")
start = time. time()
clf = tree.DecisionTreeClassifier()
clf.fit(X=x_train, y=y_train)
result = clf.predict(x_test)
score = clf.score(X=x_test, y=y_test)
print(score)
end = time. time()
print("fitted in "+str(end - start))
#0.9952959623676989
algo = algo.append({'name':'DecisionTreeClassifier', 'result':result, 'score': score, 'res': 'DecisionTree Classif ' + format(score*100,".2f") + ' %'}, ignore_index=True)

print("KNeighborsClassifier(n_neighbors=3)")
start = time. time()
clf = KNeighborsClassifier(n_neighbors=2)
# fitting the model
clf.fit(x_train, y_train)
# predict the response
result = clf.predict(x_test)
# evaluate accuracy
score = accuracy_score(y_test, result)
print(accuracy_score(y_test, result))
end = time. time()
print("fitted in "+str(end - start))
#0.9972559780478244
algo = algo.append({'name':'KNeighborsClassifier', 'result':result, 'score': score,'res': 'KNeighbors Classif (n=3) ' + format(score*100,".2f") + ' %'}, ignore_index=True)



clf = GaussianNB()
clf.fit(x_train, y_train)
result = clf.predict(x_test)
score =clf.score(X=x_test, y=y_test)
print(clf.score(X=x_test, y=y_test))
algo = algo.append({'name':'GaussianNB', 'result':result, 'score': score,'res': 'GaussianNB ' + format(score*100,".2f") + ' %'}, ignore_index=True)

#0.9623676989415916
'''

'''
clf = LogisticRegression(solver="lbfgs", multi_class="multinomial")
clf.fit(x_train, y_train)
result = clf.predict(x_test)
score = clf.score(X=x_test, y=y_test)
print(clf.score(X=x_test, y=y_test))
algo = algo.append({'name':'LogisticRegression', 'result':result, 'score': score,'res': 'Logistic Regression ' + format(score*100,".2f") + ' %'}, ignore_index=True)

#0.9619756958055664
'''

'''
#clf = SVC(kernel='linear')  
#0.9658957271658173
clf = SVC(kernel='poly', degree=2)  
#0.9851038808310466
#clf = SVC(kernel='rbf')  
#0.9854958839670718
#clf = SVC(kernel='sigmoid') 
#0.9541356330850647 
clf.fit(x_train, y_train)
result = clf.predict(x_test)

print(clf.score(X=x_test, y=y_test))
score = clf.score(X=x_test, y=y_test)
algo = algo.append({'name':'SVC kernel=rbf', 'result':result, 'score': score,'res': 'SVC kernel=rbf ' + format(score*100,".2f") + ' %'}, ignore_index=True)

'''

'''
clf = RandomForestClassifier() 
clf.fit(x_train, y_train)
result = clf.predict(x_test)
print(clf.score(X=x_test, y=y_test))
score = clf.score(X=x_test, y=y_test)
algo = algo.append({'name':'RandomForestClassifier', 'result':result, 'score': score,'res': 'RandomForest Classif ' + format(score*100,".2f") + ' %'}, ignore_index=True)

#0.9964719717757742
'''



clf =MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100), learning_rate='constant',
       learning_rate_init=0.001, max_iter=400, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
'''

clf = MLPClassifier(hidden_layer_sizes=(20,20,10),max_iter=1000)

clf.fit(x_train, y_train)
result = clf.predict(x_test)
print(clf.score(X=x_test, y=y_test))
score = clf.score(X=x_test, y=y_test)
algo = algo.append({'name':'Neural network', 'result':result, 'score': score,'res': 'Neural network ' + format(score*100,".2f") + ' %'}, ignore_index=True)

#0.9972559780478244
'''

'''
'''
import tensorflow as tf
#mnist = tf.keras.datasets.mnist

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0
#x_train, x_test, y_train, y_test


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(7650, 2)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu,input_shape=(7650, 2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print(model.evaluate(x_test, y_test))

'''
plt.figure(figsize=(7,7))





plt.ylim(0, 1)
plt.xlim(0, 1)
plt.scatter(df['x'],df['y'],s=1,c=df['color'])


#/********************************/
max_accuracy = 0.00
winner = ""
for index, model in algo.iterrows():
    print(model['name'] + " "+str(model['score']))
    if(float(model['score']) > float(max_accuracy)):
        max_accuracy = model['score']
        winner = model['name']
        print("IF "+str(max_accuracy) + " "+winner)
        
xmax = max(algo['score'])
indexx = algo['score'] >= xmax
name_win = algo['name'][indexx]
strx = ''
for n in name_win.items():
    print(n[1])
    strx = strx+" -- "+n[1]

x=algo.index
labels=algo['res']
y=algo['score']*100

plt.figure(figsize=(10,7))
plt.bar(x,y, align='edge', width=0.3,color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(x, labels)
plt.xticks(rotation=45)
plt.title("précision maximale "+format(xmax*100,".2f") + " % => "+strx)



# Create labels
#label = ['n = 6', 'n = 25', 'n = 13', 'n = 36', 'n = 30', 'n = 11', 'n = 16', 'n = 37', 'n = 14', 'n = 4', 'n = 31', 'n = 34']
 
# Text on the top of each barplot
#for i in range(7):
#plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)


plt.subplot_tool()
plt.show()
'''
predictions = result
new = pd.DataFrame({"predit":y_test == predictions})
#new = y_test == predictions
concat_res = pd.concat([x_test, y_test,new], axis=1)
plt.figure(figsize=(7,7))


plt.ylim(0, 1)
plt.xlim(0, 1)
#plt.scatter(df['x'],df['y'],s=1,c=df['color'])

for i,s in concat_res.iterrows():
    #if s.predit == False and s.valeur == 1:
    if s.predit == False:
        if s.valeur == 0:
            plt.scatter(s.x,s.y,s=7,c='springgreen')
        elif s.valeur == 1:
            plt.scatter(s.x,s.y,s=7,c='navy')
    #elif s.predit == True and s.valeur == 0:
        #plt.scatter(s.x,s.y,s=7,c='navy')
        #plt.plot(s.x, s.y, 'ro', color='green',markersize=2)

plt.show()


from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test, predictions) )
Xnew = [[0.2,0.2]]
ynew = clf.predict(Xnew)
print(ynew)
'''
'''
'''
