from numpy.core.defchararray import array
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

wine = pd.read_csv('winequality-red.csv')
wine1 = wine[(wine['quality'] == 5) | (wine['quality'] == 6) | (wine['quality'] == 7) ]

dummies = pd.get_dummies(wine1['quality'])
wine1 = pd.concat([wine1.drop('quality',axis=1), dummies],axis=1)



from sklearn.model_selection import train_test_split
X = wine1.drop([5,6,7],axis=1).values
y= wine1[[5,6,7]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=30)

#print(X_train)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

seq = Sequential()
seq.add(Dense(11, input_dim = 11, activation='relu'))
#seq.add(Dropout(0.4))

seq.add(Dense(25, activation='relu'))
#seq.add(Dropout(0.1))

seq.add(Dense(25, activation='relu'))
#seq.add(Dropout(0.1))

seq.add(Dense(11, activation='relu'))
#seq.add(Dropout(0.5))

seq.add(Dense(3, activation='softmax'))

seq.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

seq.fit(x=X_train,y=y_train,epochs=250, validation_data=(X_test,y_test),callbacks=[early_stop])

from tensorflow.keras.models import load_model

seq.save('full_wine_project_model.h5')

loss = pd.DataFrame(seq.history.history)
loss.plot()

"""from sklearn.metrics import classification_report,confusion_matrix
pred = seq.predict_classes(X_test)

print(classification_report(y_test,pred)) 
print(confusion_matrix(y_test,pred))"""



#sns.heatmap(wine.corr(), cmap='flare',annot=True)
#wine['quality'].hist(bins=40)
"""sns.set_style('darkgrid')
g = sns.FacetGrid(wine, col='quality')
g.map(plt.scatter,'pH', 'alcohol')"""
plt.show()
