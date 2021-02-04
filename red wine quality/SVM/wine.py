import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

wine = pd.read_csv('winequality-red.csv')

wine1 = wine[(wine['quality'] == 4) | (wine['quality'] == 5) | (wine['quality'] == 6) | (wine['quality'] == 7) ]

from sklearn.model_selection import train_test_split

X = wine1.drop('quality',axis=1)
y = wine1['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#sns.heatmap(wine.corr(), cmap='flare',annot=True)
#wine['quality'].hist(bins=40)
"""sns.set_style('darkgrid')
g = sns.FacetGrid(wine, col='quality')
g.map(plt.scatter,'pH', 'alcohol')
plt.show()"""

from sklearn.svm import SVC
 
model = SVC()

model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

"""print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))"""

from sklearn.model_selection import GridSearchCV
param = {'C': [1,10,100,1000,10000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param, refit=True,verbose=2)
grid.fit(X_train,y_train)

pred1 = grid.predict(X_test)

print(confusion_matrix(y_test,pred1))
print(classification_report(y_test,pred1))
