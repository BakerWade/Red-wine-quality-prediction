import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

wine = pd.read_csv('winequality-red.csv')

#sns.heatmap(wine.corr(), cmap='flare',annot=True)
#wine['quality'].hist(bins=40)
"""sns.set_style('darkgrid')
g = sns.FacetGrid(wine, col='quality')
g.map(plt.scatter,'pH', 'alcohol')
plt.show()"""

wine1 = wine[(wine['quality'] == 5) | (wine['quality'] == 6) | (wine['quality'] == 7) ]

from sklearn.model_selection import train_test_split

X = wine1.drop('quality',axis=1)
y = wine1['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)
pred = tree.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

"""print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))"""

from sklearn.ensemble import RandomForestClassifier

ran = RandomForestClassifier(n_estimators=500)

ran.fit(X_train,y_train)

pred1 = ran.predict(X_test)

print(confusion_matrix(y_test,pred1))
print(classification_report(y_test,pred1))