import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #replace '?' with -99999
df.drop(['id'], 1, inplace=True) #drop the id column

#predict y
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

#define training and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier() #define a classifier
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

#test data tuple (ensure it does not exist in the dataset!)
#minus id and class column
#example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
