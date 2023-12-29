import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score

art = np.genfromtxt('./data/Art_Art.csv', delimiter=',')
train_features = art[:,0:2048]
train_labels = art[:,2048].astype(np.int16)

realWorld = np.genfromtxt('./data/Art_RealWorld.csv', delimiter=',')
test_features = realWorld[:,0:2048]
test_labels = realWorld[:,2048].astype(np.int16)

model = LogisticRegression(multi_class='ovr')
model = svm.SVC(decision_function_shape='ovr')
model.fit(train_features, train_labels)

y_pred = model.predict(train_features)

t = 0
for i in range(train_labels.shape[0]):
    if y_pred[i]==train_labels[i]:
        t += 1
acc = accuracy_score(train_labels, y_pred)
print('train accuracy: ', acc)
print('train accuracy: ', t/train_labels.shape[0])

y_pred = model.predict(test_features)
t = 0
for i in range(test_labels.shape[0]):
    if y_pred[i]==test_labels[i]:
        t += 1

acc = accuracy_score(test_labels, y_pred)
print('test accuracy: ', acc)
print('test accuracy: ', t/test_labels.shape[0])