#60161879 송윤근
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

X1 = np.random.randint(0,101,size = 300).reshape(100,3)
Y1 = np.zeros(100).reshape(100,1)
Data1 = np.concatenate([X1,Y1],axis = 1)
X2 = np.random.randint(50,101,size = 300).reshape(100,3)
Y2 = np.ones(100).reshape(100,1)
Data2 = np.concatenate([X2,Y2],axis = 1)
Data = np.concatenate([Data1,Data2],axis=0)
print(Data.shape)
X = Data[:,:3]
Y = Data[:,3]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model = LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
print(pred)
print(accuracy_score(Y_test,pred))