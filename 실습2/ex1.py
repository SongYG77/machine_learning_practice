import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
height_weight = np.loadtxt('heights.csv',delimiter=',')
x = height_weight[:,0]
x = x.reshape(len(x),1)
y = height_weight[:,1]
y = y.reshape(len(y),1)
LR = LinearRegression()
LR.fit(x,y)
yp = LR.predict(x)
plt.plot(x,y,'o')
plt.plot(x,yp)
print("기울기 :",LR.coef_)
print("y절편 :",LR.intercept_)
plt.show()
