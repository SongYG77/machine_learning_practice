import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
# 필요한 모듈을 임포트해서 불러온다.
diabetes = datasets.load_diabetes()
#sklearn 모듈에 있는 diabetes 데이터들을 불러와 변수로 선언
diabetes_X_train = diabetes.data[:-20,:]
diabetes_X_test = diabetes.data[-20:,:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
#data와 target에 있는 값들을 test와 train으로 나눌 것이다. test의 개수는 20개이고 train은 전체에 -20개이다
#shape으로 확인 했을 때 422개가 train으로 사용이 된다.
model = LinearRegression()
#선형회귀분석을 위해 model을 만들어 준다.
model.fit(diabetes_X_train, diabetes_y_train)
#만들어 준 모델에 x_trian값과 y_train값으로 채워준다. 그러면 data와 target에 연관된 가장 적절한 식을 찾을 수 있다.
y_pred = model.predict(diabetes_X_test)
#만들어진 적절한 모델에 이제 x_test값을 넣는다. predict를 하면 우리는 모델에서 생성된 추측값 y들을 얻을 수 있다.
plt.plot(diabetes_y_test, y_pred,'.')
#실제 테스트 값과 위에서 만들어진 추측값을 좌표에 찍는다.
x = np.linspace(0, 330, 100)
#0부터 330까지 100개로 나눈 x배열을 만든다.
y = x
plt.plot(x, y)
#x와 y의 값이 같아서 x=y형태의 그래프가 그려진다.
plt.show()
