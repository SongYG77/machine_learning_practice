import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
# sklearn 데이터의 숫자 이미지중6번째 클래스이 있는 내용 불러오기
digits = load_digits(n_class=6)
# 숫자의 타겟이1인 인덱스 가지고 오기
one_idx = np.argwhere(digits.target ==1)
# 5x5의 서브 그림을그리기 위한판 마련하기
fig, ax = plt.subplots(5,5, figsize=(6,6))
j =1
lst = []
# 인덱스0 부터 타겟이1인 이미지를 25개만 그리기
#이 코드를 통해 one_idx 138번째에서 1이 25개가 모두 채워진다.
for i in range(int(one_idx.size)):
    if i in one_idx:
        plt.subplot(5,5,j)
        a = digits.images[i] # 1에 해당하는 배열을 따로 변수로 저장
        a = a[1:7, 2:6] #numpy 배열 형태를 슬라이싱을 통해 원하는 부분을 다시 만든다
        lst.append(a)#답을 한번에 저장하기 위해 리스트를 생성
        plt.imshow(a,cmap='binary')#이미지 표시
        j+=1
    if j > 25:
        break
answer = np.array(lst)#원하는 그림들의 리스트를 배열로 변환
print(answer)
plt.show()