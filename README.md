# 서포트 벡터 머신(SVM, Support Vector Machine)
## 1. Introduction
이번 시간에는 지도학습 분류 알고리즘인 서포트 벡터 머신(SVM)과 그 종류에 대해 알아보고 사용자가 조절할 수 있는 하이퍼 파라미터(C, gamma)의 변화에 따른 모델의 변화를 알아보고자 한다.

## 2. 서포트 벡터 머신이란?
서포트 벡터 머신은 두가지 범주에 대해 선형의 경계면을 만들어 두 범주를 분류하는 선형 분류 알고리즘이다. 이러한 분류 알고리즘의 종류는 매우 많으나 그 중 서포트 벡터 머신이 가지고 있는 특징은 분류 경계면이 두 범주로 부터 가장 큰 여유 공간(margin)을 가진다는 것이다. 이를 그림으로 표현하면 다음과 같다.

<br/>

<br/>


## 3. SVM의 종류
서포트 벡터 머신은 본래 선형 분류기이지만 저차원의 데이터를 고차원으로 맵핑시켜 분류한 후에 다시 이를 저차원으로 사영시키면 비선형 분류기처럼 사용 할 수 있다. 이러한 서포트 벡터 머신의 종류를 python 코드를  통해 좀 더 알아보도록 하자.
<br/>
<br/>
기본적으로 사용할 python 패키지들은 다음과 같다.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
### 1. Linear SVM
가장 기본적인 선형 서포트 벡터 머신이다. 두 범주를 잘 나누어주는 선형 분류 경계면을 만드는 것을 목표로 한다.
```python
#데이터 셋 생성
from sklearn import datasets

X,y = datasets.make_blobs(centers=2,random_state=0)
df = pd.DataFrame(X,columns=['x1','x2'])
df['class'] = y

plt.scatter('x1','x2',c='class',data=df)
```


![](https://github.com/cyp-ark/SVM/blob/main/plot/linearsvmdata.png)


```python
#선형 서포트 벡터 머신 정의 및 분류 결과 플롯
from sklearn.svm import LinearSVC

linear_svm = LinearSVC(C=1,max_iter=10e4).fit(X,y)

line = np.linspace(-15,15)

plt.scatter('x1','x2',c='class',data=df)

for coef, intercept, color in zip(linear_svm.coef_,linear_svm.intercept_,["black","blue","red"]):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
    plt.fill_between(line,-(line*coef[0]+intercept+1)/coef[1],-(line*coef[0]+intercept-1)/coef[1],color=color,alpha=0.3)

plt.xlim([min(df['x1'])-1,max(df['x1'])+1])
plt.ylim([min(df['x2'])-1,max(df['x2'])+1])

plt.show()
```

![](https://github.com/cyp-ark/SVM/blob/main/plot/linearsvmresult.png)

### 2. Non-linear SVM
```python
X2, y2 = datasets.make_gaussian_quantiles(n_classes=2,random_state=4)

df2 = pd.DataFrame(X2,columns=['x1','x2'])
df2['class'] = y2

plt.scatter('x1','x2',c='class',data=df2)
```

![](https://github.com/cyp-ark/SVM/blob/main/plot/nonlinearsvm.png)


### 3. Kernel SVM

### 4. Multiclass SVM

