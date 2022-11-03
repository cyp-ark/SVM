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


<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/linearsvmdata.png">


```python
#선형 서포트 벡터 머신 정의 및 분류 결과 플롯
from sklearn.svm import LinearSVC

linear_svm = LinearSVC(C=1,max_iter=10e4).fit(X,y)

line = np.linspace(-15,15)

plt.scatter('x1','x2',c='class',data=df)

coef = linear_svm.coef_
intercept = linear_svm.intercept_
plt.plot(line,-(line*coef[:,0]+intercept)/coef[:,1],c=color)
plt.fill_between(line,-(line*coef[:,0]+intercept+1)/coef[:,1],-(line*coef[:,0]+intercept-1)/coef[:,1],color=color,alpha=0.3)

plt.xlim([min(df['x1'])-1,max(df['x1'])+1])
plt.ylim([min(df['x2'])-1,max(df['x2'])+1])

plt.show()
```

<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/linearsvmresult.png">

### 2. Non-linear SVM
그렇다면 다음 예제를 한번 보자.


```python
X2, y2 = datasets.make_gaussian_quantiles(n_classes=2,random_state=4)

df2 = pd.DataFrame(X2,columns=['x1','x2'])
df2['class'] = y2

plt.scatter('x1','x2',c='class',data=df2)
```

<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/nonlinearsvm.png">
    
이 데이터 셋의 경우 원점을 기준으로 원형으로 두 분류의 값이 분포하고 있는 것으로 보인다. 이에 대해 기존의 선형 서포트 벡터 머신을 이용하여 두 범주를 잘 구별할 수 있는 선형 분류기를 만들어낼 수 있을까? 이를 한번 적용에 본 결과는 다음과 같다.


```python
#일반적인 Linear SVM으로 분류해보기
linear_svm2 = LinearSVC(C=1,max_iter=10e4).fit(X2,y2)

plt.scatter('x1','x2',c='class',data=df2)

coef,intercept = linear_svm2.coef_, linear_svm2.intercept_
plt.plot(line,-(line*coef[:,0]+intercept)/coef[:,1],c=color)
plt.fill_between(line,-(line*coef[:,0]+intercept+1)/coef[:,1],-(line*coef[:,0]+intercept-1)/coef[:,1],color=color,alpha=0.3)

plt.xlim([min(df2['x1'])-1,max(df2['x1'])+1])
plt.ylim([min(df2['x2'])-1,max(df2['x2'])+1])

plt.show()
```

<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/nonlinear.png">


<br/>두 분류의 데이터가 원형으로 분포하고 있기 때문에 서포트 벡터 머신이 명확한 선형 경계면을 만들어내지 못한 것을 알 수 있다. 그렇다면 이러한 데이터의 분포에 대해서는 어떻게 서포트 벡터 머신을 적용할 수 있을까?<br/><br/>
현재의 데이터에서 x,y 이외에 새로운 좌표축인 z를 정의하고 ${z=x^2+y^2$라 하자.


    
```python
X2_new = np.hstack((X2,(X2[:,0]**2 + X2[:,1]**2).reshape(-1,1)))

df2 = pd.DataFrame(X2_new,columns=['x1','x2','x3'])
df2['class'] = y2

    
fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111,projection='3d')
ax.scatter(df2['x1'],df2['x2'],df2['x3'],c='class',data=df2)


ax.view_init(0,45)

plt.show()
```
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/nonlinearsvm3d.png">

```python
linear_svm3 = LinearSVC().fit(X2_new,y2)
#%%
coef, intercept = linear_svm3.coef_.ravel(), linear_svm3.intercept_
#%%
xx = np.linspace(X2_new[:,0].min()-1,X2_new[:,0].max()+1)
yy = np.linspace(X2_new[:,1].min()-1,X2_new[:,1].max()+1)
#%%
XX, YY = np.meshgrid(xx,yy)
zz = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
zz1 = (coef[0] * XX + coef[1] * YY + intercept +1) / -coef[2]
zz2 = (coef[0] * XX + coef[1] * YY + intercept -1) / -coef[2]
#%%
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(XX, YY, zz, rstride=8, cstride=8, alpha=0.5, color='lightgreen')


ax.scatter(df2['x1'],df2['x2'],df2['x3'],c='class',data=df2)
ax.view_init(5,70)

plt.show

```

    
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/nonlinearsvm3dresult.png">

```python
ZZ = XX ** 2 + YY ** 2

# np.ravel : 다차원 배열을 1차원으로 데이터 변환
dec = linear_svm3.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])

# plt.contourf : 등고선을 그려준다
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],alpha=0.5)
plt.scatter('x1','x2',c='class',data=df2)

plt.show()
```
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/nonlinear3dto2d.png">

    
### 3. Kernel SVM

### 4. Multiclass SVM

