# 서포트 벡터 머신(SVM, Support Vector Machine)
## 1. Introduction
이번 시간에는 지도학습 분류 알고리즘인 서포트 벡터 머신(SVM)과 그 종류에 대해 알아보고 사용자가 조절할 수 있는 하이퍼 파라미터(C, gamma)의 변화에 따른 모델의 변화를 알아보고자 한다.

## 2. 서포트 벡터 머신이란?
서포트 벡터 머신은 두가지 범주에 대해 선형의 경계면을 만들어 두 범주를 분류하는 선형 분류 알고리즘이다. 이러한 분류 알고리즘의 종류는 매우 많으나 그 중 서포트 벡터 머신이 가지고 있는 특징은 분류 경계면이 두 범주로 부터 가장 큰 여유 공간(margin)을 가진다는 것이다. 이를 그림으로 표현하면 다음과 같다.
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/margin.png">
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
### 1) Linear SVM
가장 기본적인 선형 서포트 벡터 머신이다. 두 범주를 잘 나누어주면서 마진을 최대로 하는 선형 분류 경계면을 만드는 것을 목표로 한다.
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
#선형 서포트 벡터 머신 정의 및 분류 결과 
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

<br/>
<p align="left"> 여기서 C는 마진을 최대로 하는 분류 경계면을 만드는 과정에서 분류 경계면을 넘을 경우 발생하는 패널티 항(penalty term)이다. 즉 C값이 작을 수록 분류 경계면을 넘는 점들에 대해 패널티를 적게 부과하기 때문에 더 많은 점들이 분류 경계면을 넘을 것이며 그만큼 마진의 폭 또한 커질 것이다. 반대로 C값이 클수록 분류 경계면을 넘는 점들에 대해 패널티를 크게 부과하기 때문에 분류경계면을 되도록 넘지 않도록 하면서 작은 폭의 마진을 가지는 분류 경계면이 만들어질 것이다.

### 2) Non-linear SVM
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
현재의 데이터에서 $x, y$ 이외에 새로운 좌표축 $z$를 정의하고 $z=x^2+y^2$라 하자. 그리고 이러한 3차원 좌표를 공간에 시각화 하면 다음과 같을 것이다.


    
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

<br/>위의 시각화 결과를 살펴보자면 이전의 2차원 공간에 있을 때와 비교해 xy평면에 평행한 분류 경계면을 만들면 두 분류를 구분할 수 있어 보인다.
<br/>3차원 공간에서 서포트 벡터 머신을 적용하여 분류 경계면을 만든 결과는 다음과 같다.


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

    
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/nonlinearsvmresults.png">

<br/>앞서 예상한대로 xy평면에 거의 평행한 분류 경계면이 만들어져 두 뷴류를 구별하고 있는 것을 알 수 있다. 하지만 본래의 데이터는 2차원 데이터 였기 때문에 서포트 벡터 머신을 통해 분류한 결과를 다시 저차원으로 복원시켜야 한다.

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

서포트 벡터 머신은 본래 선형분류기이지만 선형이 아닌 기존의 저차원 데이터를 고차원으로 맵핑시켜 고차원 공간에서 서포트 벡터 머신을 통해 선형 분류 경계면을 만든 후 다시 원래의 저차원 공간으로 복원시키면 선형 분류기 이지만 비선형 분류 경계면 처럼 동작하게 할 수 있다.
    
### 3) Kernel SVM
이전 예제의 경우 데이터 값들이 원점을 중심으로 두 분류가 분포 되어있었고, 비교적 간단한 고차원으로의 맵핑을 통해 서포트 벡터 머신을 이용한 분류 경계면을 찾을 수 있었다. 그렇다면 우리는 비선형적인 분포를 가진 데이터에 대해 서포트 벡터 머신을 적용해 분류 경계면을 찾으려고 하면 그때마다 항상 어떠한 임의의 고차원에 대해서 데이터를 맵핑한 후 서포트 벡터 머신을 적용해야할까? 정답은 "아니오"이다.
<br/><br/>
다음의 예시 데이터를 보면 방금 전 비선형 서포트 벡터 머신과 같이 현재 차원에서는 선형 분류기를 이용해 두 범주를 분류 할 수 없으니, 고차원으로 맵핑시킨 후 서포트 벡터 머신을 사용해야 한다는 것을 알 수 있다. 
 

```python
X3, y3 = datasets.make_moons(noise=0.1,random_state=0)
df3 = pd.DataFrame(X3,columns=['x1','x2'])
df3['class'] = y3

plt.xlim([min(df3['x1'])-1,max(df3['x1'])+1])
plt.ylim([min(df3['x2'])-1,max(df3['x2'])+1])

plt.scatter('x1','x2',c='class',data=df3)

plt.show()
```
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/kernelsvmdata.png">

앞선 예제와 같이 저차원의 데이터를 고차원으로 맵핑 시켜주는 함수를 커널함수라고 한다. 싸이킷런의 서포트 벡터 머신 패키지에서는 총 4가지의 커널 함수를 제공하고 있다.
- linear : $< x,x'>$
- polynomial : $(\gamma < x,x' > + r)^{d}$
- rbf(gaussian) : $exp(-\gamma||x-x'||^2)$
- sigmoid : $tanh(\gamma < x,x' > +r)$

그 중 일반적으로 많이 사용하는 rbf 커널을 이용해 서포트 벡터 머신을 적용해보도록 하자.
<br/>
```python
from sklearn.svm import SVC

svm = SVC(C=1,gamma=0.5,kernel="rbf")
svm.fit(X3,y3)
```
```python
from sklearn.inspection import DecisionBoundaryDisplay

ax = plt.gca()

plt.scatter('x1','x2',c='class',data=df3)


DecisionBoundaryDisplay.from_estimator(
    svm,
    X3,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax
)

ax.scatter(
    svm.support_vectors_[:, 0],
    svm.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

plt.xlim([min(df3['x1'])-1,max(df3['x1'])+1])
plt.ylim([min(df3['x2'])-1,max(df3['x2'])+1])

plt.show()
```
<br/>
<p align="center"> < C값에 따른 rbf kernel SVM의 분류 경계면 변화 >
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/kernelsvmrbfc.png">
<br/>
<p align="left">앞선 두 예시와 마찬가지로 C값을 조정해 rbf 커널을 적용한 서포트 벡터 머신의 경계면 변화를 살펴보자. 경계면을 넘는것에 대한 패널티가 클 수록 마진의 폭이 좁아지면서 좀 더 원래의 데이터 모양을 따라가는 경향이 있다.


<br/>
<br/>
<p align="center">< gamma값에 따른 rbf kernel SVM의 분류 경계면 변화 >
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/kernelsvmrbfgamma.png">
<br/>
<p align="left"> rbf 커널에서의 하이퍼 파라미터인 gamma를 조정할 경우 서포트 벡터 머신의 경계면의 변화를 살펴보자. gamma값이 작으면 경계면의 구조가 매우 단순해지고 평평한 모양을 띄는 것을 알 수 있다. 반대로 gamma 값이 커지면 경계면이 복잡한 형태를 띄면서 데이터 점들에 거의 fit하게 경계면이 만들어지는 것을 확인할 수 있다.

### 4) Multiclass SVM
지금까지는 
```python
X4, y4 = datasets.make_blobs(random_state=17)

df4 = pd.DataFrame(X4, columns=['x1', 'x2'])
df4['class'] = y4

plt.scatter('x1', 'x2', c='class', data=df4)
```
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/multiclasssvmdata.png">


```python
linear_svm4 = LinearSVC().fit(X4,y4)

line = np.linspace(-15,15)

fig = plt.figure()

plt.scatter('x1', 'x2', c='class',data=df4)

for coef, intercept in zip(linear_svm4.coef_,linear_svm4.intercept_):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1])
    plt.fill_between(line,-(line*coef[0]+intercept+1)/coef[1],-(line*coef[0]+intercept-1)/coef[1],alpha=0.3)

plt.xlim([min(df4['x1'])-1,max(df4['x1'])+1])
plt.ylim([min(df4['x2'])-1,max(df4['x2'])+1])

plt.show()
```
<p align="center"> <img src="https://github.com/cyp-ark/SVM/blob/main/plot/multiclasssvm.png">

## 3. 결론




## 4. Reference
1. 데이터 사이언스 스쿨 - 5.2 분류용 가상 데이터 생성 [[Link]](https://datascienceschool.net/03%20machine%20learning/09.02%20%EB%B6%84%EB%A5%98%EC%9A%A9%20%EA%B0%80%EC%83%81%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%83%9D%EC%84%B1.html)
2. 초롱스쿨 - [Python]분류분석 - SVM(Support Vector Machine) [[Link]](https://studychfhd.tistory.com/224)
3. scikit learn - 1.4 Support Vector Machines [[Link]](https://scikit-learn.org/stable/modules/svm.html)
