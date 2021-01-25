# [DAY6] Numpy / 벡터 / 행렬

### 1. Numpy

- 'Numerical Python'의 줄임말
- Python의 **고성능 과학 계산용 패키지**
- Array 연산(*ex. Matrix, Vector*)의 사실상 표준 패키지

#### 특징

- 일반 List에 비해 빠르고, 효율적인 메모리
- 반복문 없이 데이터 배열 처리 지원
- 선형대수와 관련된 다양한 기능을 제공함
- C, C++, 포트란 등의 언어와 통합 가능

#### 설치

```bash
activate ml
conta install numpy
```

#### 호출

```python
import numpy as np
```

- np라는 별칭(alias) 이용해서 호출함

#### array creation

```python
test_array = np.array([1,4,5,8], float)
print(test_array) #[1. 4. 5. 8.]
type(test_array[3]) #numpy.float64
```

- np.array 함수로 활용 배열 생성함 -> **ndarray**

- numpy는 **하나의 데이터 type** 만 배열에 넣을 수 있음

- C의 Array를 사용하여 배열을 생성함 

- List와의 차이점 -> **dynamic typing not supported**

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오전 11.00.46.png" alt="스크린샷 2021-01-25 오전 11.00.46" style="zoom: 33%;" />

  * Python List는 각각의 주소값을 할당하는 방식으로 값이 List에 저장되는데, ndarray는 값 자체가 바로 저장됨

  * List는 주소값 할당 방식이기 때문에 다양한 type의 데이터가 저장될 수 있는 반면, ndarray는 값 자체가 저장되기 때문에 연산 속도가 빠르다는 장점

    <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오전 11.32.40.png" alt="스크린샷 2021-01-25 오전 11.32.40" style="zoom:33%;" />

  * 더불어 메모리 크기가 일정하기 때문에 메모리를 잡기에도 좋음

  ```python
  a = [1,2,3,4,5]
  b = [5,4,3,2,1]
  a[0] is b[-1] #True
  ```

  * python list의 메모리는 static하기 때문에 1이라는 값으로 값이 똑같아지면 True 출력

  ```python
  a = np.array(a)
  b = np.array(b)
  a[0] is b[-1] #False
  ```

  * ndarray 에서는 같은 값이라고 하더라도 메모리가 다르기 때문에 False가 출력됨

- shape & type

  ```python
  test_array = np.array([1, 4, 5, "8"], float) 
  print(test_array) #[1. 4. 5. 8.]
  print(type(test_array[3])) #<class 'numpy.float64'>
  print(test_array.dtype) #float64
  print(test_array.shape) #(4,)
  
  ```

  * shape : numpy array의 dimension 구성 반환
    * array의 RANK에 따라 불리는 이름이 있음

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오전 11.39.21.png" alt="스크린샷 2021-01-25 오전 11.39.21" style="zoom: 50%;" />

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오전 11.47.56.png" alt="스크린샷 2021-01-25 오전 11.47.56" style="zoom:33%;" />

  * dtype : numpy array의 데이터 type 반환
    * ndarray의 single element가 가지는 data type

  * (추가) nbytes : ndarray object 메모리 크기를 반환함
    * 1bytes = 4bits (*ex. float32 는 32bits = 4bytes*) 

  ```python
  np.array([[1,2,3],[4.5,"5","6"]], dtype=np.float32).nbytes #24
  np.array([[1,2,3],[4.5,"5","6"]], dtype=np.int8).nbytes #6
  np.array([[1,2,3],[4.5,"5","6"]], dtype=np.float64).nbytes #48
  ```

#### Handling shape

* reshape

  * element 개수를 동일하게 하며, array의 shape 크기를 변경.

    ```python
    test_matrix = [[1,2,3,4],[1,2,5,8]]
    np.array(test_matrix).shape #(2,4)
    np.array(test_matrix).reshape(8,) #array([1,2,3,4,1,2,5,8])
    np.array(test_matrix).reshape(8,).shape #(8,)
    #-1 : 전체 size 기반으로 하여 row 개수 선정
    np.array(test_matrix).reshape(-1,2).shape #(4,2)
    np.array(test_matrix).reshape(-1,4).shape #(2,4)
    ```

* flatten

  * 다차원 array를 1차원 array로 변환

    ```python
    test_matrix = [[[1,2,3,4],[1,2,5,8]],[[1,2,3,4],[1,2,5,8]]]
    np.array(test_matrix).flatten() #array([1,2,3,4,1,2,5,8,1,2,3,4,1,2,5,8])
    ```

#### Indexing & Slicing

* list와 달리 이차원 배열에서 [0,0] 표기법을 제공함

  ```python
  a = np.array([[1, 2, 3], [4.5, 5, 6]], int)
  print(a[0,0]) # 2D array 표기법 1
  print(a[0][0]) # 2D array 표기법 2
  ```

* list와 달리 행과 열 부분을 나누어 slicing 가능 -> matrix 부분집합 추출시 유용함

  ```python
  a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int) 
  a[:,2:] # 전체 Row의 2열 이상
  a[1,1:3] # 1 Row의 1열 ~ 2열
  a[1:3] # 1 Row ~ 2Row 전체
  ```

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 12.12.38.png" alt="스크린샷 2021-01-25 오후 12.12.38" style="zoom:33%;" />

  * 값은 같지만 의미가 다름.

  ```python
  a[1] #array([6,7,8,9,10])
  a[1:3] #array([[6,7,8,9,10]])
  ```

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 12.20.32.png" alt="스크린샷 2021-01-25 오후 12.20.32" style="zoom:33%;" />

#### creation function

* arange
  * array의 범위를 지정하여, 값의 list를 생성하는 명령어

  * *'어레인지'*라고 읽기보다 *'에이-레인지'* 라고 읽는 것 같음

    ```python
    np.arange(10) #array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
    
    #floating point도 표시 가능
    np.arange(0,4,0.5) #array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5])
    
    np.arange(30).reshape(5,6)
    #array([[ 0,  1,  2,  3,  4,  5],
    #       [ 6,  7,  8,  9, 10, 11],
    #       [12, 13, 14, 15, 16, 17],
    #       [18, 19, 20, 21, 22, 23],
    #       [24, 25, 26, 27, 28, 29]])
    ```

* ones, zeros, empty

  * zeros : 0으로 가득찬 ndarray 생성

  ```python
  np.zeros(shape=(10,), dtype=np.int8)
  #array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)
  np.zeros((2,5))
  #array([[0., 0., 0., 0., 0.],
  #       [0., 0., 0., 0., 0.]])
  ```

  * ones : 1로 가득찬 ndarray 생성

  ```python
  np.ones(shape=(10,), dtype=np.int8)
  #array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int8)
  np.zeros((2,5))
  #array([[1., 1., 1., 1., 1.],
  #       [1., 1., 1., 1., 1.]])
  ```

  * empty : shape만 주어진 빈 ndarray 생성 (memory initialization X)

  ```python
  np.empty(shape=(10,), dtype=np.int8)
  np.empty((3,5))
  ```

  * something_like : 기존 ndarray의 shape 크기만큼의 1, 0, empty를 반환

  ```python
  test_matrix = np.arange(30).reshape(5,6)
  np.ones_like(test_matrix)
  #array([[1, 1, 1, 1, 1, 1],
  #       [1, 1, 1, 1, 1, 1],
  #       [1, 1, 1, 1, 1, 1],
  #       [1, 1, 1, 1, 1, 1],
  #       [1, 1, 1, 1, 1, 1]])
  ```

* 독특한 array

  * identity : 단위 행렬 생성

  ```python
  np.identity(n=3, dtype=np.int8)
  #array([[1, 0, 0],
  #       [0, 1, 0],
  #       [0, 0, 1]], dtype=int8)
  ```

  * eye : 대각선이 1인 행렬, k값으로 시작 index 변경 가능

  ```python
  np.eye(3)
  #array([[1., 0., 0.],
  #       [0., 1., 0.],
  #       [0., 0., 1.]])
  np.eye(3,5,k=2)
  #array([[0., 0., 1., 0., 0.],
  #       [0., 0., 0., 1., 0.],
  #       [0., 0., 0., 0., 1.]])
  ```

  * diag : 대각 행렬 값 추출, k값으로 시작 index 변경 가능

  ```python
  matrix = np.arange(9).reshape(3,3)
  np.diag(matrix) #array([0,4,8])
  ```

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 12.41.33.png" alt="스크린샷 2021-01-25 오후 12.41.33" style="zoom:25%;" />

  * random sampling : 데이터 분포에 따른 sampling

  ```python
  np.random.uniform(0,1,10).reshape(2,5) #균등분포
  np.random.normal(0,1,10).reshape(2,5) #정규분포
  ```

#### Operation functions

* sum

  * ndarray element의 합을 구함

  ```python
  test_array = np.arange(1,11) #1부터 10까지의 값을 갖는 array 생성
  test_array.sum(dtype=np.float) #55.0
  ```

* axis

  * 모든 operation function을 실행할 때 기준이 되는 dimension 축

  ```python
  test_array = np.arange(1,13).reshape(3,4)
  test_array.sum(axis=1) #array([10, 26, 42])
  test_array.sum(axis=0) #array([15, 18, 21, 24])
  ```

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 12.57.41.png" alt="스크린샷 2021-01-25 오후 12.57.41" style="zoom:50%;" />

* mean & std

  * 평균, 표준편차 반환 (axis 사용 가능)

  ```python
  test_array = np.arange(1,13).reshape(3,4)
  test_array.mean() #6.5
  test_array.std() #3.452052529534663
  ```

* vstack, hstack, concatenate

  * ndarray를 붙이는 합수

  ```python
  a = np.array([1,2,3])
  b = np.array([2,3,4])
  np.vstack((a,b))
  #array([[1, 2, 3],
  #       [2, 3, 4]])
  ```

  ```python
  a = np.array([[1],[2],[3]])
  b = np.array([[2],[3],[4]])
  np.hstack((a,b))
  #array([[1, 2],
  #       [2, 3],
  #       [3, 4]])
  ```

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 1.10.56.png" alt="스크린샷 2021-01-25 오후 1.10.56" style="zoom:33%;" />

  ```python
  a = np.array([[1,2,3]])
  b = np.array([[2,3,4]])
  np.concatenate((a,b), axis=0)
  ```

  * dimension 맞춰주기

    * concatenate로 array를 합치다보면 아래와 같은 상황이 발생할 수도 있다.

    ```python
    a = np.array([[1,2],[3,4]])
    b = np.array([5,6])
    np.concatenate((a,b.T),axis = 1)
    ```

    * 이 경우, ValueError: all the input arrays must have same number of dimensions 라는 에러가 뜨는데, a는 2D, b는 1D array이기에 나타나는 오류이다.
    * 이런 에러가 뜰 때에는 당황하지 말고, b의 dimension을 아래와 같은 방법으로 늘리면 된다.
    * newaxis에 대한 자세한 설명은 [링크](https://azanewta.tistory.com/3)를 참고하면 된다

    ```python
    a = np.array([[1,2],[3,4]])
    b = np.array([5,6])
    b = b[np.newaxis,:] #array([[5, 6]])
    np.concatenate((a,b.T),axis = 1)
    ```

#### array operations

* numpy는 array간의 기본적인 사칙 연산을 제공함

```python
test_a = np.array([[1,2,3],[4,5,6]],float)
test_a + test_a
test_a - test_a
```

* element-wise operations

  * array 간 shape이 같을 때 일어나는 연산

  ```python
  matrix_a = np.arange(1,13).reshape(3,4)
  matrix_a * matrix_a
  #array([[  1,   4,   9,  16],
  #       [ 25,  36,  49,  64],
  #       [ 81, 100, 121, 144]])
  ```

  ![스크린샷 2021-01-25 오후 1.35.10](/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 1.35.10.png)

* dot product

  * matrix의 행렬곱 연산

  ```python
  test_a = np.arange(1,7).reshape(2,3)
  test_b = np.arange(7.13).reshape(3.2)
  test_a.dot(test_b) #행렬곱
  ```

* transpose

  * 전치행렬 변환

  ```python
  test_a.transpose()
  test_a.T
  ```

* broadcasting

  * shape이 다른 배열 간 연산을 지원하는 기능
  * 더하기, 빼기, 곱셈, 나눗셈(/, //, %) 모두 지원

  ```python
  test_matrix = np.array([[1,2,3],[4,5,6]],float)
  scalar = 3
  test_matrix + scalar # mat + scalar
  #array([[4., 5., 6.],
  #       [7., 8., 9.]])
  ```

  * scalar-vector 외에도 vector-matrix 연산도 지원

  ```python
  test_matrix = np.arange(1,13).reshape(4,3)
  test_vector = np.arange(10,40,10) #[10,20,30]
  test_matrix+test_vector
  #array([[11, 22, 33],
  #       [14, 25, 36],
  #       [17, 28, 39],
  #       [20, 31, 42]])
  ```

#### performance

* timeit(Jupiter 환경에서 코드 performance 체크하는 함수)를 이용하여 성능 체크
* 일반적으로 속도는 for loop < list comprehension < numpy
* 100,000,000번의 loop가 돌 때 약 4배 이상의 성능 차이를 보임
* numpy는 C로 구현되어 있어, 성능 확보 대신 dynamic typing을 포기함
* **대용량 계산에서는 흔히 사용**됨
* concatenate와 같이 계산이 아닌 **할당에서는 연산 속도 이점이 없음**

#### comparisons

* array의 데이터 전부(and) 혹은 일부(or)가 조건에 만족하는지의 여부를 반환

```python
a = np.arange(10)
np.any(a>5),np.any(a<0) #(True, False)
np.all(a>5),np.all(a<10) #(False, True)
```

* 배열의 크기가 동일할 때, element간 비교 결과를 boolean type으로 반환

```python
test_a = np.array([1,3,0],float)
test_b = np.array([5,2,1],float)
test_a > test_b #array([False,  True, False])
test_a == test_b #array([False, False, False])
(test_a > test_b).any() #True
```

```python
a = np.array([1,3,0],float)
np.logical_and(a > 0,a < 3) #array([ True, False, False])
```

* np.where

```python
#where(condition, True, False)
#condition이 True인 곳에 3이 들어가고 False인 곳에 2가 들어감 
np.where(a > 0, 3, 2) #array([3, 3, 2])

a = np.arange(10)
np.where(a>5) #(array([6, 7, 8, 9]),)
```

```python
a = np.array([1, np.NaN, np.Inf], float)
#not a number
np.isnan(a) #array([False,  True, False])
#is finite number
np.isfinite(a) #array([ True, False, False])
```

* argmax & argmin

  * array 내 최댓값 또는 최솟값의 index를 반환함

  ```python
  a = np.array([1,2,4,5,8,78,23,3])
  a.argsort() #array([0, 1, 7, 2, 3, 4, 6, 5])
  np.argmax(a) #5
  np.argmin(a) #0
  ```

  ```python
  a = np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
  np.argmax(a, axis=1), np.argmin(a, axis=0) 
  #(array([3, 1, 1]), array([0, 0, 2, 2]))
  ```

#### Boolean & fancy index

* 특정 조건에 따른 값을 배열 형태로 추출.

  ```python
  test_array = np.array([1,4,0,2,3,8,9,7],float)
  test_array > 3
  #array([False,  True, False, False, False,  True,  True,  True])
  test_array[test_array > 3] #array([4., 8., 9., 7.])
  ```

* fancy index

  * a[b] : b 배열의 값을 index로 하여 a 값을 추출함

  ```python
  a = np.array([2,4,6,8],float)
  b = np.array([0,0,1,3,2,1],int) #반드시 int로 선언
  a[b] #array([2., 2., 4., 8., 6., 4.])
  ```

  * matrix 형태의 데이터도 가능

  ```python
  a = np.array([[1,4],[9,16]],float)
  b = np.array([0,0,1,1,0],int)
  c = np.array([0,1,1,1,1],int)
  a[b,c] #array([ 1.,  4., 16., 16.,  4.])
  ```

#### numpy data i/o

```python
a = np.loadtxt("./populations.txt", delimiter="\t")
a_int = a.astype(int)
np.savetxt("int_data.csv", a_int, fmt="%.2e", delimiter=",")
```



### 2. 벡터

* 숫자를 원소로 가지는 리스트(List) 혹은 배열(Array)

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.03.32.png" alt="스크린샷 2021-01-25 오후 6.03.32" style="zoom: 33%;" />

* 벡터는 공간에서 **한 점**을 나타내며, 원점으로부터 **상대적 위치**를 표현함. 

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.04.18.png" alt="스크린샷 2021-01-25 오후 6.04.18" style="zoom:33%;" />

* 벡터에 숫자(scalar)를 곱해주면, 방향은 유지되나 길이만 변함.
  (단, 0보다 작은 숫자를 곱했을 때에는 반대 방향으로 유지됨)

* **같은 모양의 벡터** 끼리는 덧셈, 뺄셈, 성분곱(Hadamard product)을 계산할 수 있음
* 두 벡터의 덧셈은 다른 벡터로부터의 상대적 위치 이동을 표현함.

#### 노름 (norm)

* 벡터의 노름(norm)은 **원점으로부터의 거리**를 말함



<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.13.18.png" alt="스크린샷 2021-01-25 오후 6.13.18" style="zoom:33%;" />

* L1-norm은 각 성분의 **변화량의 절댓값**을 모두 더함

* L2-norm은 피타고라스 정리를 활용하여 **유클리드 거리**를 계산함

  ```python
  def l1_norm(x):
    x_norm = np.abs(x)
    x_norm = np.sum(x_norm)
    return x_norm
  
  def l2_norm(x):
    x_norm = x*x
    x_norm = np.sum(x_norm)
    x_norm = np.sqrt(x_norm)
    return x_norm
  ```

* norm의 종류에 따라 기하학적 성질이 달라지며, 각 성질들이 필요할 때가 있으므로 둘 다 사용함

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.15.56.png" alt="스크린샷 2021-01-25 오후 6.15.56" style="zoom:33%;" />

#### 거리

* L1, L2 norm을 이용하여 두 벡터 사이의 거리를 계산할 수 있음
* 두 벡터 사이의 거리를 계산할 때는 **벡터의 뺄셈**을 이용함

#### 각도

* 제 2 코사인 법칙에 의하여 두 벡터 사이의 각도를 계산할 수 있음

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.17.47.png" alt="스크린샷 2021-01-25 오후 6.17.47" style="zoom:33%;" />

* cos 식에서의 분자는 **2<x,y>의 형태로 내적(inner product)을 활용**하면 쉽게 계산 가능함

```python
def angle(x,y):
  v = np.inner(x, y) / (l2_norm(x)*l2_norm(y))
  theta = np.arccos(v)
  return theta
```



#### 내적

* 내적은 **정사영(orthogonal projection)된 벡터의 길이**와 관련 있음
* 내적은 정사영의 길이를 **벡터 y의 길이만큼 조정**한 값

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.20.51.png" alt="스크린샷 2021-01-25 오후 6.20.51" style="zoom:33%;" />

### 3. 행렬

* 행렬(matrix)은 벡터를 원소로 가지는 **2차원 배열**
* **행(row)**과 **열(column)**이라는 인덱스(index)를 가짐
* 행열의 특정 행/열을 고정하면 행/열벡터라고 부름

* **전치행렬(transpose matrix)** : 행과 열의 인덱스가 바뀐 행렬
* 벡터가 공간에서의 한 점을 의미한다면, 행렬은 **여러 점들**을 나타냄
  * 행렬 x\[i]\[j]는 i번째 데이터의 j번째 변수 값을 말함

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.45.29.png" alt="스크린샷 2021-01-25 오후 6.45.29" style="zoom:33%;" />

* 행렬끼리 **같은 모양**을 가지면 덧셈, 뺄셈을 계산할 수 있음
  * 성분곱, 스칼라곱은 벡터와 차이가 없음

#### 곱셈 (matrix multiplication)

* **i번째 행벡터와  j번째 열터 사이의 내적**을 성분으로 가지는 행렬

* 행렬곱 X x Y을 위해서는 **행렬  X의 열과 행렬  Y의 행**이 같아야 함

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 6.48.38.png" alt="스크린샷 2021-01-25 오후 6.48.38" style="zoom:33%;" />

  ```python
  X = np.array([1,2,3])
  Y = np.array([[1],[2],[3]])
  X @ Y #행렬곱 연산
  ```

* 행렬곱을 통해 벡터를 **다른 차원의 공간**으로 보낼 수 있음
* 행렬곱을 통해 **패턴을 추출**할 수 있고 **데이터를 압축**할 수도 있음

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 7.23.19.png" alt="스크린샷 2021-01-25 오후 7.23.19" style="zoom:33%;" />

#### 내적

```python
X = np.array([[1,-2,3],[7,5,0],[-2,-1,2]])
Y = np.array([[0,1,-1],[1,-1,0]])
np.inner(X,Y)
#array([[-5,  3],
#       [ 5,  2],
#       [-3, -1]])
```

* numpy의 np.inner는 i번째 행벡터와 j번째 행벡터 사이의 내적을 성분으로 가지는 행렬 계산함
* 수학에서 말하는 내적(XY^T)과는 다름

#### 역행렬 (inverse matrix)

* 어떤 행렬 **A**의 연산을 거꾸로 되돌리는 행렬

* **행과 열 숫자가 같고 행렬식(determinant)이 0이 아닌 경우**에만 계산 가능

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 7.26.46.png" alt="스크린샷 2021-01-25 오후 7.26.46" style="zoom:33%;" />

```python
X = np.array([[1,-2,3],[7,5,0],[-2,-1,2]])
np.linalg.inv(X) #역행렬 계산
```

* 만약 역행렬을 계산할 수 없다면 **유사역행렬(pseudo-inverse)** 또는 **무어-펜로즈(Moore-Penrose) 역행렬** **A+**을 이용한다.

  <img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 7.29.06.png" alt="스크린샷 2021-01-25 오후 7.29.06" style="zoom:33%;" />

  ```python
  Y = np.array([[0,1],[1,-1],[-2,1]])
  np.linalg.pinv(Y) #유사 역행렬 계산
  ```

#### 응용1 : 연립방정식 풀기

![스크린샷 2021-01-25 오후 7.34.59](/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 7.34.59.png)

#### 응용2 : 선형회귀분석

* 선형회귀분석은 **X와 y가 주어진 상황에서 계수 beta를 찾아야 함**
* 하지만 연립방정식과 달리 행이 더 크므로 방정식을 푸는 것은 불가능

<img src="/Users/jisukim/Desktop/day6/스크린샷 2021-01-25 오후 7.40.36.png" alt="스크린샷 2021-01-25 오후 7.40.36" style="zoom:33%;" />

```python
# scikit Learn을 활용한 회귀분석
from sklearn.linear_medel import LinearRegression
model = LinearRegression()
model.fit(X,y)
y_test = model.predict(x_test)

#Moore-Penrose 역행렬
X_ = np.array([np.append(x,[1]) for x in X]) #intercept항 추가
beta = np.linalg.pinv(X) @ y
y_test = np.append(x_test) @ beta
```



