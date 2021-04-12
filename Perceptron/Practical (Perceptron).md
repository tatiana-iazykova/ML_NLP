#### Task 1
*Implement your own Scalar and Vector classes, without using any other modules:*

```python

from typing import Union, List
from math import sqrt
class Scalar:
  pass
class Vector:
  pass

class Scalar:
  def __init__(self: Scalar, val: float):
    self.val = float(val)
  def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
    # hint: use isinstance to decide what `other` is
    # raise an error if `other` isn't Scalar or Vector!
    if isinstance(other, Vector):
      res = []
      for i in range(len(other.entries)):
        res.append(self.val * other.entries[i])
      return Vector(*res)
    elif isinstance(other, Scalar):
        return Scalar(self.val * other.val)
    else:
      print("Not happening")
  def __add__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val + other.val)
  def __sub__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val - other.val)
  def __truediv__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val / other.val)
  def __rtruediv__(self: Scalar, other: Vector) -> Vector:
    res = []
    for i in range(len(other)):
      x = other.entries[i]/self.val
      res.append(x)
    return Vector(*res)
  def __repr__(self: Scalar) -> str:
    return "Scalar(%r)" % self.val
  def sign(self: Scalar) -> int:
    if self.val == 0:
      return 0
    elif self.val < 0:
      return -1
    else:
      return 1
  def __float__(self: Scalar) -> float:
    return self.val

class Vector:
  def __init__(self: Vector, *entries: List[float]):
    self.entries = entries
  def zero(size: int) -> Vector:
    return Vector(*[0 for i in range(size)])
  def __add__(self: Vector, other: Vector) -> Vector:
    if len(self.entries) == len(other):
      res = []
      for i in range(len(self.entries)):
        res.append(self.entries[i] + other.entries[i])
    return Vector(*res)
  def __sub__(self: Vector, other: Vector) -> Vector:
    if len(self.entries) == len(other):
      res = []
      for i in range(len(self.entries)):
        res.append(self.entries[i] - other.entries[i])
    return Vector(*res)
  def __mul__(self: Vector, other: Vector) -> Scalar:
    res_v = 0
    if len(self.entries) == len(other.entries):
      for i in range(len(self.entries)):
        res_v += self.entries[i] * other.entries[i]
    return Scalar(res_v)
  def magnitude(self: Vector) -> Scalar:
    res = 0
    for i in range(len(self.entries)):
      res += self.entries[i]**2
    return Scalar(sqrt(res))
  def unit(self: Vector) -> Vector:
    return self / self.magnitude()
  def __len__(self: Vector) -> int:
    return len(self.entries)
  def __repr__(self: Vector) -> str:
    return "Vector%s" % repr(self.entries)
  def __iter__(self: Vector):
    return iter(self.entries)
```

#### Task 2
*Implement the PerceptronTrain and PerceptronTest functions, using your Vector and Scalar classes. 
Do not permute the dataset when training; run through it linearly.*
*(Hint on how to use the classes: make w and x instances of Vector, y and b instances of Scalar. 
What should the type of D be? Where do you see the vector operation formulas?)*

Type of D should be a list as it consists of 2 elements.

Vector operation formulas would be present when we compute *a*, where we first use vector multiplication (Vector x by Vector w) and then we use Scalar addition. 
Upon calculating w we multiply vector by a scalar and upon calculating b we perform scalar addition

```python
def PerceptronTrain(D, maxiter = 100):
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(maxiter):
    for x, y in D:
      a = x*w + b
      if (y*a).sign() <= 0: 
        w += y*x
        b += y
  return w, b
```

```python
def PerceptronTest(w, b, D):
  res = []
  for x,y in D:
    a = x*w + b
    res.append(a.sign())
  return res
```

#### Task 3
*Make a 90-10 test-train split and evaluate your algorithm on the following dataset:*

```python

from random import randint

v = Vector(randint(-100, 100), randint(-100, 100)) # random vector, gold hyperplane
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs] #v dot x -> perfect 

def merge(list1, list2):       
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list      
    
from random import shuffle

def train_test(data, split, shuff = True):
  #shuffle the data
  if shuff:
    shuffle(data)

  b = int(len(data)*split)
  train = data[:b]
  test= data[b:]

  return train, test

D = merge(xs, ys)

train, test = train_test(D, 0.9, shuff = False)

w1, b1 = PerceptronTrain(train)
y_pred = PerceptronTest(w1, b1, test)

def score(y_pred, y_true):
  all = len(y_true)
  correct = 0
  for i in range(all):
    if y_pred[i] == y_true[i][1].sign():
      correct += 1
  return correct/all*100

score(y_pred, test)
```

```
82.0
```

#### Task 4
*Make a 90-10 test-train split and evaluate your algorithm on the xor dataset:*

```python
xs_xor = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys_xor = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else Scalar(-1) for x in xs_xor]

D_xor = merge(xs_xor, ys_xor)

train_xor, test_xor = train_test(D_xor, 0.9, shuff = False)

w2, b2 = PerceptronTrain(train_xor)
y_pred_xor = PerceptronTest(w2, b2, test_xor)

score(y_pred_xor, test_xor)
```

```
44.0
```

#### Task 5
*Sort the training data from task 3 so that all samples with y < 0 come first, then all samples with y = 0, then all samples with y > 0. (That is, sort by y.)*

Graph the performance (computed by PerceptronTest) on both train and test sets versus epochs for perceptrons trained on

* no permutation
* random permutation at the beginning
* random permutation at each epoch

![graph1](https://raw.githubusercontent.com/tatiana-iazykova/ML_NLP/main/Perceptron/different_strategies.png)

As you can see the performance of all algorithms is relatively similar.

#### Task 6
*Implement AveragedPerceptronTrain; using random permutation at each epoch, compare its performance with PerceptronTrain using the dataset from task 3.*

```python
def AveragedPerceptronTrain(D, maxiter = 100):
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)
  u = Vector.zero(len(D[0][0]))
  beta = Scalar(0)
  c = Scalar(1)
  for i in range(maxiter):
    shuffle(D)
    for x, y in D:
      a = x*w + b
      if (y*a).sign() <= 0: 
        w += y*x
        b += y
        u += y*c*x
        beta += y*c
      c += Scalar(1)
  return w-(Scalar(1)/c)*u, b-beta*(Scalar(1)/c)
```
![graph2](https://raw.githubusercontent.com/tatiana-iazykova/ML_NLP/main/Perceptron/averaged_perc.png)


In comparison to the previous algorithm, this one shows drastically more stable performance
