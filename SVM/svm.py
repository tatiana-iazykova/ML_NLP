# -*- coding: utf-8 -*-
"""SVM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1myQCuBsWaGocZMuKApTEWSFzK5-5Brpa
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

"""# Task 1

Read the documentation for scipy.optimize.minimize, paying special attention to the Jacobian argument jac. Who computes the gradient, the minimize function itself, or the developer using it?

Run the following two examples; which performs better?

"""

def f(x):
  return x**2

def df(x):
  return 2*x

"""Example 1"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# np.random.seed(seed=13)
# print(scipy.optimize.minimize(f, np.random.randint(-1000, 1000), jac=df))

"""Example 2"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# np.random.seed(seed=13)
# print(scipy.optimize.minimize(f, np.random.randint(-1000, 1000), jac=False))

"""The first one is faster

# Task 2

Write in python the loss function for support vector machines from equation (7.48) of Daumé. You can use the following hinge loss surrogate:
"""

def hinge_loss_surrogate(y_gold, y_pred):
  return max(0, 1 - y_gold * y_pred)
 
def svm_loss(params, C, D):
  w, b = params[:-1],  params[-1]
  return 0.5*np.dot(w, w) + C*sum(hinge_loss_surrogate(pair[1], np.dot(w,pair[0])+b) for pair in D)

"""# Task 3

Use `scipy.optimize.minimize` with `jac=False` to implement support vector machines.
"""

def svm(D, C=1):
  w = np.zeros(len(D[0][0]))
  b = np.array([0])
  func = lambda P: svm_loss(P, C=C, D=D)
  res = scipy.optimize.minimize(func, x0=np.concatenate([w,b]), jac=False).x
  opt_w = res[:-1]
  opt_b = res[-1]
  return opt_w, opt_b

"""# Task 4
Implement the gradient of svm_loss, and add an optional flag to svm to use it:
"""

def gradient_hinge_loss_surrogate(y_gold, y_pred):
  if hinge_loss_surrogate(y_gold, y_pred) == 0:
    return 0
  else:
    return -y_gold

def gradient_svm_loss(params, D, C=1):
  w, b = params[:-1],  params[-1]
  d_w = w + C*sum(gradient_hinge_loss_surrogate(pair[1], np.dot(w,pair[0])+b)*pair[0] for pair in D)
  d_b = C*sum(gradient_hinge_loss_surrogate(pair[1], np.dot(w,pair[0])+b) for pair in D)
  return np.hstack([d_w,d_b])

def svm(D, use_gradient=False, C=1):
  w = np.random.randint(100, size=len(D[0][0]))
  b = np.random.rand(1)
  if use_gradient:
    func = lambda P: svm_loss(P, C=C, D=D)
    func1  = lambda P: gradient_svm_loss(P, C=C, D=D)
    res = scipy.optimize.minimize(func, x0=np.concatenate([w,b]), jac=func1).x
    opt_w = res[:-1]
    opt_b = res[-1]
  else:
    func = lambda P: svm_loss(P, C=C, D=D)
    res = scipy.optimize.minimize(func, x0=np.concatenate([w,b]), jac=False).x
    opt_w = res[:-1]
    opt_b = res[-1]
  return opt_w, opt_b

"""# Task 5

Use numpy.random.normal to generate two isolated clusters of points in 2 dimensions, one x_plus and one x_minus, and graph the three hyperplanes found by training:

* an averaged perceptron
* support vector machine without gradient
* support vector machine with gradient.

"""

x_plus = np.random.normal(loc=[-1,-1], scale=0.5, size=(20,2))
x_minus = np.random.normal(loc=[1,1], scale=0.5, size=(20,2))

x_plus_y = [(x, 1) for x in x_plus]
x_minus_y = [(x, -1) for x in x_minus]

D =  x_plus_y + x_minus_y

plt.scatter(
	x_plus[:,0], x_plus[:,1],
	marker='+',
	color='blue'
)
plt.scatter(
	x_minus[:,0], x_minus[:,1],
	marker='x',
	color='red'
)

def hyperplane(w, b, color, label):
  x = [-3,3]
  y = [-(w[0]*x[0] + b)/w[1], -(w[0]*x[1] + b)/w[1]]
  print(y)
  return plt.plot(x, y, color=color, label=label)

"""an averaged perceptron"""

def AveragedPerceptronTrain(D, maxiter = 100):
  w = np.zeros(len(D[0][0]))
  b = np.zeros(1)
  u = np.zeros(len(D[0][0]))
  beta = 0
  c = 1
  for i in range(maxiter):
    for x, y in D:
      a = np.dot(x,w) + b
      if np.sign(y*a) <= 0: 
        w += y*x
        b += y
        u += y*c*x
        beta += y*c
      c += 1
  return w-(1/c)*u, b-beta*(1/c)

w_svm, b_svm = svm(D)

w_svm_g, b_svm_g = svm(D, use_gradient=True)

w_perceptron, b_perceptron = AveragedPerceptronTrain(D)

_ = plt.scatter(x_plus[:,0], x_plus[:,1],	marker='+',	color='blue')
_ = plt.scatter(x_minus[:,0], x_minus[:,1],	marker='x',	color='red')
_ = hyperplane(w_svm, b_svm, color='#077a07', label='support vector machine without gradient')
_ = hyperplane(w_svm_g, b_svm_g, color='#e58d67', label='support vector machine with gradient')
_ = hyperplane(w_perceptron, b_perceptron, color='#89dfd2', label='averaged perceptron')
_ = plt.legend(loc='lower right')
plt.savefig("svm-svm-perceptron.pdf") 
plt.show()
