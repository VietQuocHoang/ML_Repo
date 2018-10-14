import math
import numpy as np
import matplotlib as plt


def grad(x):
    return 2*x+5*np.cos(x)


def cost(x):
    return x**2 + 5*np.sin(x)


def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        print("--------- Iteration %d ----------" % (it+1))
        x_new = x[-1] - eta*grad(x[-1])
        print("x_new = %f" % x_new)
        if(abs(grad(x_new)) < 1e-3):
            break
        x.append(x_new)
    return (x, it)


print("x = -5, learning rate = 0.1")
(x1, it1) = myGD1(.1, -5)

print("-----------------------------------------------------")
print("x = 5, learning rate = 0.1")
(x2, it2) = myGD1(.1, 5)
print("Solution x1 = %f, cost = %f, obtained after %d iterations" %
      (x1[-1], cost(x1[-1]), it1))
print("Solution x2 = %f, cost = %f, obtained after %d iterations" %
      (x2[-1], cost(x2[-1]), it2))
