import matplotlib.pyplot as plt
import numpy as np
#functions 
#let a = x1
#let b = x2
#f(a,b) = (a−1)^2 + 2(b-2)^2
#h(a,b) = [1 - a^2 - b^2, a+b]^T

f = lambda a,b: pow((a-1),2) + 2* pow(b-2,2)
h1 = lambda a,b: 1 - pow(a,2) - pow(b,2)
h2 = lambda a,b: a+b


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)


Z = f(X, Y)
H1 = h1(X, Y)
H2 = h2(X, Y)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.contour3D(X, Y, H1, 50, cmap='binary')
ax.contour3D(X, Y, H2, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
ax.view_init(20, 30)
ax.set_title('f(x)');