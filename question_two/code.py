import matplotlib.pyplot as plt
import numpy as np
import math

f = lambda a,b: pow((a-1),2) + 2* pow(b-2,2)
h1 = lambda a,b: 1 - pow(a,2) - pow(b,2)
h2 = lambda a,b: a+b


x = np.linspace(0, 20, 30)
y = np.linspace(0, 20, 30)

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
ax.view_init(40,30)
ax.set_title('f(x)');

#========================================
f = lambda a,b: pow((a-1),2) + 2* pow(b-2,2) - np.log(a+b)
x = np.linspace(1, 20, 30)
y = np.linspace(1, 20, 30)

X, Y = np.meshgrid(x, y)

Z_two = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z_two, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
ax.view_init(40,0)
ax.set_title('f(x) with barrier functions (feasible set)');

#========================================
#Find a solution to the problem using the natural logarithmic barrier function, i.e., the barrier 
#function is -log(h1(𝑥)) - log(h2(𝑥)). Use initialization vector [0.5 0.5]T and the initial penalty 
#parameter equal to 1 and reduce it by ½ in each iteration. Use a stopping threshold of 0.002;  
penalty = 1
rate = 0.001 
stopping_threshold = 0.002
cur_a = 0.5
cur_b = 0.5
precision = pow(10,-6) #This tells us when to stop the algorithm
max_iters = 10000 # maximum number of iterations
previous_step_size_a = 1 
previous_step_size_b = 1 
iters = 0 #iteration counter
da = lambda a,b: 2*a-2 #d/da
db = lambda a,b: 4*b-8 #d/db

da_log = lambda a,b: 1/(math.log(10)*(a*b)) #d/da LOG PART
db_log = lambda a,b: 1/(math.log(10)*(a*b)) #d/db LOG PART

iterations = []
value_of_objctive_function = []

while previous_step_size_a > precision or previous_step_size_b > precision and iters < max_iters:
    prev_a = cur_a #Store current a value in prev_a
    prev_b = cur_b #Store current a value in prev_a

    cur_a = cur_a - rate * da(prev_a,prev_b) - penalty*da_log(prev_a,prev_b) #Grad descent a
    cur_b = cur_b - rate * db(prev_a,prev_b) - penalty*db_log(prev_a,prev_b) #Grad descent b

    previous_step_size_a = abs(cur_a - prev_a) #Change in a
    previous_step_size_b = abs(cur_b - prev_b) #Change in b

    penalty*=0.5
    
    iters = iters+1 #iteration count
    iterations.append(iters)

    value_of_objctive_function.append(f(cur_a,cur_b))
    
print("The local minimum occurs at", cur_a,cur_b, " After: ",iters, " iterations" )

#========================================
plt.plot(iterations, value_of_objctive_function)
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')