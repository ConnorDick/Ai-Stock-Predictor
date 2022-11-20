import matplotlib.pyplot as plt

#initial values
cur_a = 1 
cur_b = 1 
cur_c = 1 

rate = 0.001 # Learning rate
precision = pow(10,-6) #This tells us when to stop the algorithm
previous_step_size_a = 1 
previous_step_size_b = 1 
previous_step_size_c = 1 
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
da = lambda a,b,c: (2*a) + 10 + (4*pow(b,2)*a) + (8*pow(c,2)*a) #d/da
db = lambda a,b: (2*b) + 16 + (4*pow(a,2)*b) #d/db
dc = lambda a,c: (2*c) + 14 + (8*pow(a,2)*c) #d/dc

while previous_step_size_a > precision or previous_step_size_b > precision or previous_step_size_c > precision  and iters < max_iters:
    prev_a = cur_a #Store current a value in prev_a
    prev_b = cur_b #Store current a value in prev_a
    prev_c = cur_c #Store current a value in prev_a

    cur_a = cur_a - rate * da(prev_a,prev_b, prev_c) #Grad descent a
    cur_b = cur_b - rate * db(prev_a,prev_b) #Grad descent b
    cur_c = cur_c - rate * dc(prev_a,prev_c) #Grad descent c

    previous_step_size_a = abs(cur_a - prev_a) #Change in a
    previous_step_size_b = abs(cur_b - prev_b) #Change in b
    previous_step_size_c = abs(cur_c - prev_c) #Change in c
    
    iters = iters+1 #iteration count
    
print("The local minimum occurs at", cur_a,cur_b,cur_c, " After: ",iters, " iterations" )

#Create the plot
plt.plot(iterations, value_of_objctive_function)
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
