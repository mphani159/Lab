import numpy as np
import matplotlib.pyplot as plt

cur_x = 0.50 # The algorithm starts at x=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 20000 # maximum number of iterations
iters = 0 #iteration counter
f = lambda x: x**8 + x**7 - x**6 + x**5 + x**4 - x**3 + x**2 - x + 2
df = lambda x: 8*x**7 + 7*x**6 - 6*x**5 + 5*x**4 + 4*x**3 - 3*x**2 + 2*x + 1 #Gradient of our function

#f = lambda x: x**8+x**7+x**6-x**5+x**4-x**3+x**2-x+1              #Function
#df = lambda x: 8*x**7-7*x**6+6*x**5-5*x**4+4*x**3-3*x**2+2*x-1

x = np.arange(-1,1,0.0001)
plt.plot(x,f(x))
plt.show()

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x  # Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x)  # Grad descent
    previous_step_size = abs(cur_x - prev_x)  # Change in x
    iters = iters + 1  # iteration count
    #print("Iteration", iters, "\nX value is", cur_x)  # Print iterations

plt.plot(x,f(x),'b')
plt.plot(cur_x,f(cur_x),'r+')
plt.show()
print("The local minimum occurs at", cur_x)