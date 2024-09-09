#perceptron_mine algoritmasını burda test ediyorum
import numpy as np
import matplotlib.pyplot as plt
from perceptron_mine import perceptron

np.random.seed(0)

def generate_random_point(N):
    return np.random.uniform(-1, 1, (N,2))

def generate_target_function():
    point1 = generate_random_point(1)[0]
    point2 = generate_random_point(1)[0]
    
    def classify_point(point):
        x1, y1 = point1
        x2, y2 = point2
        x, y = point
        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross_product > 0:
            return 1
        else:
            return 0
    
    return [point1,point2 ,classify_point]

f = generate_target_function()
cls = f[2]
X = np.random.uniform(-1, 1, (1000,2))
y = np.array([cls(x) for x in X])

p=perceptron(0.001,10000)
p.learn(X,y)
coordinates=p.weight
print(coordinates)
print(p.bias)
print(p.iter)

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

x1_1 = (-p.weight[0] * x0_1 - p.bias) / p.weight[1]
x1_2 = (-p.weight[0] * x0_2 - p.bias) / p.weight[1]

plt.plot([x0_1, x0_2], [x1_1, x1_2], "r")

##plt.figure(figsize=(5,5))
for ind,x in enumerate(X):
    if y[ind]==1:
        color = 'blue'
    else: color = 'red'
    plt.plot(x[0],x[1],'o', color=color) 

t1=f[0]
t2=f[1]   
plt.plot([t1[0],t2[0]],[t1[1],t2[1]], 'k-')
plt.xlim(-1, 1)  # Setting x-axis limits from -1 to 1
plt.ylim(-1, 1)  # Setting y-axis limits from -1 to 1

# Add grid, labels, and title
plt.grid(True)  # Adding a grid to the plot
plt.xlabel('X-axis')  # Labeling the x-axis
plt.ylabel('Y-axis')  # Labeling the y-axis
plt.title('Perceptron(red) with target function(black)')  # Adding a title to the plot


plt.show()
