import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate the synthetic dataset
#np.random.seed(0)
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
            return -1
    
    return [point1,point2 ,classify_point]

f = generate_target_function()
cls = f[2]
X = np.random.uniform(-1, 1, (100,2))
y = np.array([cls(x) for x in X])

# Combine the data
x1 = X[:,0]
x2 = X[:,1]
model=LinearRegression()
model.fit(X,y)
coef = model.coef_
intercept = model.intercept_
t1 = np.linspace(-1, 1, 1000)
t2 = np.linspace(-1, 1, 1000)
t1,t2=np.meshgrid(t1,t2)
a = coef[0]
b = coef[1]
z = t1*a + t2*b + intercept


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot
ax.scatter(x1, x2, y, c='r', marker='o')
ax.plot_surface(t1, t2, z, alpha=0.5, rstride=100, cstride=100, color='yellow')
# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show plot
plt.show()


