import numpy as np
from perceptron_mine import perceptron


def USF(n):
    return np.where(n > 0, 1, 0)

def generate_random_point():
    return np.random.uniform(-1, 1, 2)

def generate_target_function():
    point1 = generate_random_point()
    point2 = generate_random_point()
    
    def classify_point(point):
        x1, y1 = point1
        x2, y2 = point2
        x, y = point
        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross_product > 0:
            return 1
        else:
            return 0
    
    return classify_point

def generate_data(N, target_function):
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.array([target_function(x) for x in X])
    return X, y


N=100
L=[]
for i in range(10):
    p=perceptron(0.01,1000)
    target_function = generate_target_function()
    X, y = generate_data(N, target_function)
    iterations = p.learn(X, y)
    L.append(p.iter)
print(np.mean(L))    
print(p.weight)