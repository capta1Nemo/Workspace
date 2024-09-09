import numpy as np

def generate_random_point():
    return np.random.uniform(-1, 1, 2)

def generate_target_function():
    point1 = generate_random_point()
    point2 = generate_random_point()
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    intercept = point1[1] - slope * point1[0]
    return lambda x: np.sign(x[1] - (slope * x[0] + intercept))

def generate_data(N, target_function):
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.array([target_function(x) for x in X])
    return X, y

def perceptron_learning_algorithm(X, y):
    N = len(y)
    X = np.hstack((np.ones((N, 1)), X))  # Add bias term
    w = np.zeros(3)
    iterations = 0
    while True:
        misclassified_points = []
        for i in range(N):
            if np.sign(np.dot(w, X[i])) != y[i]:
                misclassified_points.append((X[i], y[i]))
        if not misclassified_points:
            break
        x_i, y_i = misclassified_points[np.random.randint(len(misclassified_points))]
        w += y_i * x_i
        iterations += 1
    return iterations

# Parameters
N = 10
num_runs = 1000

iterations_list = []

for _ in range(num_runs):
    target_function = generate_target_function()
    X, y = generate_data(N, target_function)
    iterations = perceptron_learning_algorithm(X, y)
    iterations_list.append(iterations)

average_iterations = np.mean(iterations_list)
print(f"Average number of iterations for PLA to converge: {average_iterations}")