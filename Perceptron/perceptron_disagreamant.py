import numpy as np
from perceptron_mine import perceptron 


def generate_target_function():
    point1 = np.random.uniform(-1, 1, 2)
    point2 = np.random.uniform(-1, 1, 2)
    
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
function = generate_target_function()
X = np.random.uniform(-1, 1, (100, 2))
y = np.array([function(x) for x in X])
p=perceptron(0.001,10000)
p.learn(X,y)

X2=X = np.random.uniform(-1, 1, (1000, 2))#test data
y2 = np.array([function(x) for x in X])#test output
y_in=np.array([p.predict(x) for x in X2])#perceptron output

def compare(x1,x2):
    disagreamant=0
    for i in range(len(x1)):
        if x1[i]!=x2[i]:
            disagreamant +=1
    return disagreamant/len(x1)  
result=0
for i in range(100):
    result+=compare(y2,y_in)
print(f'disgreement is %{result}')
