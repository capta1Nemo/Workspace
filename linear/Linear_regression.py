#kendim oluşturduğum linear regression kodu
#L2 (MSE) error
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)
def data_generator(N):
    X,noise = np.random.rand(2,N)
    m,b=np.random.rand(2,1)
    print(f'initial weight and bias for data creation:{m},{b}')
    y=m*X+b
    y+= noise/5
    return X,y 
N=100
d=data_generator(N)
x=d[0]
y=d[1]  
  
def calculate_error(m,b,d):
    y=d[1]
    x=d[0] 
    disagrement=0
    for i in range(len(x)):
        disagrement += (y[i]-m*x[i]-b)**2
    return disagrement/len(x)    

def descent(m,b,d,l):
    x=d[0]
    y=d[1]
    dEm=0
    dEb=0
    for i in range(len(x)):
        dEm += x[i] * (y[i] - m*x[i] - b)
        dEb += y[i] - m*x[i] - b
    dEb = -(dEb/len(x))*2  
    dEm= -(dEm/len(x))*2   
    m = m - l*dEm
    b = b - l*dEb
    return m, b

def linear_regression(d,l,max_iter):
    m=0
    b=0
    for i in range(max_iter):
        m,b=descent(m,b,d,l)
    return m,b
    
m,b=linear_regression(d,0.01,10000)

print(f'{float(m),float(b)} error:{calculate_error(m,b,d)}')
t=np.linspace(0,1,100)
z=m*t+b
plt.scatter(x,y,color='black')
plt.plot(t,z,color='red')
plt.show()
np.random.seed(0)
n_samples_per_class = 50