#burada classification için birinci periyodda yazmaya çalıştığım bir kod var 
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
#reLU
def USF(n):
    if n>0:
        return 1
    else: return 0


class perceptron():
    def __init__(self,learning_rate, max_iterations):
        self.l=learning_rate
        self.max_it=max_iterations
        self.activation_func = USF
        self.iter=0
        self.weight=0
        self.bias=0

    def learn(self,x,y): #x data y target
        model=LinearRegression()
        model.fit(x,y)#scikit algoritmasini 
        #weight ve bıas için başlangıç değeri bulmak için kullanıyorum
        sample_size,atribute_size=x.shape
        self.weight=model.coef_
        self.bias=model.intercept_
        misclassified = [i for i in range (sample_size)] 
        self.iter=0
        while self.iter <= self.max_it and len(misclassified) != 0:
            temp = misclassified
            for i in temp:
                y_in = np.dot(self.weight,x[i])+ self.bias #y=w.x+b
                result=self.activation_func(y_in)
                if result==y[i]:
                    misclassified.remove(i)
                 #update weights
                update = self.l*(y[i] - result) 
                for t in range(len(self.weight)):
                    self.weight[t] += update * x[i][t] 
                self.bias += update
                self.iter += 1
            if len(misclassified)==0: #son kontrol
                for i in range(sample_size):
                    y_in = np.dot(self.weight,x[i]) + self.bias #y=w.x+b
                    result=self.activation_func(y_in)
                    if result != y[i]:
                        misclassified.append(i)  
    def predict(self,x):
        y_in = np.dot(x,self.weight)+ self.bias #y=w.x+b
        result=self.activation_func(y_in)
        return result
        