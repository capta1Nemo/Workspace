
#scikit learnin lienear regression icin kütüphanesi varmış 
# onu denemek için    
import matplotlib.pyplot as plt 
from sklearn import linear_model  
import pandas as pd 
   
#buradaki datadan fiyat ve evin boyutunu aldım 
df = pd.read_csv(r"/home/master/Desktop/Workspace/linear/Housing.csv")
Y = df['price'] 
X = df['lotsize'] 
print(df)

   
X=X.values.reshape(len(X),1) 
Y=Y.values.reshape(len(Y),1) 
   
# 250test datası
X_train = X[:-250] 
X_test = X[-250:] 
   
Y_train = Y[:-250] 
Y_test = Y[-250:] 
   
plt.scatter(X_test, Y_test,  color='black') 
plt.title('Test Data') 
plt.xlabel('Size') 
plt.ylabel('Price') 
plt.xticks(()) 
plt.yticks(()) 
  
regr = linear_model.LinearRegression() 
   
regr.fit(X_train, Y_train) 
   
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3) 
plt.show() 