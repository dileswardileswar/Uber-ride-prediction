import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r'F:\8.End_to_End_Project\1.Uber Ride weekly Analisys\MY_Own_Code\taxi.csv')
print(data.head())
print(data.shape)
data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values
print(type(data_x))
print(data_x)
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=0)
reg = LinearRegression()
reg.fit(x_train,y_train)
print("Train Score",reg.score(x_train,y_train))
print("Test Score",reg.score(x_test,y_test))
pickle.dump(reg,open(r'F:\8.End_to_End_Project\1.Uber Ride weekly Analisys\MY_Own_Code\taxi.pkl','wb'))
model = pickle.load(open(r'F:\8.End_to_End_Project\1.Uber Ride weekly Analisys\MY_Own_Code\taxi.pkl','rb'))
print(model.predict([[59,1779000,4900,97]]))