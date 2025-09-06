import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#Loading dataset
iris=datasets.load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species']=iris.target
print(df.head())

#Scatter plot : Sepal Length vs Sepal Width
plt.scatter=(df['sepal length (cm)'],df['sepal width (cm)'],c=df['species'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Sepal Length vs Width')
plt.show()