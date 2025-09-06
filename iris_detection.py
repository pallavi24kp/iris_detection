import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#Loading dataset
iris=datasets.load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species']=iris.target
print(df.head())

#Scatter plot : Sepal Length vs Sepal Width (comment this to check the accuracy)
plt.scatter(df['sepal length (cm)'],df['sepal width (cm)'],c=df['species'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Sepal Length vs Width')
plt.show()

from sklearn.model_selection import train_test_split

#Features and Labels
X=df.iloc[:,:-1]
y=df['species']

#Train-Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

#Create and Train Model
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,classification_report

#Making predictions
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred),flush=True)
print("Classification Report:\n",classification_report(y_test,y_pred,target_names=iris.target_names))

#Predict new sample
import pandas as pd
sample=pd.DataFrame([[5.1,3.5,1.4,0.2]],columns=iris.feature_names)
prediction=model.predict(sample)
print("Predicted Species :",iris.target_names[prediction[0]])
