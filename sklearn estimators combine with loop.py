from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=15)
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(),
          KMeans(), linear_model.LinearRegression()]
model_names = ["Logistic Regression :", "SVM :", "Decision Tree :", "Random Forest :", "KNN :", "K-Mean :",
               "Linear Regression :"]
models_scores = []
for model, model_name in zip(models, model_names):
    model.fit(x_train, y_train)
    y_pred = model.score(x_test, y_test)
    print(model_name, y_pred)
