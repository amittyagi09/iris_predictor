import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("Simple Iris Flower Pridictor")
st.write("This app pridicts **Iris Flower** type.")

st.sidebar.header("user Input Features")
sepal_length=st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
sepal_width=st.sidebar.slider("Sepal width", 2.0, 4.4, 3.4)
petal_lenth=st.sidebar.slider("Petal Length", 1.0, 6.9, 1.3)
petal_width=st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

def load_data():
    data={"Sepal Length":sepal_length,
          "Sepal Width":sepal_width,
          "Petal Length":petal_lenth,
          "Petal Width":petal_width}
    data=pd.DataFrame(data, index=[0])
    return data

data=load_data()

st.subheader("User Input Parameters")
st.write(data)

iris=datasets.load_iris()
x=iris.data
y=iris.target

clf=RandomForestClassifier()
clf.fit(x,y)

prediction=clf.predict(data)
prediction_proba=clf.predict_proba(data)

st.subheader("Class labels and their corrsponding index number")
st.dataframe(iris.target_names)

st.subheader("Pridiction")
st.write(iris.target_names[prediction])

st.subheader("Prediction probability")
st.write(prediction_proba)