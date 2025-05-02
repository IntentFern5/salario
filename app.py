import pandas as pd
import plotly.express as px
import streamlit as st
import pickle as pkl
import numpy as np

wage = pd.read_csv('wage.csv')


st.title("Salario esperado")

with open("model.pickle", "rb") as m:
    modelo = pkl.load(m)

Tab1, Tab2, Tab3 = st.tabs(['Analisis Univariado', 'Analisis Bivariado', 'Modelo'])
 
with Tab1:

    st.subheader("Estadísticas descriptivas - Variables numéricas")
    st.dataframe(wage[['salario', 'educ', 'exper', 'permanencia']].describe())
 
    st.subheader("Distribución de variables numéricas")
    fig = px.histogram(wage, x='salario')
    st.plotly_chart(fig)
 
    fig2 = px.histogram(wage, x='educ')
    st.plotly_chart(fig2)
 
    fig3 = px.histogram(wage, x='exper')
    st.plotly_chart(fig3)
 
    fig4 = px.histogram(wage, x='permanencia')
    st.plotly_chart(fig4)
 
    st.subheader("Frecuencias - Variables categóricas")
   
    st.write("Género")
    st.dataframe(wage['sexo'].value_counts())
 
    st.write("Estado civil")
    st.dataframe(wage['estado civil'].value_counts())
 
    st.subheader("Distribución de variables categóricas")
    fig5 = px.histogram(wage, x='sexo')
    st.plotly_chart(fig5)
 
    fig6 = px.histogram(wage, x='estado civil')
    st.plotly_chart(fig6)


with Tab2:
    st.subheader("Relación entre variables y salario")
 
    fig1 = px.scatter(wage, x='educ', y='salario', title='Educación vs Salario')
    st.plotly_chart(fig1)
 
    fig2 = px.scatter(wage, x='exper', y='salario', title='Experiencia vs Salario')
    st.plotly_chart(fig2)
 
    fig3 = px.scatter(wage, x='permanencia', y='salario', title='Permanencia vs Salario')
    st.plotly_chart(fig3)
 
    fig4 = px.box(wage, x='sexo', y='salario', title='Salario por Género')
    st.plotly_chart(fig4)
 
    fig5 = px.box(wage, x='estado civil', y='salario', title='Salario por Estado Civil')
    st.plotly_chart(fig5)

with Tab3:

    st.title("Modelo")

    educ = st.slider("Años educacion", 0, 18)

    exper = st.slider("Años de experiencia", 1, 51)

    permanencia= st.slider("Años en la empresa", 0, 44)

    sexo = st.selectbox('sexo', ['female', 'male'])
    if sexo == 'female':
       sexo = 1
    else:
       sexo = 0
    

    estado_civil =  st.selectbox('estado civil', ['married', 'single'])
    if estado_civil == 'married':
       estado_civil = 1
    else:
       estado_civil = 0


    if st.button("Predecir"):
       pred = modelo.predict(np.array([[educ, exper, permanencia, sexo, estado_civil]]))
       st.write(pred [0])

       print(10)
 