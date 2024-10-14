import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simular un modelo básico (normalmente deberías cargar un modelo preentrenado)
model = RandomForestClassifier()

# Datos de ejemplo para entrenar un modelo básico
data = pd.DataFrame({
    'Mileage': [15000, 30000],
    'Reported_Issues': [1, 0],
    'Vehicle_Age': [5, 3],
    'Engine_Size': [2000, 1600],
    'Odometer_Reading': [50000, 45000],
    'Fuel_Efficiency': [15, 18],
    'Tire_Condition': [2, 1],
    'Brake_Condition': [2, 1],
    'Battery_Status': [1, 2]
})
X = data.drop('Battery_Status', axis=1)
y = data['Battery_Status']
model.fit(X, y)

# Título de la app
st.title('Predicción de Mantenimiento de Vehículos')

# Crear campos para que el usuario ingrese datos
def user_input_features():
    Mileage = st.number_input('Kilometraje (en kilómetros)', min_value=0)
    Reported_Issues = st.number_input('Problemas reportados', min_value=0)
    Vehicle_Age = st.number_input('Edad del vehículo (en años)', min_value=0)
    Engine_Size = st.number_input('Tamaño del motor (en cc)', min_value=0)
    Odometer_Reading = st.number_input('Lectura del odómetro', min_value=0)
    Fuel_Efficiency = st.number_input('Eficiencia del combustible (km/l)', min_value=0.0)
    Tire_Condition = st.selectbox('Condición de los neumáticos', ('Nuevo', 'Usado', 'Desgastado'))
    Brake_Condition = st.selectbox('Condición de los frenos', ('Nuevo', 'Usado', 'Desgastado'))
    
    data = {
        'Mileage': Mileage,
        'Reported_Issues': Reported_Issues,
        'Vehicle_Age': Vehicle_Age,
        'Engine_Size': Engine_Size,
        'Odometer_Reading': Odometer_Reading,
        'Fuel_Efficiency': Fuel_Efficiency,
        'Tire_Condition': Tire_Condition,
        'Brake_Condition': Brake_Condition,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Mostrar los datos ingresados
st.subheader('Datos ingresados:')
st.write(input_df)

# Realizar la predicción (dummy model)
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Predicción:')
st.write('Necesita mantenimiento' if prediction[0] == 1 else 'No necesita mantenimiento')

st.subheader('Probabilidad:')
st.write(f"Probabilidad de necesitar mantenimiento: {prediction_proba[0][1] * 100:.2f}%")
