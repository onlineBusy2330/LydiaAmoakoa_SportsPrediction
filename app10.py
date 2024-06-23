import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

def main():
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Overall Rating Of A Player </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.title('Machine Learning Model Deployment')
    st.write('This web application predicts using a trained XGBoost model.')

    # Load the trained model and selected features dictionary
    model = pkl.load(open('XGBRegressor_tuned (2).pkl', 'rb'))
    input_fields = pkl.load(open('selected_features_dict.pkl', 'rb'))

    # Create input fields for user input
    user_inputs = {}
    for field, properties in input_fields.items():
        user_inputs[field] = st.number_input(field, min_value=properties['min_value'],
                                             max_value=properties['max_value'],
                                             value=properties['sample_value'])

    # Prepare input data for prediction
    input_data = pd.DataFrame([user_inputs])
    st.write('Input Data:')
    st.write(input_data)

    # Make prediction on button click
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.write(f'The predicted value is: {prediction}')

if __name__ == '__main__':
    main()



