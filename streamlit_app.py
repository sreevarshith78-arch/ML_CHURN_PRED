import streamlit as st
import pandas as pd
from typing import Optional

# Import the helper prediction function from your script
from test import load_model_and_predict


def predict_row_dict(row: dict, model_path: str = 'customer_churn_model.pkl', encoder_path: str = 'encoder.pkl'):
    # Uses the helper in test.py to predict for a single-row dict
    pred, prob = load_model_and_predict(row, model_path=model_path, encoder_path=encoder_path)
    return pred, prob


def batch_predict(df: pd.DataFrame, model_path: str = 'customer_churn_model.pkl', encoder_path: str = 'encoder.pkl') -> pd.DataFrame:
    # For each row, convert to dict and call the single-predict function.
    results = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        pred, prob = predict_row_dict(row_dict, model_path=model_path, encoder_path=encoder_path)
        results.append({'prediction': int(pred), 'probability': prob})

    res_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)


def main():
    st.title('Customer Churn — Prediction App')

    st.sidebar.header('Run Options')
    use_example = st.sidebar.checkbox('Use example customer (no upload)', value=True)

    uploaded_file = None
    if not use_example:
        uploaded_file = st.sidebar.file_uploader('Upload CSV with customer rows', type=['csv'])

    st.sidebar.markdown('Model files must be in the same folder as this app:')
    st.sidebar.text('customer_churn_model.pkl')
    st.sidebar.text('encoder.pkl')

    if use_example:
        st.subheader('Example single-customer prediction')
        # Minimal example inputs (you can add more fields if desired)
        gender = st.selectbox('gender', ['Female', 'Male'], index=0)
        SeniorCitizen = st.text_input('SeniorCitizen', '0')
        tenure = st.text_input('tenure', '1')
        MonthlyCharges = st.text_input('MonthlyCharges', '29.85')
        TotalCharges = st.text_input('TotalCharges', '29.85')

        other_fields_expander = st.expander('Other categorical fields (optional)')
        with other_fields_expander:
            Partner = st.selectbox('Partner', ['Yes', 'No'])
            Dependents = st.selectbox('Dependents', ['Yes', 'No'])
            Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

        if st.button('Predict (example)'):
            sample = {
                'gender': gender,
                'SeniorCitizen': SeniorCitizen,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges,
                'Contract': Contract,
            }
            with st.spinner('Running prediction...'):
                pred, prob = predict_row_dict(sample)
            churn_status = 'Yes' if pred == 1 else 'No'
            st.success(f'Predicted Churn: {churn_status}')
            if prob is not None:
                st.write(f'Prediction probability: {prob:.4f}')
            else:
                st.write('Prediction probability: N/A')

    else:
        st.subheader('Batch predictions from CSV')
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write('Preview of uploaded data:')
            st.dataframe(df.head())

            if st.button('Run batch predictions'):
                with st.spinner('Running batch predictions...'):
                    out_df = batch_predict(df)
                st.success('Done — results below')
                st.dataframe(out_df.head(200))
                st.download_button('Download predictions CSV', out_df.to_csv(index=False), file_name='predictions.csv')
        else:
            st.info('Upload a CSV to run batch predictions')


if __name__ == '__main__':
    main()


