import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_and_prepare(path='test.csv'):
  df = pd.read_csv(path)
  # Drop identifier column if present
  if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

  # Replace blank TotalCharges with 0 and convert
  if 'TotalCharges' in df.columns:
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

  # Map target
  if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

  # Determine object columns for label encoding
  object_columns = df.select_dtypes(include='object').columns.tolist()

  # Fit label encoders for object columns and transform
  encoders = {}
  for col in object_columns:
    le = LabelEncoder()
    # Fill NA so LabelEncoder won't fail
    df[col] = df[col].fillna('')
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

  return df, encoders


def train_and_save_model(df, encoders, model_path='customer_churn_model.pkl', encoder_path='encoder.pkl'):
  x = df.drop(columns=['Churn'])
  y = df['Churn']

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

  smote = SMOTE(random_state=40)
  X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

  # Train a RandomForest as the final model
  rfc = RandomForestClassifier(random_state=44)
  rfc.fit(X_train_smote, y_train_smote)

  model_data = {"model": rfc, "feature_names": x.columns.tolist()}

  with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

  with open(encoder_path, 'wb') as f:
    pickle.dump(encoders, f)

  return model_path, encoder_path


def load_model_and_predict(sample_dict, model_path='customer_churn_model.pkl', encoder_path='encoder.pkl'):
  with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

  with open(encoder_path, 'rb') as f:
    encoders = pickle.load(f)

  model = model_data['model']
  feature_names = model_data['feature_names']

  input_df = pd.DataFrame([sample_dict])

  # Apply encoders to categorical columns if present
  for col, le in encoders.items():
    if col in input_df.columns:
      # Ensure the same categories exist; unseen labels will raise, so fillna and map unknowns to '' then transform
      input_df[col] = input_df[col].fillna('')
      try:
        input_df[col] = le.transform(input_df[col])
      except Exception:
        # If transform fails because of unseen labels, map to -1
        input_df[col] = input_df[col].apply(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

  # Convert numeric columns
  for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
    if col in input_df.columns:
      input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

  # Ensure columns order matches training features
  input_df = input_df.reindex(columns=feature_names, fill_value=0)

  prediction = model.predict(input_df)
  # prediction probability may not exist for all estimators
  try:
    pred_prob = model.predict_proba(input_df)[0][prediction[0]]
  except Exception:
    pred_prob = None

  return int(prediction[0]), pred_prob


if __name__ == '__main__':
  # Load and prepare data
  df, encoders = load_and_prepare('test.csv')

  # Train model and save artifacts
  model_path, encoder_path = train_and_save_model(df, encoders)
  print(f"Model saved to {model_path}; encoders saved to {encoder_path}")

  # Example input for prediction
  customer_data = {
    'gender': 'Female',
    'SeniorCitizen': '0',
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': '1',
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': '29.85',
    'TotalCharges': '29.85'
  }

  pred, prob = load_model_and_predict(customer_data, model_path=model_path, encoder_path=encoder_path)
  churn_status = 'Yes' if pred == 1 else 'No'
  print(f"Predicted Churn Status for the given customer: {churn_status}")
  if prob is not None:
    print(f"prediction_probability: {prob:.4f}")
  else:
    print("prediction_probability: N/A")