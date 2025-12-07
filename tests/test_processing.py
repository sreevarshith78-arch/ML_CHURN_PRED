import pandas as pd

from process_csv import clean, prepare_features


def make_sample_df():
    return pd.DataFrame(
        [
            {
                "customerID": "A-1",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "InternetService": "DSL",
                "MonthlyCharges": 70.0,
                "TotalCharges": "350",
                "Churn": "No",
            },
            {
                "customerID": "B-2",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "No",
                "InternetService": "Fiber optic",
                "MonthlyCharges": 99.0,
                "TotalCharges": "",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_and_prepare():
    df = make_sample_df()
    dfc = clean(df)
    assert "TotalCharges" in dfc.columns
    # TotalCharges for second row should be filled (MonthlyCharges * tenure = 198)
    assert dfc.loc[1, "TotalCharges"] == 198.0

    X, y = prepare_features(dfc)
    # features should not include customerID or Churn
    assert "customerID" not in X.columns
    assert len(y) == len(dfc)
