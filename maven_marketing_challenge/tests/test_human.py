import os
import pandas as pd
import numpy as np
from solution import human


def create_valid_human_dataframe(n=20):
    np.random.seed(42)

    return pd.DataFrame({
        "ID": np.arange(1, n+1),
        "Year_Birth": np.random.randint(1960, 1995, n),
        "Income": np.random.randint(30000, 80000, n),
        "Kidhome": np.random.randint(0, 2, n),
        "Teenhome": np.random.randint(0, 2, n),
        "Dt_Customer": pd.date_range("2018-01-01", periods=n),
        "NumWebPurchases": np.random.randint(1, 10, n),
        "NumWebVisitsMonth": np.random.randint(1, 20, n),
        "NumStorePurchases": np.random.randint(0, 8, n),
        "NumCatalogPurchases": np.random.randint(0, 5, n),
        "MntWines": np.random.randint(10, 300, n),
        "Response": np.random.randint(0, 2, n),
        "AcceptedCmp1": np.random.randint(0, 2, n),
    })


def test_human_pipeline_creates_output(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    df = create_valid_human_dataframe()
    df.to_csv("marketing_data.csv", index=False)

    human.run_pipeline()

    # 1️⃣ Verifica export
    assert os.path.exists("marketing.csv")

    result = pd.read_csv("marketing.csv")

    # 2️⃣ Features importantes foram criadas
    expected_cols = ["Dependents", "TotalMnt", "TotalPurchases", "TotalCampaignsAcc"]
    for col in expected_cols:
        assert col in result.columns

    # 3️⃣ Número de linhas mantido
    assert len(result) == len(df)