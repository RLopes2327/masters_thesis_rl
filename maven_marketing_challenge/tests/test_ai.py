import os
import builtins
import pandas as pd
import numpy as np
from solution import ai


def create_valid_ai_dataframe(n=20):
    np.random.seed(42)

    return pd.DataFrame({
        "Year_Birth": np.random.randint(1960, 1995, n),
        "Income": np.random.randint(30000, 80000, n),
        "Kidhome": np.random.randint(0, 2, n),
        "Teenhome": np.random.randint(0, 2, n),
        "Dt_Customer": pd.date_range("2018-01-01", periods=n),
        "NumWebPurchases": np.random.randint(1, 10, n),
        "NumWebVisitsMonth": np.random.randint(1, 20, n),
        "NumStorePurchases": np.random.randint(0, 8, n),
        "NumCatalogPurchases": np.random.randint(0, 5, n),
        "Response": np.random.randint(0, 2, n),
        "MntWines": np.random.randint(10, 300, n),
    })


def test_ai_pipeline_creates_silver(tmp_path, monkeypatch):

    monkeypatch.setattr(builtins, "display", lambda *args, **kwargs: None, raising=False)
    monkeypatch.chdir(tmp_path)

    df = create_valid_ai_dataframe()
    df.to_csv("marketing_data.csv", index=False)

    ai.run_pipeline()

    # 1️⃣ Verifica se criou o ficheiro
    assert os.path.exists("silver_layer_marketing_data.csv")

    # 2️⃣ Verifica se manteve o número de linhas
    result = pd.read_csv("silver_layer_marketing_data.csv")
    assert len(result) == len(df)

    # 3️⃣ Verifica se criou features importantes
    expected_cols = ["Age", "Total_Spend", "Total_Purchases"]
    for col in expected_cols:
        assert col in result.columns