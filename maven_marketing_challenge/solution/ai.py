# %%
# Imports & global config
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import statsmodels.api as sm

def run_pipeline():

    pd.set_option('display.max_columns', 200)
    sns.set(style='whitegrid')
    np.random.seed(42)

    # DATA_PATH = 'maven_marketing_challenge/marketing_data.csv'
    # DICT_PATH = 'maven_marketing_challenge/marketing_data_dictionary.csv'
    # EXPORT_PATH = 'maven_marketing_challenge/silver_layer_marketing_data.csv'

    DATA_PATH = 'marketing_data.csv'
    DICT_PATH = 'marketing_data_dictionary.csv'
    EXPORT_PATH = 'silver_layer_marketing_data.csv'

    def safe_div(a, b):
        return np.where(b==0, np.nan, a/b)

    # %%
    # Ingest
    df = pd.read_csv(DATA_PATH)
    data_dict = None
    try:
        data_dict = pd.read_csv(DICT_PATH)
    except Exception:
        data_dict = None

    # Coerce likely numeric columns to numeric to avoid text types in Power BI
    num_like = [
        c for c in df.columns
        if c.startswith(('Mnt','Num')) or c in ['Income','Recency','Kidhome','Teenhome','Year_Birth','TenureMonths','Share_Store','Share_Catalog']
    ]
    for c in num_like:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Parse likely date columns
    for col in df.columns:
        if 'Dt_' in col or 'Date' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    print('Shape:', df.shape)
    print('Columns:', list(df.columns))
    display(df.head(3))

    # Null overview
    nulls = df.isna().sum().sort_values(ascending=False)
    display(nulls[nulls>0])

    # Dtype overview
    display(df.dtypes)

    # %% [markdown]
    # ## Cleaning functions
    # - Nulls: median for numeric, mode for categorical; dates left as NaT
    # - Outliers: IQR capping for monetary and count-like variables ($1.5\times IQR$)
    # - Transformations: $\log(1+x)$ for right-skewed positive variables to stabilize variance

    # %%
    def is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                # Keep NaT; dates often informative when missing
                continue
            if is_numeric_series(s):
                if s.isna().any():
                    med = s.median()
                    # Median is robust to outliers; preserves distribution center
                    df[col] = s.fillna(med)
            else:
                if s.isna().any():
                    mode = s.mode(dropna=True)
                    fill_val = mode.iloc[0] if len(mode)>0 else 'Unknown'
                    df[col] = s.fillna(fill_val)
        return df

    def cap_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 1.5) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col not in df.columns or not is_numeric_series(df[col]):
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            # Capping (winsorizing) preserves sample size; avoids undue influence
            df[col] = np.clip(df[col], lower, upper)
        return df

    def add_log_transforms(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in df.columns and is_numeric_series(df[col]):
                if (df[col] >= 0).all():
                    df[f'log1p_{col}'] = np.log1p(df[col])
        return df

    def detect_skewed_positive_cols(df: pd.DataFrame, candidates: list, skew_thr: float = 1.0) -> list:
        skewed = []
        for col in candidates:
            if col in df.columns and is_numeric_series(df[col]):
                s = df[col].dropna()
                if len(s) > 0 and s.min() >= 0:
                    sk = stats.skew(s)
                    if sk > skew_thr:
                        skewed.append(col)
        return skewed

    # %% [markdown]
    # ## Feature engineering
    # - Age from Year_Birth
    # - Tenure in months from Dt_Customer
    # - Total_Spend from product amounts (Mnt*)
    # - Total_Purchases and channel shares
    # - Campaign acceptance aggregates
    # - Web conversion (NumWebPurchases / NumWebVisitsMonth)

    # %%
    def product_amount_cols(df):
        return [c for c in df.columns if c.startswith('Mnt')]

    def purchase_count_cols(df):
        return [c for c in df.columns if ('Purchases' in c and c.startswith('Num'))]

    def campaign_cols(df):
        # AcceptedCmp1..5 and Response
        acc = [c for c in df.columns if c.startswith('AcceptedCmp')]
        resp = [c for c in df.columns if c == 'Response']
        return acc + resp

    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Age
        if 'Year_Birth' in df.columns:
            current_year = pd.Timestamp.today().year
            df['Age'] = current_year - df['Year_Birth']
            # Cap implausible ages
            df['Age'] = df['Age'].clip(lower=18, upper=100)
        # Tenure in months
        if 'Dt_Customer' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Dt_Customer']):
            df['TenureMonths'] = ((pd.Timestamp.today() - df['Dt_Customer']).dt.days / 30.44).round().astype('Int64')
        # Spend & purchases
        mnt_cols = product_amount_cols(df)
        df['Total_Spend'] = df[mnt_cols].sum(axis=1) if len(mnt_cols)>0 else np.nan
        purch_cols = purchase_count_cols(df)
        df['Total_Purchases'] = df[purch_cols].sum(axis=1) if len(purch_cols)>0 else np.nan
        # Channel shares
        if 'NumWebPurchases' in df.columns:
            df['Share_Web'] = safe_div(df['NumWebPurchases'], df['Total_Purchases'])
        if 'NumCatalogPurchases' in df.columns:
            df['Share_Catalog'] = safe_div(df['NumCatalogPurchases'], df['Total_Purchases'])
        if 'NumStorePurchases' in df.columns:
            df['Share_Store'] = safe_div(df['NumStorePurchases'], df['Total_Purchases'])
        # Web conversion
        if 'NumWebPurchases' in df.columns and 'NumWebVisitsMonth' in df.columns:
            df['Web_Conversion'] = safe_div(df['NumWebPurchases'], df['NumWebVisitsMonth'])
        # Campaign acceptance
        acc_cols = campaign_cols(df)
        if len(acc_cols) > 0:
            df['Campaign_Accepted_Any'] = df[acc_cols].sum(axis=1).clip(upper=1)
            df['Campaign_Accepted_Count'] = df[acc_cols].sum(axis=1)
        return df

    # %% [markdown]
    # ## Apply cleaning & feature engineering
    # - Impute nulls
    # - Cap outliers for monetary and purchase count columns
    # - Transform skewed monetary columns

    # %%
    # Make a working copy
    work = df.copy()
    work = impute_nulls(work)

    # Outlier capping (IQR) for spend and count-like columns
    mnt_cols = product_amount_cols(work)
    num_cols = purchase_count_cols(work)
    extra_cols = [c for c in ['Income', 'Recency'] if c in work.columns]
    cap_cols = list(set(mnt_cols + num_cols + extra_cols))
    work = cap_outliers_iqr(work, cap_cols, k=1.5)

    # Log transforms for skewed positive variables
    skew_candidates = mnt_cols + [c for c in ['Income'] if c in work.columns]
    skewed = detect_skewed_positive_cols(work, skew_candidates, skew_thr=1.0)
    work = add_log_transforms(work, skewed)

    # Engineer features
    work = engineer_features(work)
    display(work.head(3))

    # Basic sanity checks
    assert work.shape[0] == df.shape[0], 'Row count changed unexpectedly.'
    print('Engineered columns added:', [c for c in work.columns if c not in df.columns][:10], '...')

    # %% [markdown]
    # ## Q1: Factors related to number of web purchases
    # Approach:
    # - Pearson correlations
    # - OLS with standardized predictors
    # Correlation formula: $$\rho_{X,Y} = \frac{n\sum XY - (\sum X)(\sum Y)}{\sqrt{\left[n\sum X^2 - (\sum X)^2\right]\left[n\sum Y^2 - (\sum Y)^2\right]}}$$

    # %%
    targets = ['NumWebPurchases']
    predictors_pref = [
        'NumWebVisitsMonth','Income','Total_Spend','Age','Recency',
        'NumCatalogPurchases','NumStorePurchases','TenureMonths','Campaign_Accepted_Count'
    ]
    predictors = [c for c in predictors_pref if c in work.columns]
    y = work[targets[0]] if targets[0] in work.columns else None
    X = work[predictors].copy() if len(predictors)>0 else None

    if y is not None and X is not None:
        # Correlations
        corr = pd.Series({c: np.corrcoef(work[c], y)[0,1] for c in predictors})
        display(corr.sort_values(ascending=False))
        
        # OLS with standardized predictors
        X_std = (X - X.mean()) / X.std(ddof=0)
        X_std = sm.add_constant(X_std)
        # Ensure numeric dtypes to avoid object/pd.NA issues
        X_std = X_std.apply(pd.to_numeric, errors='coerce').astype(float)
        y_num = pd.to_numeric(y, errors='coerce').astype(float)
        model = sm.OLS(y_num, X_std, missing='drop').fit()
        print(model.summary())
    else:
        print('Required columns for web purchases analysis not found.')

    # %% [markdown]
    # ## Q2: Which marketing campaign is most successful?
    # Measured by acceptance (mean of AcceptedCmpX or Response).

    # %%
    acc_cols = [c for c in work.columns if c.startswith('AcceptedCmp')] + [c for c in work.columns if c=='Response']
    if len(acc_cols)>0:
        rates = work[acc_cols].mean().sort_values(ascending=False)
        display(rates.to_frame('AcceptanceRate'))
        print('Top campaign:', rates.index[0])
    else:
        print('No campaign columns found.')

    # %% [markdown]
    # ## Q3: What does the average customer look like?

    # %%
    profile_cols = [c for c in ['Age','Income','TenureMonths','Total_Spend','Total_Purchases','Share_Web','Share_Catalog','Share_Store'] if c in work.columns]
    display(work[profile_cols].describe().loc[['mean','50%']].rename(index={'50%':'median'}))

    cat_cols = [c for c in ['Education','Marital_Status','Kidhome','Teenhome'] if c in work.columns]
    for c in cat_cols:
        display(work[c].value_counts(normalize=True).mul(100).round(1).rename('pct'))

    # %% [markdown]
    # ## Q4: Which products are performing best?
    # Rank by total spend and average spend per customer.

    # %%
    mnt_cols = [c for c in work.columns if c.startswith('Mnt')]
    if len(mnt_cols)>0:
        totals = work[mnt_cols].sum().sort_values(ascending=False)
        avgs = work[mnt_cols].mean().sort_values(ascending=False)
        print('Total spend by product:')
        display(totals.to_frame('TotalSpend'))
        print('Average spend by product:')
        display(avgs.to_frame('AvgSpend'))
    else:
        print('No product amount columns found.')

    # %% [markdown]
    # ## Q5: Which channels are underperforming?
    # Approximate conversion rates:
    # - Web: NumWebPurchases / NumWebVisitsMonth
    # - Store & Catalog: relative purchase shares (as proxy if no visit/contact counts)

    # %%
    metrics = {}
    if 'NumWebPurchases' in work.columns and 'NumWebVisitsMonth' in work.columns:
        metrics['Web_Conversion_Rate'] = np.nanmean(work['Web_Conversion'])
    if 'NumStorePurchases' in work.columns and 'Total_Purchases' in work.columns:
        metrics['Store_Share'] = np.nanmean(work['Share_Store'])
    if 'NumCatalogPurchases' in work.columns and 'Total_Purchases' in work.columns:
        metrics['Catalog_Share'] = np.nanmean(work['Share_Catalog'])
    print('Channel metrics:')
    display(pd.Series(metrics))

    # Identify underperforming as lowest conversion/share
    if len(metrics)>0:
        under = pd.Series(metrics).sort_values(ascending=True)
        print('Underperforming channel (lowest metric):', under.index[0])
    else:
        print('Not enough columns to compute channel performance.')

    # %% [markdown]
    # ## Export silver layer
    # Save cleaned and enriched dataset for Power BI.

    # %%
    work.to_csv(EXPORT_PATH, index=False)
    print('Exported:', EXPORT_PATH, 'shape:', work.shape)


if __name__ == "__main__":
    run_pipeline()