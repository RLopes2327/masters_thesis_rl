# %%
import pandas as pd

def run_pipeline():

    # %%
    df = pd.read_csv("marketing_data.csv")

    # %%
    df

    # %%
    df.info()

    # %%
    #24 missing in income so instead of removing roes lets replace by median for income for missing values

    # %%
    # clean up column names that contain whitespace
    df.columns = df.columns.str.replace(' ', '')

    # %%
    df['Income'] = df['Income'].fillna(df['Income'].median())

    # %%
    df = df[df['Year_Birth'] > 1900].reset_index(drop=True)

    df['Year_Birth'].plot(kind='box', patch_artist=True);

    # %%
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

    # %%
    # Dependents
    df['Dependents'] = df['Kidhome'] + df['Teenhome']

    # Year becoming a Customer
    df['Year_Customer'] = pd.DatetimeIndex(df['Dt_Customer']).year

    # Total Amount Spent
    mnt_cols = [col for col in df.columns if 'Mnt' in col]
    df['TotalMnt'] = df[mnt_cols].sum(axis=1)

    # Total Purchases
    purchases_cols = [col for col in df.columns if 'Purchases' in col]
    df['TotalPurchases'] = df[purchases_cols].sum(axis=1)

    # Total Campaigns Accepted
    campaigns_cols = [col for col in df.columns if 'Cmp' in col] + ['Response'] # 'Response' is for the latest campaign
    df['TotalCampaignsAcc'] = df[campaigns_cols].sum(axis=1)



    # %%
    # view new features, by customer ID
    df[['ID', 'Dependents', 'Year_Customer', 'TotalMnt', 'TotalPurchases', 'TotalCampaignsAcc']].head()

    # %%
    df

    # %%
    df.to_csv("marketing.csv")

    # %%
    df['TotalPurchases'].sum()

    # %%
    df['NumWebVisitsMonth']

    # %%
    len(df['ID'].unique())

    # %%
    # - Campaign success
    # - Mode of Purchase
    # - Demographics - Age, Income, Marital status,Educartion
    # - campaign sales Trend over time
    # - Recency vs sUCCESS
    # - COUNTRY VS SUCCESS

    # %%

if __name__ == "__main__":
    run_pipeline()

