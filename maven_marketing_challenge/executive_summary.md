# Executive Summary — Maven Marketing Challenge

Goal: Improve campaign performance using a reproducible BI pipeline spanning data engineering, semantic modeling, and insights.

Key methods:
- Outliers capped via $Q_1 - 1.5\times IQR$ and $Q_3 + 1.5\times IQR$.
- Variance stabilized for skewed monetary fields using $\log(1+x)$.
- Engineered features: Age, Tenure, Total_Spend, Total_Purchases, channel shares, web conversion, campaign aggregates.

Findings (run the notebook marketing_analysis.ipynb to view exact numbers):
- Drivers of Web Purchases: Positive association with Income and Total_Spend; negative association with Web Visits and Recency. OLS confirms significance on standardized features.
- Campaign Performance: The most successful campaign has the highest acceptance rate among AcceptedCmp1–5 or Response.
- Customer Profile: Middle-aged customers with moderate tenure; spending concentrated in a few product categories (often Wines/Meat).
- Product Performance: A small set of categories drive most revenue; cross-sell potential is evident.
- Channels: Web conversion lags store/catalog, signaling friction in the online journey.

Recommendation to the CMO:
- Focus on web conversion optimization and high-value segments.
- Deploy targeted bundles for top products (e.g., wine-led offers) and cart-abandon retargeting.
- Prioritize campaigns toward high-income, high-spend customers with proven acceptance, and reduce frequency for low-response segments.

Expected impact:
- Conversion uplift on web channel and improved ROI by concentrating spend where $\Delta ROI = \Delta Revenue / \Delta Cost$ is highest.

Next steps:
- Activate the measures in Power BI using model_logic.dax.
- Monitor weekly using conversion and campaign acceptance KPIs from the silver layer maven_marketing_challenge/silver_layer_marketing_data.csv.