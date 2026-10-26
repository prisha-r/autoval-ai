AutoVal: Pre-Owned Vehicle Valuation & Depreciation Analytics

Website Application- https://autoval-ai-cqghczi4fpv3u3yaxdr34r.streamlit.app/

Data-driven pre-owned vehicle valuation and non-linear depreciation analytics web application built with Python, Streamlit, and XGBoost. 
Developed as a B.Tech Data Analytics mini-project at the Manipal Institute of Technology (MIT), MAHE.


Overview

AutoVal is an end-to-end data analytics and machine learning solution which is designed to address the volatility and pricing inefficiencies in the pre-owned automotive market. By integrating econometric depreciation modeling, supervised ensemble regression, and statistical confirmatory tests, this platform offers robust valuation insights and interactive predictive capabilities.


Key Features

Depreciation & Segment Analytics: Multi-line trajectory plots examining residual values across brand tiers, fuel architectures, and transmission types, paired with feature correlation heatmaps and cross-sectional age-price gradients.
Live Valuation Engine: Real-time machine learning inference using serialized ensemble regression models, complete with counterfactual baseline comparisons and annual compound depreciation rate calculations.
Model Benchmarking & CDA: Supervised regression comparison suite (Linear Regression, Random Forest, and XGBoost) evaluated on log-transformed price scales, alongside Confirmatory Data Analysis including Welch t-tests and One-Way ANOVA testing.


Language: Python

Web Framework: Streamlit Cloud

Machine Learning & Preprocessing: Scikit-learn, XGBoost, Joblib, SciPy

Data Manipulation & Visualization: Pandas, NumPy, Matplotlib, Seaborn


Project Structure

├── application.py # Main Streamlit application script 

├── Car details v3.csv # Pre-owned vehicle dataset 

├── autoval_model.pkl # Pre-trained XGBoost regression pipeline 

├── requirements.txt # Python package dependencies 

└── README.md # Project documentation


Local Setup & Installation

Clone the repository: git clone https://github.com/prisha-r/autoval-ai cd autoval

Install dependencies: pip install -r requirements.txt

Run the application locally: streamlit run application.py


References

V. Viswanatha et al., "Predicting the Price of Used Cars using Machine Learning," Proc. EASCT, 2023.

Y. Li et al., "Research on Used Car Price Prediction Based on Random Forest and LightGBM," Proc. ICDSCA, 2022.

S. Bergmann and S. Feuerriegel, "Machine Learning for Predicting Used Car Resale Prices," Expert Syst. Appl., vol. 261, 2025.

A. R. Lakamsani et al., "Machine Learning-Based Used Car Price Prediction System," IEEE Access, vol. 13, 2025.

T. Li, "Used Car Price Prediction and Feature Importance Analysis Using XGBoost and SHAP," HBEM, vol. 48, 2025.


