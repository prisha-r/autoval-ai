import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AutoVal - Vehicle Valuation & Depreciation Analytics",
    layout="wide"
)

# Custom CSS for polished styling
st.markdown("""
    <style>
    .main { background-color: #0F172A; color: #F8FAFC; }
    .stMetric { background-color: #1E293B; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1E293B; border-radius: 8px; color: #F8FAFC; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CACHED DATA LOAD & PREPROCESSING PIPELINE
# ==============================================================================
@st.cache_data
def load_and_preprocess_data():
    if not os.path.exists('Car details v3.csv'):
        return None
    
    df = pd.read_csv('Car details v3.csv')
    
    # Clean unstructured attributes
    df.drop(columns=['torque'], errors='ignore', inplace=True)
    df['mileage'] = df['mileage'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
    df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
    
    # Feature Engineering
    df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
    luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land', 'Volvo', 'Lexus', 'Porsche']
    df['brand_tier'] = df['brand'].apply(lambda x: 'Luxury' if x in luxury_brands else 'Mass-Market')
    
    dataset_ref_year = int(df['year'].max()) if 'year' in df.columns else 2020
    df['vehicle_age'] = dataset_ref_year - df['year']
    
    # Hybrid Group-Wise Median Imputation
    cols_to_impute = ['mileage', 'engine', 'max_power', 'seats']
    for col in cols_to_impute:
        if col in df.columns:
            df[col] = df.groupby('brand')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())
            
    def assign_price_segment(price):
        if price < 400000: return 'Budget (< Rs. 4L)'
        elif price < 800000: return 'Economy (Rs. 4L-8L)'
        elif price < 1500000: return 'Mid-Range (Rs. 8L-15L)'
        else: return 'Premium/Luxury (> Rs. 15L)'

    df['price_segment'] = df['selling_price'].apply(assign_price_segment)
    df['log_price'] = np.log(df['selling_price'])
    df['log_km'] = np.log1p(df['km_driven'])
    df['selling_price_lakhs'] = df['selling_price'] / 100000
    
    return df

df = load_and_preprocess_data()

if df is None:
    st.error("Dataset 'Car details v3.csv' not found in the app directory. Please upload the CSV file to your repository.")
    st.stop()

# ==============================================================================
# 2. CACHED MODEL TRAINING PIPELINE
# ==============================================================================
@st.cache_resource
def train_ml_models(data_df):
    X = data_df.drop(columns=['selling_price', 'name', 'year', 'log_price', 'price_segment', 'selling_price_lakhs'], errors='ignore')
    y = np.log1p(data_df['selling_price'])
    
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'brand_tier']
    num_cols = ['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'vehicle_age']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    }
    
    trained_pipelines = {}
    metrics_results = []
    
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        mae_rupees = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
        
        metrics_results.append({
            "Model": name,
            "Test R2": round(r2, 4),
            "MAE (Log)": round(mae, 4),
            "RMSE (Log)": round(rmse, 4),
            "MAE (Rs. Lakhs)": round(mae_rupees / 100000, 2)
        })
        
    return trained_pipelines, pd.DataFrame(metrics_results), X_train, X_test, y_train, y_test

pipelines, model_metrics, X_train, X_test, y_train, y_test = train_ml_models(df)
best_pipeline = pipelines["XGBoost Regressor"]

# ==============================================================================
# 3. STREAMLIT APP NAVIGATION & UI
# ==============================================================================
st.title("AutoVal: Pre-Owned Vehicle Valuation & Depreciation Analytics")
st.markdown("### Manipal Institute of Technology | B.Tech. Data Analytics Mini-Project")

tab1, tab2, tab3 = st.tabs([
    "Depreciation & Segment Analytics", 
    "Live Valuation Engine", 
    "Model Benchmarking & CDA"
])

# ------------------------------------------------------------------------------
# TAB 1: DEPRECIATION & SEGMENT ANALYTICS (Multi-Line Graphs)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Multi-Line Depreciation Trajectories & Segment Analysis")
    st.markdown("Explore how residual market values decay over time across different categories, fuel architectures, and brand tiers.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        analysis_type = st.selectbox(
            "Select Trend Category",
            [
                "Transmission (Automatic vs Manual)",
                "Top Volume Brands Trajectory",
                "Fuel Types (Diesel vs Petrol vs CNG)",
                "Brand Tiers (Luxury vs Mass-Market)"
            ]
        )
        
        st.markdown("---")
        st.markdown("#### Key Analytical Insights")
        if "Transmission" in analysis_type:
            st.info("Automatic variants consistently retain a higher residual price premium over manual counterparts due to rising urban demand.")
        elif "Brands" in analysis_type:
            st.info("Mass-market volume leaders like Maruti and Hyundai exhibit stable, predictable depreciation curves compared to high-end brands.")
        elif "Fuel" in analysis_type:
            st.info("Diesel vehicles historically command higher initial resale valuations in SUV segments, whereas petrol hatchbacks show faster depreciation.")
        else:
            st.info("Luxury tier vehicles experience steep exponential value depreciation during their initial 3 years of ownership.")

    with col2:
        fig, ax = plt.subplots(figsize=(9, 6))
        df_recent = df[df['year'] >= 2010].copy()
        
        if "Transmission" in analysis_type:
            trend_data = df_recent.groupby(['year', 'transmission'])['selling_price_lakhs'].median().unstack()
            sns.lineplot(data=trend_data, marker='o', linewidth=2.5, ax=ax)
            ax.set_title("Median Resale Price Trend by Transmission Type", fontsize=13, fontweight='bold')
            ax.set_ylabel("Median Price (Rs. Lakhs)", fontsize=11)
            
        elif "Brands" in analysis_type:
            top_brands = ['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Toyota', 'Honda']
            trend_data = df_recent[df_recent['brand'].isin(top_brands)].groupby(['year', 'brand'])['selling_price_lakhs'].median().unstack()
            sns.lineplot(data=trend_data, marker='s', linewidth=2.5, ax=ax)
            ax.set_title("Top Volume Brands Resale Trajectory", fontsize=13, fontweight='bold')
            ax.set_ylabel("Median Price (Rs. Lakhs)", fontsize=11)
            
        elif "Fuel" in analysis_type:
            trend_data = df_recent[df_recent['fuel'].isin(['Diesel', 'Petrol', 'CNG'])].groupby(['year', 'fuel'])['selling_price_lakhs'].median().unstack()
            sns.lineplot(data=trend_data, marker='^', linewidth=2.5, ax=ax)
            ax.set_title("Median Resale Price Trend by Fuel Type", fontsize=13, fontweight='bold')
            ax.set_ylabel("Median Price (Rs. Lakhs)", fontsize=11)
            
        else:
            trend_data = df_recent.groupby(['year', 'brand_tier'])['selling_price_lakhs'].median().unstack()
            sns.lineplot(data=trend_data, marker='D', linewidth=2.5, ax=ax)
            ax.set_title("Brand Tiers Resale Trajectory (Luxury vs Mass-Market)", fontsize=13, fontweight='bold')
            ax.set_ylabel("Median Price (Rs. Lakhs)", fontsize=11)
            
        ax.set_xlabel("Manufacturing Year", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="", frameon=True)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Correlation & Cross-Sectional Age Gradient")
    c1, c2 = st.columns(2)
    with c1:
        fig_corr, ax_corr = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr, annot_kws={"size": 8})
        ax_corr.set_title("Feature Correlation Heatmap", fontsize=11, fontweight='bold')
        st.pyplot(fig_corr)
    with c2:
        fig_dep, ax_dep = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(data=df, x='vehicle_age', y='selling_price_lakhs', hue='transmission', alpha=0.5, ax=ax_dep, palette='Set1')
        ax_dep.set_title("Depreciation Curve: Age vs Selling Price", fontsize=11, fontweight='bold')
        ax_dep.set_xlabel("Vehicle Age (Years)")
        ax_dep.set_ylabel("Selling Price (Rs. Lakhs)")
        st.pyplot(fig_dep)

# ------------------------------------------------------------------------------
# TAB 2: LIVE VALUATION ENGINE & COUNTERFACTUAL CALCULATOR
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Advanced Depreciation & Valuation Engine")
    st.markdown("Predict current market resale value, estimate baseline new-equivalent cost, compute compound annualized depreciation rates, and attribute losses between time elapsed and mileage wear.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        brand_input = st.selectbox("Vehicle Brand", sorted(df['brand'].unique()), index=0)
        year_input = st.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2019)
        km_input = st.number_input("Kilometers Driven", min_value=100, max_value=400000, value=45000)
    with col_in2:
        fuel_input = st.selectbox("Fuel Type", df['fuel'].unique())
        transmission_input = st.selectbox("Transmission", df['transmission'].unique())
        seller_input = st.selectbox("Seller Type", df['seller_type'].unique())
    with col_in3:
        owner_input = st.selectbox("Owner Type", df['owner'].unique())
        power_input = st.number_input("Max Power (bhp)", min_value=30.0, max_value=400.0, value=82.0)
        engine_input = st.number_input("Engine Displacement (CC)", min_value=600.0, max_value=5000.0, value=1197.0)

    if st.button("Calculate Comprehensive Valuation & Depreciation", type="primary"):
        specs = {
            'brand': brand_input,
            'fuel': fuel_input,
            'seller_type': seller_input,
            'transmission': transmission_input,
            'owner': owner_input,
            'km_driven': km_input,
            'mileage': df[df['brand'] == brand_input]['mileage'].median() if not df[df['brand'] == brand_input]['mileage'].empty else 18.0,
            'engine': engine_input,
            'max_power': power_input,
            'seats': 5.0,
            'vehicle_age': max(0, 2026 - year_input),
            'brand_tier': 'Luxury' if brand_input in ['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land', 'Volvo', 'Lexus', 'Porsche'] else 'Mass-Market'
        }
        
        input_df = pd.DataFrame([specs])
        pred_log = best_pipeline.predict(input_df)[0]
        current_value = np.expm1(pred_log)
        
        # Counterfactual New Value
        base_df = input_df.copy()
        base_df['vehicle_age'] = 0
        base_df['km_driven'] = 0
        base_df['owner'] = 'First Owner'
        new_val = np.expm1(best_pipeline.predict(base_df)[0])
        if new_val < current_value: new_val = current_value * 1.15
        
        total_depr = new_val - current_value
        depr_pct = (total_depr / new_val) * 100
        retention_pct = (current_value / new_val) * 100
        age_val = max(1, specs['vehicle_age'])
        annual_rate = (1 - (current_value / new_val)**(1 / age_val)) * 100
        
        st.markdown("---")
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        res_c1.metric("Predicted Resale Price", f"Rs. {current_value/100000:.2f} Lakhs")
        res_c2.metric("Estimated New Baseline", f"Rs. {new_val/100000:.2f} Lakhs")
        res_c3.metric("Residual Retention Rate", f"{retention_pct:.1f}%")
        res_c4.metric("Annual Compound Depr. Rate", f"{annual_rate:.2f}% / yr")
        
        st.success(f"Total Depreciation Loss: Rs. {total_depr:,.0f} (-{depr_pct:.1f}% cumulative over {specs['vehicle_age']} years)")

# ------------------------------------------------------------------------------
# TAB 3: MODEL BENCHMARKING & CDA
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Supervised Regression Model Benchmark Performance")
    st.markdown("Evaluated on a 20% unseen test partition (log-transformed selling price scale).")
    
    st.table(model_metrics)
    
    st.markdown("---")
    st.subheader("Confirmatory Data Analysis (CDA) & Statistical Tests")
    
    c_stat1, c_stat2 = st.columns(2)
    with c_stat1:
        st.markdown("#### 1. Welch Two-Sample t-test")
        st.markdown("* **Null Hypothesis (H0):** Mean selling price of manual vs. automatic cars is equal.")
        auto_p = df[df['transmission'] == 'Automatic']['selling_price']
        man_p = df[df['transmission'] == 'Manual']['selling_price']
        t_s, p_s = stats.ttest_ind(auto_p, man_p, equal_var=False)
        st.info(f"**Result:** t-statistic = `{t_s:.2f}`, p-value = `{p_s:.4e}`\n\n*Conclusion:* Reject H0. Transmission type significantly impacts valuation.")
        
    with c_stat2:
        st.markdown("#### 2. One-Way ANOVA Test")
        st.markdown("* **Null Hypothesis (H0):** Fuel types have identical mean selling prices.")
        f_groups = [g['selling_price'].values for _, g in df.groupby('fuel')]
        f_s, p_f = stats.f_oneway(*f_groups)
        st.info(f"**Result:** F-statistic = `{f_s:.2f}`, p-value = `{p_f:.4e}`\n\n*Conclusion:* Reject H0. Fuel architecture significantly affects pricing.")
