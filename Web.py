import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load model
model_xgb = joblib.load('XGBoost.pkl')

# 2. Configure feature names
# 特征顺序必须与模型训练时完全一致
feature_label = [
    'IC_wavelet-LHL_glcm_ClusterShade_h1',
    'PEI_DL_11',
    'VMI_wavelet-HLL_gldm_DependenceVariance_h2',
    'IC_logarithm_glszm_HighGrayLevelZoneEmphasis_h2',
    'IC_wavelet-LLL_firstorder_Kurtosis_h2',
    'Zeff_DL_19',
    'IC_wavelet-HHH_glszm_GrayLevelNonUniformityNormalized_h1',
    'VMI_DL_248',
    'VMI_wavelet-LHL_firstorder_Skewness_h1',
    'Zeff_DL_132',
    'Zeff_wavelet-HLL_glszm_ZoneEntropy_h3',
    'VMI_DL_176',
    'VMI_DL_91',
    'PEI_lbp-2D_firstorder_Variance_h3',
    'IC_DL_286',
    'Zeff_DL_4',
    'IC_DL_197',
    'Zeff_wavelet-LLL_glszm_LongRunLowGrayLevelEmphasis_h1',
    'IC_DL_341',
    'Differentiation'
]

# 3. Default values
# 最后一个 Differentiation 是临床二分类变量：
# 0 = Well/Moderate
# 1 = Poor
default_values = [
    1.02989367,
    1.681772972,
    2.107408721,
    0.5782**616,
    0.916627446,
    1.557858735,
    0.366022396,
    3.35410914,
    2.150222055,
    0.773485096,
    3.3759809,
    8.813182328,
    7.751701335,
    5.403969299,
    4.407647692,
    -1.606873453,
    -0.321764867,
    1.194331593,
    10.20636933,
    0
]

# 4. Streamlit input
st.title('Web Predictor for GR to nCRT in Patients with LARC')
st.sidebar.header('Input Features')

# Input feature form
inputs = {}

for i, feature in enumerate(feature_label):
    default_val = default_values[i]

    if feature == 'Differentiation':
        # Categorical variable
        # 0 = Well/Moderate, 1 = Poor
        options = ['Well/Moderate', 'Poor']

        if int(default_val) == 0:
            default_index = 0
        else:
            default_index = 1

        inputs[feature] = st.sidebar.radio(
            feature,
            options=options,
            index=default_index
        )

    else:
        inputs[feature] = st.sidebar.number_input(
            feature,
            min_value=-20.0,
            max_value=30.0,
            value=round(float(default_val), 8),
            format="%.8f"
        )

# 5. Map categorical variable to numerical value
diff_map = {
    'Well/Moderate': 0,
    'Poor': 1
}

inputs['Differentiation'] = diff_map[inputs['Differentiation']]

# 6. Convert input values into a Pandas DataFrame
# 注意：DataFrame 的列顺序必须与 feature_label 一致
input_df = pd.DataFrame([inputs], columns=feature_label)

# 7. Prediction button
if st.sidebar.button('Predict'):
    try:
        # Convert to XGBoost DMatrix
        input_data = xgb.DMatrix(input_df)

        # Prediction
        prediction = model_xgb.predict(input_data)[0]

        # Display prediction result
        st.subheader('Predicted probability of GR to nCRT')
        st.markdown(
            f'<span style="color:red; font-size:30px;">'
            f'Predicted probability of GR to nCRT; Predicted probability: {prediction:.8f}'
            f'</span>',
            unsafe_allow_html=True
        )

        # 8. Compute SHAP values
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(input_df)

        # 9. Display SHAP force plot
        st.subheader('SHAP Force Plot')

        shap.initjs()

        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0, :],
            feature_names=feature_label,
            matplotlib=True,
            contribution_threshold=0.3,
            show=False
        )

        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=120)
        plt.close()

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
