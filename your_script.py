import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#安装并加载xgboost包
from xgboost import XGBClassifier
#安装并加载lightgbm包，在LightGBM提出之前，最有名的GBDT工具就是gbdtoost了，
#它是基于预排序方法的决策树算法。lightgbm是对gbdtoost的优化
from lightgbm import LGBMClassifier
#安装并加载catboost包。catboost也基于决策树。但是嵌入了自动将类别型特征处
#理为数值型特征的创新算法。
#Catboost采用排序提升的方法对抗训练集中的噪声点，从而避免梯度估计的偏
from catboost import CatBoostClassifier
from warnings import filterwarnings
# 加载模型和数据
model=joblib.load('LGBMClassifier.pkl')
X_test = pd.read_csv('X_test.csv')

# Streamlit 用户界面
st.title("Differentiating IASLC system")

# 用户输入
ITH_2D = st.number_input("ITH.2D:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
ITH_3D = st.number_input("ITH.3D:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
shape_Sphericity = st.number_input("shape_Sphericity:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
glcm_Correlation = st.number_input("glcm_Correlation:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
glcm_MCC = st.number_input("glcm_MCC:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
glszm_ZonePercentage = st.number_input("glszm_ZonePercentage:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
glrlm_RunPercentage= st.number_input("glrlm_RunPercentage:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
ngtdm_Coarseness = st.number_input("ngtdm_Coarseness:", min_value=0.0, max_value=1.0, value=0.41, step=0.01)
Size = st.number_input("Tumor size:", min_value=0.0, max_value=30.0, value=7.62, step=0.01)
Pleural_indentation = st.selectbox("Pleural indentation:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Age = st.number_input("Age:", min_value=21.0, max_value=100.0, value=64.0, step=1.0)
Location = st.selectbox("Location:", options=[1, 2, 3, 4, 5], format_func=lambda x: "RUL" if x == 1 else ("RLL" if x == 2 else ("RML" if x == 3 else ("LUL" if x == 4 else "LLL"))))
Shape = st.selectbox("Shape:", options=[0, 1], format_func=lambda x: "Regular" if x == 1 else "Irregular")
Spiculation = st.selectbox("Spiculation:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Vacuole_sign = st.selectbox("Vacuole sign:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")
Lobulation = st.selectbox("Lobulation:", options=[0, 1], format_func=lambda x: "Absent" if x == 0 else "Present")

# 特征名称
feature_names = ["ITH_2D", "ITH_3D","Size", "shape_Sphericity","glcm_Correlation","glcm_MCC", "glszm_ZonePercentage","glrlm_RunPercentage","ngtdm_Coarseness","Pleural_indentation", "Age", "Location", "Shape","Spiculation","Vacuole_sign","Lobulation"]

# 处理输入并进行预测
feature_values = [ITH_2D, ITH_3D,Size, shape_Sphericity, glcm_Correlation ,glcm_MCC ,glszm_ZonePercentage,glrlm_RunPercentage,ngtdm_Coarseness,Pleural_indentation, Age, Location, Shape,Spiculation,Vacuole_sign,Lobulation]
features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    st.write(f"**Predicted Label**: {predicted_class} (1: grade2-3, 0: grade1)")
    st.write(f"**Predicted Probability**: {predicted_proba}")
    
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"Our model indicates a low probability being pathologically identified as grade1."
            f"The model estimates the probability of grade1 as {probability:.1f}%."
            "It's important to maintain a healthy lifestyle and keep having regular check-ups."
        )
    else:
        advice = (
            f"Our model indicates a high probability  being pathologically identified as grade2-3."
            f"The model estimates the probability of grade2-3 as {probability:.1f}%."
            "Operative intervention is advised, specifically an anatomic lobectomy in conjunction with systematic lymph node dissection."
        )  # 根据预测结果生成建议
    st.write(advice)
    
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 绘图
    shap.initjs()  # 初始化 SHAP
    print('----------------------------------------')
    print(explainer_shap.type())
    shap.force_plot(explainer_shap.expected_value[predicted_class], shap_values[predicted_class], pd.DataFrame([feature_values], columns=feature_names))
    # shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # LIME 解释
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(X_test.values, feature_names=feature_names, class_names=['LPA', 'Others'], mode='classification')
    lime_exp = lime_explainer.explain_instance(features.flatten(), predict_fn=model.predict_proba)
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)
