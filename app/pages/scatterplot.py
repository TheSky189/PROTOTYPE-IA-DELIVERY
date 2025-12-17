import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.title("Predicción de Temperatura y Precipitaciones según día")

# ----------------- Subida de archivos -----------------
st.header("Subir datasets")
temp_file = st.file_uploader("CSV de Temperaturas", type=["csv"])
prec_file = st.file_uploader("CSV de Precipitaciones", type=["csv"])

# --------- FUNCIÓN PARA CALCULAR P-VALUES ---------
def corr_pvalues(df):
    df = df.dropna()
    numeric_df = df.select_dtypes(include=np.number)
    cols = numeric_df.columns
    pvals = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    for i in cols:
        for j in cols:
            if i == j:
                pvals.loc[i,j] = 0.0
            else:
                try:
                    _, pval = pearsonr(numeric_df[i], numeric_df[j])
                    pvals.loc[i,j] = pval
                except:
                    pvals.loc[i,j] = None
    return pvals

# -------------------- SI EXISTEN LOS CSV --------------------
if temp_file is not None and prec_file is not None:
    df_temp = pd.read_csv(temp_file)
    df_prec = pd.read_csv(prec_file)

    # Eliminar Desc_Mes si existe
    if "Desc_Mes" in df_temp.columns:
        df_temp = df_temp.drop(columns=["Desc_Mes"])

    if "Desc_Mes" in df_prec.columns:
        df_prec = df_prec.drop(columns=["Desc_Mes"])

    st.success("Archivos cargados correctamente.")

    st.write("### Vista previa: Temperaturas")
    st.dataframe(df_temp.head())

    st.write("### Vista previa: Precipitaciones")
    st.dataframe(df_prec.head())

    # Guardamos en session_state
    st.session_state["df_temp"] = df_temp
    st.session_state["df_prec"] = df_prec

    # Unimos columnas numéricas
    numeric_cols_temp = df_temp.select_dtypes(include=np.number).columns.tolist()
    numeric_cols_prec = df_prec.select_dtypes(include=np.number).columns.tolist()

    # ----------------- MATRIZ DE CORRELACIÓN -----------------
    st.write("## Análisis de correlación")

    st.write("### Correlación Temperatura")
    corr_temp = df_temp[numeric_cols_temp].corr()
    st.dataframe(corr_temp)

    st.write("### p-values Temperatura")
    pvals_temp = corr_pvalues(df_temp[numeric_cols_temp])
    st.dataframe(pvals_temp)

    st.write("---")

    st.write("### Correlación Precipitaciones")
    corr_prec = df_prec[numeric_cols_prec].corr()
    st.dataframe(corr_prec)

    st.write("### p-values Precipitaciones")
    pvals_prec = corr_pvalues(df_prec[numeric_cols_prec])
    st.dataframe(pvals_prec)

    # ----------------- Scatterplot -----------------
    st.write("---")
    st.write("## Scatterplot entre variables")

    all_numerics = list(set(numeric_cols_temp + numeric_cols_prec))
    col1 = st.selectbox("Variable X", all_numerics, key="scatter_x")
    col2 = st.selectbox("Variable Y", all_numerics, key="scatter_y")

    fig, ax = plt.subplots()
    ax.scatter(df_temp.get(col1, df_prec.get(col1)),
               df_temp.get(col2, df_prec.get(col2)))
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"{col1} vs {col2}")
    st.pyplot(fig)

    # ----------------- ENTRENAMIENTO DE MODELOS -----------------
    st.write("---")
    st.write("## Entrenamiento de modelos")

    # Aseguramos estructura
    required_cols = {"any", "mes", "Temperatura"}
    if not required_cols.issubset(df_temp.columns):
        st.error("El dataset de temperatura debe contener: any, mes, Temperatura")
    else:
        # Modelo Temperatura
        X_temp = df_temp[["any", "mes"]]
        y_temp = df_temp["Temperatura"]

        split = int(len(df_temp) * 0.8)
        X_train_t, X_test_t = X_temp.iloc[:split], X_temp.iloc[split:]
        y_train_t, y_test_t = y_temp.iloc[:split], y_temp.iloc[split:]

        model_temp = RandomForestRegressor()
        model_temp.fit(X_train_t, y_train_t)
        pred_test_temp = model_temp.predict(X_test_t)
        mae_temp = mean_absolute_error(y_test_t, pred_test_temp)

        st.success(f"MAE Temperatura: {mae_temp:.3f}")

    required_cols_prec = {"any", "mes", "Precipitacion"}
    if not required_cols_prec.issubset(df_prec.columns):
        st.error("El dataset de precipitaciones debe contener: any, mes, Precipitacion")
    else:
        # Modelo Precipitación
        X_prec = df_prec[["any", "mes"]]
        y_prec = df_prec["Precipitacion"]

        split = int(len(df_prec) * 0.8)
        X_train_p, X_test_p = X_prec.iloc[:split], X_prec.iloc[split:]
        y_train_p, y_test_p = y_prec.iloc[:split], y_prec.iloc[split:]

        model_prec = RandomForestRegressor()
        model_prec.fit(X_train_p, y_train_p)
        pred_test_prec = model_prec.predict(X_test_p)
        mae_prec = mean_absolute_error(y_test_p, pred_test_prec)

        st.success(f"MAE Precipitaciones: {mae_prec:.3f}")

    # ----------------- PREDICCIÓN MANUAL -----------------
    st.write("---")
    st.write("## Predicción manual según día")

    anyo = st.number_input("Año:", min_value=1900, max_value=2100, step=1)
    mes = st.number_input("Mes:", min_value=1, max_value=12, step=1)

    if st.button("Predecir"):
        input_df = pd.DataFrame({"any": [anyo], "mes": [mes]})

        temp_pred = model_temp.predict(input_df)[0]
        prec_pred = model_prec.predict(input_df)[0]

        st.write("### Resultado")
        st.success(f"Temperatura estimada: {temp_pred:.2f} ºC")
        st.success(f"Precipitación estimada: {prec_pred:.2f} mm")
