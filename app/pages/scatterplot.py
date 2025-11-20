import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

st.title("Subir archivo CSV")

archivo = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

# --------- FUNCIÓN PARA CALCULAR P-VALUES ---------
def corr_pvalues(df):
    df = df.dropna()
    numeric_df = df.select_dtypes(include=np.number)  # Filtrar solo columnas numéricas
    cols = numeric_df.columns
    pvals = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)

    for i in cols:
        for j in cols:
            if i == j:
                pvals.loc[i, j] = 0.0  # correlación consigo misma
            else:
                try:
                    _, pval = pearsonr(numeric_df[i], numeric_df[j])
                    pvals.loc[i, j] = pval
                except:
                    pvals.loc[i, j] = None  # si no se puede calcular
    return pvals

if archivo is not None:
    df = pd.read_csv(archivo)
    st.success("CSV cargado correctamente.")
    
    st.write("### Vista previa del CSV")
    st.dataframe(df.head())

    # Guardar en session_state por si lo necesitas en otra página
    st.session_state["csv_data"] = df

    # Filtrar automáticamente columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 1:
        # Matriz de correlación
        st.write("### Matriz de correlación")
        corr_matrix = df[numeric_cols].corr()
        st.dataframe(corr_matrix)

        # Matriz de p-values
        st.write("### Matriz de p-values")
        pvals_matrix = corr_pvalues(df[numeric_cols])
        st.dataframe(pvals_matrix)
    else:
        st.warning("No hay suficientes columnas numéricas para calcular correlación.")

    st.write("---")
    st.write("### Scatterplot entre variables")

    col1 = st.selectbox("Variable en eje X:", numeric_cols, key="x")
    col2 = st.selectbox("Variable en eje Y:", numeric_cols, key="y")

    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2])
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Scatterplot: {col1} vs {col2}")

    st.pyplot(fig)

if st.button("Volver al inicio"):
    st.switch_page("app.py")
