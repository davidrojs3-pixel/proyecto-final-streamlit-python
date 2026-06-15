import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="App analizadora de datasets",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "AI Impact on Jobs 2030": DATA_DIR / "AI_Impact_on_Jobs_2030.csv",
    "Superstore": DATA_DIR / "sample_-_superstore.csv",
    "E-commerce Risk": DATA_DIR / "synthetic_ecommerce_order_risk_dataset.csv",
    "Teen Mental Health": DATA_DIR / "Teen_Mental_Health_Dataset.csv",
}

DATASET_DESCRIPTIONS = {
    "AI Impact on Jobs 2030": "Mercado laboral e impacto de IA en empleos, salarios, habilidades, demanda futura y riesgo de reemplazo.",
    "Superstore": "Ventas de una tienda: pedidos, clientes, regiones, categorías, ventas, descuentos y utilidad.",
    "E-commerce Risk": "Pedidos de e-commerce con variables de país, dispositivo, método de pago, entrega, devolución, fraude y etiqueta de riesgo.",
    "Teen Mental Health": "Hábitos digitales, sueño, actividad física, interacción social y variables de bienestar en adolescentes. Uso exploratorio, no diagnóstico clínico.",
}

# ============================================================
# FUNCIONES DE APOYO
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv(path_or_file) -> pd.DataFrame:
    """Carga CSV probando codificaciones comunes."""
    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path_or_file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path_or_file)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas para evitar errores por espacios o símbolos."""
    clean = df.copy()
    new_cols = []
    for col in clean.columns:
        col_clean = str(col).strip().lower()
        col_clean = re.sub(r"[^0-9a-zA-Z_]+", "_", col_clean)
        col_clean = re.sub(r"_+", "_", col_clean).strip("_")
        new_cols.append(col_clean)
    clean.columns = new_cols
    return clean


def convert_possible_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convierte columnas que parezcan fechas, sin detener la app ante errores."""
    clean = df.copy()
    converted = []
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            converted.append(col)
            continue
        if clean[col].dtype == "object" and ("date" in col or "fecha" in col):
            parsed = pd.to_datetime(clean[col], errors="coerce")
            success_rate = parsed.notna().mean()
            if success_rate >= 0.70:
                clean[col] = parsed
                converted.append(col)
    return clean, converted


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in date_cols]
    binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    categorical_cols = [
        c for c in df.columns
        if (df[c].dtype == "object" or df[c].dtype == "category" or c in binary_cols)
        and c not in date_cols
    ]
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "binary": binary_cols,
        "date": date_cols,
    }


def iqr_outlier_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((series < lower) | (series > upper)).sum()
        rows.append({
            "variable": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "limite_inferior": lower,
            "limite_superior": upper,
            "outliers": int(count),
            "% outliers": round(count / len(series) * 100, 2),
        })
    return pd.DataFrame(rows)


def apply_sidebar_filters(df: pd.DataFrame, cols: Dict[str, List[str]]) -> pd.DataFrame:
    filtered = df.copy()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filtros globales")

    if cols["categorical"]:
        cat_filter = st.sidebar.selectbox(
            "Filtrar por categoría",
            ["Sin filtro"] + cols["categorical"],
        )
        if cat_filter != "Sin filtro":
            values = sorted(filtered[cat_filter].dropna().astype(str).unique().tolist())
            selected = st.sidebar.multiselect("Valores", values, default=values[: min(5, len(values))])
            if selected:
                filtered = filtered[filtered[cat_filter].astype(str).isin(selected)]

    if cols["numeric"]:
        num_filter = st.sidebar.selectbox(
            "Filtrar por rango numérico",
            ["Sin filtro"] + cols["numeric"],
        )
        if num_filter != "Sin filtro":
            min_v = float(filtered[num_filter].min())
            max_v = float(filtered[num_filter].max())
            if min_v < max_v:
                selected_range = st.sidebar.slider(
                    f"Rango de {num_filter}",
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v),
                )
                filtered = filtered[
                    (filtered[num_filter] >= selected_range[0])
                    & (filtered[num_filter] <= selected_range[1])
                ]

    if cols["date"]:
        date_filter = st.sidebar.selectbox("Filtrar por fecha", ["Sin filtro"] + cols["date"])
        if date_filter != "Sin filtro":
            min_d = filtered[date_filter].min().date()
            max_d = filtered[date_filter].max().date()
            date_range = st.sidebar.date_input("Rango de fechas", value=(min_d, max_d))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                filtered = filtered[(filtered[date_filter] >= start) & (filtered[date_filter] <= end)]

    return filtered


def show_metrics(df: pd.DataFrame, cols: Dict[str, List[str]]) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    c3.metric("Numéricas", len(cols["numeric"]))
    c4.metric("Categóricas", len(cols["categorical"]))
    c5.metric("Fechas", len(cols["date"]))
    c6.metric("Duplicados", int(df.duplicated().sum()))


def top_categories_chart(df: pd.DataFrame, col: str, top_n: int = 15):
    data = df[col].astype(str).value_counts().head(top_n).reset_index()
    data.columns = [col, "conteo"]
    return px.bar(data, x=col, y="conteo", title=f"Top {top_n} categorías de {col}")


def simple_insights(df: pd.DataFrame, cols: Dict[str, List[str]]) -> List[str]:
    insights = []
    insights.append(f"El dataset filtrado tiene {df.shape[0]:,} filas y {df.shape[1]:,} columnas.")
    null_total = int(df.isna().sum().sum())
    insights.append(f"Se detectan {null_total:,} valores nulos en la muestra filtrada.")
    if cols["numeric"]:
        means = df[cols["numeric"]].mean(numeric_only=True).sort_values(ascending=False)
        if not means.empty:
            insights.append(f"La variable numérica con mayor promedio es '{means.index[0]}'.")
    if cols["categorical"]:
        cat = cols["categorical"][0]
        mode_value = df[cat].mode(dropna=True)
        if not mode_value.empty:
            insights.append(f"En la variable categórica '{cat}', el valor más frecuente es '{mode_value.iloc[0]}'.")
    if len(cols["numeric"]) >= 2:
        corr = df[cols["numeric"]].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, np.nan)
        stacked = corr.stack().sort_values(ascending=False)
        if not stacked.empty:
            v1, v2 = stacked.index[0]
            insights.append(f"El par con mayor correlación absoluta es '{v1}' y '{v2}'. Conviene revisarlo con scatter plot antes de concluir causalidad.")
    return insights

# ============================================================
# SIDEBAR Y HOME
# ============================================================
st.sidebar.title("📊 Proyecto Final")
section = st.sidebar.radio(
    "Navegación",
    ["Home", "Carga y perfil", "Procesamiento", "Análisis visual", "Guía de entrega"],
)

st.sidebar.markdown("---")
st.sidebar.caption("App analizadora de datasets con Streamlit")

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "converted_dates" not in st.session_state:
    st.session_state.converted_dates = []

if section == "Home":
    st.title("📊 App analizadora de datasets con Streamlit")
    st.markdown("""
    **Autor:** David Rojas Reyes  
    **Módulo:** Exploración y visualización de datos con Python  
    **Año:** 2026

    Esta aplicación permite cargar, validar, procesar y visualizar datasets de diferentes contextos:
    mercado laboral, ventas, comercio electrónico y bienestar digital.
    """)

    st.subheader("🎯 Objetivo")
    st.info(
        "Construir una herramienta exploratoria que permita revisar datasets con variables numéricas, categóricas, binarias y temporales, generando visualizaciones útiles para la toma de decisiones."
    )

    st.subheader("📁 Datasets disponibles")
    for name, desc in DATASET_DESCRIPTIONS.items():
        st.markdown(f"**{name}:** {desc}")

    st.subheader("🧰 Tecnologías utilizadas")
    st.write("Python, Pandas, NumPy, Streamlit, Plotly, Matplotlib, Seaborn y GitHub.")

    st.warning(
        "Nota de uso responsable: los resultados son exploratorios y no reemplazan validación técnica, estadística, médica, financiera ni profesional."
    )

elif section == "Carga y perfil":
    st.title("📥 Carga y perfil del dataset")

    source = st.radio("Selecciona la fuente de datos", ["Dataset del proyecto", "Cargar CSV propio"])

    if source == "Dataset del proyecto":
        selected_name = st.selectbox("Dataset", list(DATASETS.keys()))
        if st.button("Cargar dataset seleccionado"):
            try:
                df_raw = load_csv(DATASETS[selected_name])
                df = standardize_columns(df_raw)
                df, converted_dates = convert_possible_dates(df)
                st.session_state.df_raw = df_raw
                st.session_state.df = df
                st.session_state.dataset_name = selected_name
                st.session_state.converted_dates = converted_dates
                st.success(f"Dataset '{selected_name}' cargado correctamente ✅")
            except Exception as e:
                st.error(f"No se pudo cargar el dataset: {e}")
    else:
        uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = load_csv(uploaded_file)
                df = standardize_columns(df_raw)
                df, converted_dates = convert_possible_dates(df)
                st.session_state.df_raw = df_raw
                st.session_state.df = df
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.converted_dates = converted_dates
                st.success("Archivo cargado correctamente ✅")
            except Exception as e:
                st.error(f"Error al cargar archivo: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        cols = classify_columns(df)
        st.subheader(f"Dataset activo: {st.session_state.dataset_name}")
        show_metrics(df, cols)

        st.subheader("Vista previa")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Columnas y tipos")
            st.dataframe(pd.DataFrame({"columna": df.columns, "tipo": df.dtypes.astype(str)}), use_container_width=True)
        with col2:
            st.subheader("Clasificación automática")
            st.write("**Numéricas:**", cols["numeric"])
            st.write("**Categóricas:**", cols["categorical"])
            st.write("**Binarias:**", cols["binary"])
            st.write("**Fechas convertidas:**", st.session_state.converted_dates)
    else:
        st.info("Carga o selecciona un dataset para iniciar el análisis.")

elif section == "Procesamiento":
    st.title("🧹 Procesamiento de datos")
    if st.session_state.df is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga y perfil'.")
        st.stop()

    df = st.session_state.df
    cols = classify_columns(df)
    filtered_df = apply_sidebar_filters(df, cols)

    st.subheader("Resultado luego de filtros")
    show_metrics(filtered_df, classify_columns(filtered_df))

    tab1, tab2, tab3, tab4 = st.tabs(["Nulos", "Duplicados", "Fechas", "Outliers"])

    with tab1:
        nulls = filtered_df.isna().sum().reset_index()
        nulls.columns = ["columna", "nulos"]
        nulls["% nulos"] = (nulls["nulos"] / max(len(filtered_df), 1) * 100).round(2)
        st.dataframe(nulls.sort_values("nulos", ascending=False), use_container_width=True)
        if nulls["nulos"].sum() > 0:
            fig = px.bar(nulls[nulls["nulos"] > 0], x="columna", y="nulos", title="Valores nulos por columna")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No se detectan valores nulos en el dataset filtrado.")

    with tab2:
        dup_count = int(filtered_df.duplicated().sum())
        st.metric("Duplicados detectados", dup_count)
        if dup_count > 0:
            st.dataframe(filtered_df[filtered_df.duplicated(keep=False)].head(50), use_container_width=True)
        else:
            st.success("No se detectan registros duplicados.")

    with tab3:
        if cols["date"]:
            st.write("Columnas de fecha detectadas:", cols["date"])
            for dcol in cols["date"]:
                st.write(f"**{dcol}:** {filtered_df[dcol].min()} a {filtered_df[dcol].max()}")
        else:
            st.info("No se detectaron columnas de fecha convertibles automáticamente.")

    with tab4:
        if cols["numeric"]:
            outliers = iqr_outlier_summary(filtered_df, cols["numeric"])
            st.dataframe(outliers.sort_values("outliers", ascending=False), use_container_width=True)
            selected_num = st.selectbox("Ver boxplot de variable", cols["numeric"])
            fig = px.box(filtered_df, y=selected_num, title=f"Boxplot de {selected_num}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay variables numéricas para analizar outliers.")

elif section == "Análisis visual":
    st.title("📈 Análisis visual")
    if st.session_state.df is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga y perfil'.")
        st.stop()

    df = st.session_state.df
    cols = classify_columns(df)
    filtered_df = apply_sidebar_filters(df, cols)
    cols_filtered = classify_columns(filtered_df)

    st.caption(f"Dataset activo: {st.session_state.dataset_name} | Filas filtradas: {filtered_df.shape[0]:,}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resumen", "Univariado", "Bivariado", "Multivariado", "Temporal", "Insights"
    ])

    with tab1:
        show_metrics(filtered_df, cols_filtered)
        st.subheader("Resumen estadístico")
        if cols_filtered["numeric"]:
            st.dataframe(filtered_df[cols_filtered["numeric"]].describe().T, use_container_width=True)
        else:
            st.info("No hay variables numéricas para resumen estadístico.")
        if st.checkbox("Mostrar datos crudos"):
            st.dataframe(filtered_df, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribución numérica")
            if cols_filtered["numeric"]:
                num_col = st.selectbox("Variable numérica", cols_filtered["numeric"], key="uni_num")
                fig = px.histogram(filtered_df, x=num_col, nbins=30, marginal="box", title=f"Histograma de {num_col}")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Versión Seaborn/Matplotlib**")
                fig_m, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(filtered_df[num_col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribución de {num_col}")
                st.pyplot(fig_m)
            else:
                st.info("No hay variables numéricas disponibles.")
        with c2:
            st.subheader("Conteo categórico")
            if cols_filtered["categorical"]:
                cat_col = st.selectbox("Variable categórica", cols_filtered["categorical"], key="uni_cat")
                fig = top_categories_chart(filtered_df, cat_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay variables categóricas disponibles.")

    with tab3:
        st.subheader("Relación entre variables")
        if len(cols_filtered["numeric"]) >= 2:
            x_col = st.selectbox("Eje X", cols_filtered["numeric"], key="bi_x")
            y_col = st.selectbox("Eje Y", [c for c in cols_filtered["numeric"] if c != x_col], key="bi_y")
            color_col = None
            if cols_filtered["categorical"]:
                color_col = st.selectbox("Color por categoría", [None] + cols_filtered["categorical"], key="bi_color")
            fig = px.scatter(
                filtered_df,
                x=x_col,
                y=y_col,
                color=color_col,
                opacity=0.7,
                title=f"Scatter plot: {x_col} vs {y_col}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Se necesitan al menos dos variables numéricas para scatter plot.")

        if cols_filtered["numeric"] and cols_filtered["categorical"]:
            st.subheader("Boxplot por categoría")
            num_col = st.selectbox("Variable numérica", cols_filtered["numeric"], key="box_num")
            cat_col = st.selectbox("Variable categórica", cols_filtered["categorical"], key="box_cat")
            top_values = filtered_df[cat_col].astype(str).value_counts().head(12).index
            temp = filtered_df[filtered_df[cat_col].astype(str).isin(top_values)]
            fig = px.box(temp, x=cat_col, y=num_col, title=f"{num_col} por {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Matriz de correlación")
        if len(cols_filtered["numeric"]) >= 2:
            selected_nums = st.multiselect(
                "Selecciona variables numéricas",
                cols_filtered["numeric"],
                default=cols_filtered["numeric"][: min(6, len(cols_filtered["numeric"]))],
            )
            if len(selected_nums) >= 2:
                corr = filtered_df[selected_nums].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap de correlación")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Se necesitan al menos dos variables numéricas.")

        st.subheader("Barras apiladas")
        if len(cols_filtered["categorical"]) >= 2:
            cat1 = st.selectbox("Categoría eje X", cols_filtered["categorical"], key="stack_cat1")
            cat2 = st.selectbox("Categoría apilada", [c for c in cols_filtered["categorical"] if c != cat1], key="stack_cat2")
            if cols_filtered["numeric"]:
                val_col = st.selectbox("Valor numérico a agregar", cols_filtered["numeric"], key="stack_val")
                agg = st.selectbox("Agregación", ["sum", "mean", "count"], key="stack_agg")
                top_x = filtered_df[cat1].astype(str).value_counts().head(10).index
                temp = filtered_df[filtered_df[cat1].astype(str).isin(top_x)]
                if agg == "sum":
                    pivot = temp.pivot_table(index=cat1, columns=cat2, values=val_col, aggfunc="sum", fill_value=0)
                elif agg == "mean":
                    pivot = temp.pivot_table(index=cat1, columns=cat2, values=val_col, aggfunc="mean", fill_value=0)
                else:
                    pivot = temp.pivot_table(index=cat1, columns=cat2, values=val_col, aggfunc="count", fill_value=0)
                fig = px.bar(pivot, x=pivot.index, y=pivot.columns, title="Barras apiladas")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay variables numéricas para agregar.")
        else:
            st.info("Se necesitan al menos dos variables categóricas.")

    with tab5:
        st.subheader("Análisis temporal")
        if cols_filtered["date"] and cols_filtered["numeric"]:
            date_col = st.selectbox("Columna de fecha", cols_filtered["date"], key="temp_date")
            value_col = st.selectbox("Variable numérica", cols_filtered["numeric"], key="temp_value")
            freq = st.selectbox("Frecuencia", {"Diaria": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "Y"})
            agg = st.selectbox("Agregación temporal", ["sum", "mean", "count"], key="temp_agg")
            temp = filtered_df[[date_col, value_col]].dropna().sort_values(date_col).set_index(date_col)
            if agg == "sum":
                ts = temp[value_col].resample(freq).sum().reset_index()
            elif agg == "mean":
                ts = temp[value_col].resample(freq).mean().reset_index()
            else:
                ts = temp[value_col].resample(freq).count().reset_index()
            fig = px.line(ts, x=date_col, y=value_col, markers=True, title=f"Evolución de {value_col}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ts.head(30), use_container_width=True)
        else:
            st.info("Este dataset no tiene fechas detectadas o no tiene variables numéricas.")

    with tab6:
        st.subheader("Hallazgos automáticos iniciales")
        for item in simple_insights(filtered_df, cols_filtered):
            st.markdown(f"- {item}")
        st.warning(
            "Estos hallazgos son exploratorios. Antes de tomar decisiones, deben validarse con contexto de negocio y revisión técnica."
        )

elif section == "Guía de entrega":
    st.title("✅ Guía de entrega")
    st.markdown("""
    Para entregar el proyecto, prepara un PDF con:

    1. Portada: título, nombre, módulo y fecha.
    2. Link del repositorio público de GitHub.
    3. Link de la aplicación publicada en Streamlit Cloud.
    4. Comentario reflexivo: aprendizajes, dificultades y beneficios.
    5. Evidencia del formulario enviado.
    6. Capturas de cada sección de la app: Home, Carga y perfil, Procesamiento y Análisis visual.

    Antes de publicar, prueba localmente:

    ```bash
    streamlit run app.py
    ```
    """)
