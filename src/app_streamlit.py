# src/app_streamlit.py
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vive le Marché – INSEE Explorer", layout="wide")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fact_vive_le_marche.parquet"


# ---------- utils ----------
@st.cache_data(show_spinner=False)
def load_data(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # normalizações leves (sem alterar seu schema)
    for col in ["Ano", "REGLT", "ZELT", "NA88"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "QTD" in df.columns:
        df["QTD"] = pd.to_numeric(df["QTD"], errors="coerce").fillna(0).astype(int)
    return df


def safe_unique(df: pd.DataFrame, col: str):
    return sorted(df[col].dropna().astype(str).unique()) if col in df.columns else []


# ---------- sidebar ----------
st.sidebar.title("Vive le Marché")
st.sidebar.caption("Interactive explorer built with Streamlit + Plotly.")
st.sidebar.write(f"**Dataset:** `{DATA_PATH.name}`")

# ---------- load ----------
if not DATA_PATH.exists():
    st.error(
        f"Arquivo não encontrado: `{DATA_PATH}`\n\n"
        "Coloque o parquet consolidado em `data/fact_vive_le_marche.parquet`."
    )
    st.stop()

df = load_data(DATA_PATH)
st.success(f"Dados carregados: **{len(df):,} linhas** • **{len(df.columns)} colunas**")

# ---------- filtros principais ----------
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
anos = col_f1.multiselect("Ano", safe_unique(df, "Ano"))
regioes = col_f2.multiselect("Région (REGLT)", safe_unique(df, "REGLT"))
zonas = col_f3.multiselect("Zone d'emploi (ZELT)", safe_unique(df, "ZELT"))
na88_sel = col_f4.multiselect("NA88 (atividade 88 postes)", safe_unique(df, "NA88"))

# aplica filtros
mask = pd.Series(True, index=df.index)
if anos:
    mask &= df["Ano"].isin(anos)
if regioes:
    mask &= df["REGLT"].isin(regioes)
if zonas:
    mask &= df["ZELT"].isin(zonas)
if na88_sel:
    mask &= df["NA88"].isin(na88_sel)

df_f = df[mask].copy()

st.write(
    f"**Após filtros:** {len(df_f):,} linhas "
    f"({len(df_f)/len(df):.1%} do total)"
)

with st.expander("Ver amostra dos dados (head)", expanded=False):
    st.dataframe(df_f.head(100), use_container_width=True, height=320)

# ---------- seção de análise ----------
st.markdown("## Análise agregada")

# escolha de dimensão para agregar
dim_default = "Ano" if "Ano" in df_f.columns else (df_f.columns[0] if len(df_f.columns) else None)
dim = st.selectbox(
    "Dimensão para agregação (soma de QTD)",
    options=[c for c in ["Ano", "REGLT", "ZELT", "NA88"] if c in df_f.columns],
    index=0 if dim_default == "Ano" else 0,
)

if "QTD" not in df_f.columns:
    st.warning("Coluna **QTD** não encontrada no dataset — nada para agregar.")
else:
    if dim is None:
        st.info("Selecione uma dimensão para agregar.")
    else:
        agg = (
            df_f.groupby(dim, dropna=False, as_index=False)["QTD"]
            .sum()
            .sort_values("QTD", ascending=False)
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Total (QTD)", f"{agg['QTD'].sum():,}")
            st.dataframe(agg, use_container_width=True, height=420)

        with c2:
            fig = px.bar(
                agg.head(50),
                x=dim,
                y="QTD",
                title=f"Soma de QTD por {dim} (top 50)",
                text_auto=True,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), xaxis_title=dim, yaxis_title="QTD")
            st.plotly_chart(fig, use_container_width=True)

# ---------- notas ----------
with st.expander("Notas técnicas"):
    st.markdown(
        """
- Fonte: INSEE (dataset consolidado utilizado pelo projeto).
- Este app lê `data/fact_vive_le_marche.parquet`.
- A métrica apresentada é a soma de **QTD** após aplicação dos filtros.
- Dicas de performance:
  - Prefira filtros mais restritivos quando o dataset for grande.
  - A leitura é **cacheada** (`@st.cache_data`) para acelerar re-execuções.
        """
    )
