import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# ============== CONFIG BÁSICA DA PÁGINA ==============
st.set_page_config(page_title="ClairData • Vive le Marché", page_icon="📊", layout="wide")

# ============== CONFIG DO REPO ==============
OWNER  = "regis-zang"
REPO   = "streamlit-vive-le-marche"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
FACT_PARQUET_PATH = "data/fact_vive_le_marche.parquet"
DIMENSIONS_DIR    = "data/dimensions"

# ============== HELPERS ==============
def _get(url, stream=False):
    r = requests.get(url, stream=stream, timeout=60)
    r.raise_for_status()
    return r

@st.cache_data(ttl=3600, show_spinner=False)
def load_fact_parquet_from_github() -> pd.DataFrame:
    url = f"{RAW_BASE}/{FACT_PARQUET_PATH}"
    b = BytesIO(_get(url, stream=True).content)
    return pd.read_parquet(b)  # requer pyarrow

@st.cache_data(ttl=3600, show_spinner=False)
def list_dimension_csvs_from_github():
    url = f"{API_BASE}/{DIMENSIONS_DIR}?ref={BRANCH}"
    data = _get(url).json()
    items = []
    for item in data:
        if item.get("type") == "file" and item.get("name", "").lower().endswith(".csv"):
            items.append({"name": item["name"], "download_url": item["download_url"]})
    return sorted(items, key=lambda d: d["name"].lower())

@st.cache_data(ttl=3600, show_spinner=False)
def load_dimensions_from_github():
    dims = {}
    for it in list_dimension_csvs_from_github():
        name = it["name"].removesuffix(".csv")
        df = pd.read_csv(it["download_url"])
        dims[name] = df
    return dims

# ============== UI ==============
st.title("Vive le Marché – ClairData")

# Mostra sempre um cabeçalho para evitar tela “em branco”
with st.expander("⚙️ Fonte dos dados (GitHub)"):
    st.write(f"**Fato**: `{FACT_PARQUET_PATH}`")
    st.write(f"**Dimensões**: pasta `{DIMENSIONS_DIR}` (CSV)")

# Carregamento com tratamento de erro VISÍVEL
col1, col2 = st.columns([1,1])
with col1:
    try:
        with st.spinner("Carregando fato (parquet)..."):
            fact_df = load_fact_parquet_from_github()
        st.success(f"Fato carregado: {len(fact_df):,} linhas × {len(fact_df.columns)} colunas")
        st.dataframe(fact_df.head(20))
    except Exception as e:
        st.error("Falha ao carregar o fato do GitHub.")
        st.exception(e)
        fact_df = None

with col2:
    try:
        with st.spinner("Carregando dimensões (csv)..."):
            dims = load_dimensions_from_github()
        st.success(f"Dimensões carregadas: {len(dims)} tabelas")
        st.write(list(dims.keys())[:30])
    except Exception as e:
        st.error("Falha ao carregar dimensões do GitHub.")
        st.exception(e)
        dims = {}

# Exemplo simples de visualização (só roda se o fato foi carregado)
if isinstance(fact_df, pd.DataFrame) and not fact_df.empty:
    st.subheader("Visão rápida")
    numeric_cols = [c for c in fact_df.columns if pd.api.types.is_numeric_dtype(fact_df[c])]
    if numeric_cols:
        colnum = st.selectbox("Escolha uma coluna numérica para histograma", numeric_cols, index=0)
        st.bar_chart(fact_df[colnum].value_counts().sort_index())
    else:
        st.info("Não encontrei colunas numéricas para exemplo de gráfico.")
else:
    st.info("Aguardando carregamento dos dados para mostrar gráficos.")
