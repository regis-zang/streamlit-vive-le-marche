import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# ======= CONFIG DO REPO =======
OWNER = "regis-zang"
REPO  = "streamlit-vive-le-marche"
BRANCH = "main"

RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

FACT_PARQUET_PATH = "data/fact_vive_le_marche.parquet"
DIMENSIONS_DIR = "data/dimensions"

# ======= HELPERS =======
def _get(url, stream=False):
    r = requests.get(url, stream=stream, timeout=60)
    r.raise_for_status()
    return r

@st.cache_data(ttl=3600, show_spinner=False)
def load_fact_parquet_from_github() -> pd.DataFrame:
    """Lê o fato .parquet direto do GitHub (raw) e devolve DataFrame."""
    url = f"{RAW_BASE}/{FACT_PARQUET_PATH}"
    b = BytesIO(_get(url, stream=True).content)
    df = pd.read_parquet(b)  # requer pyarrow
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def list_dimension_csvs_from_github() -> list[dict]:
    """
    Lista arquivos do diretório de dimensões via GitHub API (somente públicos).
    Retorna uma lista de dicts com 'name' e 'download_url'.
    """
    url = f"{API_BASE}/{DIMENSIONS_DIR}?ref={BRANCH}"
    data = _get(url).json()
    items = []
    for item in data:
        if item.get("type") == "file" and item.get("name", "").lower().endswith(".csv"):
            items.append({"name": item["name"], "download_url": item["download_url"]})
    return sorted(items, key=lambda d: d["name"].lower())

@st.cache_data(ttl=3600, show_spinner=False)
def load_dimensions_from_github() -> dict[str, pd.DataFrame]:
    """
    Baixa todas as dimensões .csv e retorna um dict {nome_base: DataFrame}.
    """
    dims = {}
    for it in list_dimension_csvs_from_github():
        name = it["name"].removesuffix(".csv")
        df = pd.read_csv(it["download_url"])
        dims[name] = df
    return dims

def load_all_data():
    """Empacota as leituras com mensagens amigáveis."""
    with st.spinner("Carregando fato (parquet) do GitHub..."):
        fact_df = load_fact_parquet_from_github()
    with st.spinner("Carregando dimensões (csv) do GitHub..."):
        dims = load_dimensions_from_github()
    return fact_df, dims
