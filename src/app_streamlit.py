import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import json

# ====================== CONFIG BÁSICA ======================
st.set_page_config(page_title="ClairData • Vive le Marché", page_icon="📊", layout="wide")

OWNER  = "regis-zang"
REPO   = "streamlit-vive-le-marche"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"

FACT_PARQUET_PATH   = "data/fact_vive_le_marche.parquet"
DIM_MANIFEST        = "data/dimensions_manifest.json"

# Colunas esperadas (nomes prováveis). O app tenta se adaptar.
CAND_YEAR_COLS = ["ano", "AN", "year", "ANO"]
CAND_GS_COLS   = ["GS"]
CAND_NA5_COLS  = ["NA5"]
CAND_NA10_COLS = ["NA10"]
CAND_NA88_COLS = ["NA88"]
CAND_REGLT_COLS= ["REGLT", "REG_LT", "REGION_LT"]

LANG_OPTIONS = {"Português": "pt", "English": "en", "Français": "fr"}

# ====================== HELPERS HTTP/CACHE ======================
def _get(url, stream=False):
    r = requests.get(url, stream=stream, timeout=60)
    r.raise_for_status()
    return r

@st.cache_data(ttl=3600, show_spinner=False)
def load_fact() -> pd.DataFrame:
    url = f"{RAW_BASE}/{FACT_PARQUET_PATH}"
    b = BytesIO(_get(url, stream=True).content)
    return pd.read_parquet(b)

@st.cache_data(ttl=3600, show_spinner=False)
def load_dims_from_manifest() -> tuple[dict, dict]:
    man_url = f"{RAW_BASE}/{DIM_MANIFEST}"
    manifest = _get(man_url).json()
    dims = {}
    for t in manifest.get("tables", []):
        name = t["name"]
        pq_file = t["parquet_file"]
        raw = f"{RAW_BASE}/{pq_file}"
        b = BytesIO(_get(raw, stream=True).content)
        dims[name] = pd.read_parquet(b)
    return dims, manifest

def first_present(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    # tentativa case-insensitive
    lower = {c.lower(): c for c in df_cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def pick_numeric_metric(df: pd.DataFrame):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def get_lang_col(df: pd.DataFrame, base_col: str, lang_code: str):
    """
    Dado um dataframe de dimensão e um "base_col" (ex.: 'GS_DESC'),
    tenta retornar a coluna com sufixo de idioma (ex.: 'GS_DESC_pt', '_en', '_fr').
    Se não existir, devolve a base_col se presente, senão None.
    """
    if base_col is None:
        return None
    candidates = [f"{base_col}_{lang_code}", f"{base_col}_{lang_code.upper()}"]
    for c in candidates:
        if c in df.columns:
            return c
    if base_col in df.columns:
        return base_col
    # varre alternativas com regex
    suffixes = [f"_{lang_code}", f"_{lang_code.upper()}"]
    for c in df.columns:
        for s in suffixes:
            if c.lower() == (base_col.lower() + s):
                return c
    return None

def guess_key_and_label(dim_df: pd.DataFrame, lang_code: str):
    """
    Heurística: supõe que a 1ª coluna seja chave (id) e a 2ª alguma descrição.
    Depois tenta achar versões com sufixo de idioma.
    """
    cols = list(dim_df.columns)
    if not cols:
        return None, None
    key = cols[0]
    # tenta achar uma coluna *_DESC ou *_NAME como base
    base = None
    for c in cols[1:]:
        if any(x in c.lower() for x in ["desc", "name", "label", "lib", "nom"]):
            base = c
            break
    if base is None and len(cols) >= 2:
        base = cols[1]
    label = get_lang_col(dim_df, base, lang_code) or base
    return key, label

def ensure_cols(df: pd.DataFrame, names: list[str]) -> list[str]:
    return [c for c in names if c in df.columns]

# ====================== CARREGAMENTO DE DADOS ======================
st.title("Vive le Marché – ClairData")

with st.expander("📦 Fontes de dados (GitHub)"):
    st.write(f"**Fato**: `{FACT_PARQUET_PATH}`")
    st.write(f"**Dimensões** via manifesto: `{DIM_MANIFEST}` (Parquet)")

try:
    fact = load_fact()
    dims, manifest = load_dims_from_manifest()
    st.success(f"Dados carregados • Fato: {len(fact):,} linhas • Dimensões: {len(dims)}")
except Exception as e:
    st.error("Falha ao carregar dados.")
    st.exception(e)
    st.stop()

# ====================== FILTROS ======================
st.sidebar.header("Filtros")

# Idioma primeiro
lang_label = st.sidebar.selectbox("Idioma", list(LANG_OPTIONS.keys()), index=0)
LANG = LANG_OPTIONS[lang_label]

# Descobrir colunas
col_ano  = first_present(fact.columns, CAND_YEAR_COLS)
col_gs   = first_present(fact.columns, CAND_GS_COLS)
col_na5  = first_present(fact.columns, CAND_NA5_COLS)
col_na10 = first_present(fact.columns, CAND_NA10_COLS)
col_na88 = first_present(fact.columns, CAND_NA88_COLS)
col_reglt= first_present(fact.columns, CAND_REGLT_COLS)

if col_ano is None:
    st.warning("Coluna de Ano não encontrada (candidatas: ANO/AN/year). Alguns gráficos podem ficar indisponíveis.")

# Filtro Ano
anos = sorted(fact[col_ano].dropna().unique().tolist()) if col_ano in fact.columns else []
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos[-3:] if anos else [])

# Filtro GS / NA5 / NA10
def multiselect_for(colname: str, title: str):
    if colname and colname in fact.columns:
        vals = sorted(pd.Series(fact[colname].dropna().unique()).tolist())
        return st.sidebar.multiselect(title, vals, default=vals if len(vals) <= 10 else vals[:10])
    return []

sel_gs   = multiselect_for(col_gs,   "GS")
sel_na5  = multiselect_for(col_na5,  "NA5")
sel_na10 = multiselect_for(col_na10, "NA10")

# Aplicar filtros
df = fact.copy()
if col_ano and sel_anos:
    df = df[df[col_ano].isin(sel_anos)]
if col_gs and sel_gs:
    df = df[df[col_gs].isin(sel_gs)]
if col_na5 and sel_na5:
    df = df[df[col_na5].isin(sel_na5)]
if col_na10 and sel_na10:
    df = df[df[col_na10].isin(sel_na10)]

# Escolha da métrica (primeira numérica por padrão)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
metric = st.sidebar.selectbox("Métrica (agregação = soma)", num_cols, index=0 if num_cols else None)
if not metric:
    st.error("Não encontrei nenhuma coluna numérica para usar como métrica.")
    st.stop()

# ====================== LAYOUT DE SEÇÕES ======================
tabs = st.tabs(["📈 Dashboard", "📋 Tabela", "🗺️ Mapas", "💡 Insights"])

# ---------------------- DASHBOARD ----------------------
with tabs[0]:
    st.subheader("Dashboard")

    c1, c2 = st.columns([1.3, 1])
    # Barras históricas por Ano
    with c1:
        if col_ano in df.columns:
            serie = df.groupby(col_ano, as_index=False)[metric].sum().sort_values(col_ano)
            st.markdown("**Histórico por Ano**")
            st.bar_chart(serie.set_index(col_ano)[metric])
        else:
            st.info("Sem coluna de Ano para gráfico histórico.")

    # Pizza por GS (distribuição)
    with c2:
        if col_gs in df.columns:
            top = df.groupby(col_gs, as_index=False)[metric].sum().sort_values(metric, ascending=False)
            # Se houver dimensão GS com rótulos por idioma, tenta enriquecer
            gs_dim = dims.get("Dim_GS") or dims.get("Dim_GS".lower())
            if isinstance(gs_dim, pd.DataFrame):
                key, label = guess_key_and_label(gs_dim, LANG)
                if key and label and key in top.columns and label in gs_dim.columns:
                    top = top.merge(gs_dim[[key, label]], left_on=col_gs, right_on=key, how="left")
                    top["label"] = top[label].fillna(top[col_gs].astype(str))
                else:
                    top["label"] = top[col_gs].astype(str)
            else:
                top["label"] = top[col_gs].astype(str)
            st.markdown("**Distribuição por GS**")
            st.pyplot(pd.Series(top.set_index("label")[metric]).plot(kind="pie", autopct="%1.1f%%").get_figure())
        else:
            st.info("Coluna GS não encontrada para gráfico de pizza.")

    # Barras horizontais de NA88
    st.markdown("---")
    if col_na88 in df.columns:
        top88 = df.groupby(col_na88, as_index=False)[metric].sum().sort_values(metric, ascending=False).head(20)
        st.markdown("**Top NA88 (maior → menor)**")
        st.bar_chart(top88.set_index(col_na88)[metric])
    else:
        st.info("Coluna NA88 não encontrada para gráfico horizontal.")

# ---------------------- TABELA ----------------------
with tabs[1]:
    st.subheader("Tabela (fato filtrado)")
    st.dataframe(df, use_container_width=True, height=520)

# ---------------------- MAPAS ----------------------
with tabs[2]:
    st.subheader("Mapa – bolhas por REGLT")

    # tenta encontrar dimensão de REGLT e colunas de coordenadas
    reglt_dim = dims.get("Dim_REGLT") or dims.get("Dim_REGLT".lower()) or dims.get("Dim_REGLT".upper())
    if reglt_dim is None or col_reglt not in df.columns:
        st.info("Para o mapa, preciso da coluna REGLT no fato e da dimensão 'Dim_REGLT' contendo latitude/longitude.")
    else:
        # achar colunas de lat/lon
        cand_lat = first_present(reglt_dim.columns, ["lat","latitude","LAT","Latitude"])
        cand_lon = first_present(reglt_dim.columns, ["lon","longitude","LON","Longitude"])
        if not (cand_lat and cand_lon):
            st.info("A dimensão 'Dim_REGLT' não tem colunas de latitude/longitude detectáveis.")
        else:
            agg = df.groupby(col_reglt, as_index=False)[metric].sum()
            key, label = guess_key_and_label(reglt_dim, LANG)
            labcol = label if label in reglt_dim.columns else None
            m = agg.merge(reglt_dim, left_on=col_reglt, right_on=key if key else col_reglt, how="left")
            m = m.dropna(subset=[cand_lat, cand_lon])
            if m.empty:
                st.info("Sem coordenadas para plotar no mapa após o join.")
            else:
                st.map(m.rename(columns={cand_lat: "lat", cand_lon: "lon"})[["lat","lon"]])

                # tabela auxiliar
                show_cols = [col_reglt, metric, "lat", "lon"]
                if labcol and labcol in m.columns: 
                    show_cols.insert(1, labcol)
                st.dataframe(m.rename(columns={cand_lat: "lat", cand_lon: "lon"})[show_cols].sort_values(metric, ascending=False).head(50),
                             use_container_width=True, height=380)

# ---------------------- INSIGHTS ----------------------
with tabs[3]:
    st.subheader("Insights estatísticos")

    if df.empty:
        st.info("Sem dados após os filtros.")
    else:
        s = df[metric].dropna()
        kpis = {
            "Registros": len(s),
            "Soma": s.sum(),
            "Média": s.mean(),
            "Mediana": s.median(),
            "Mín": s.min(),
            "Máx": s.max(),
            "Desvio Padrão": s.std(),
        }
        c1, c2, c3, c4 = st.columns(4)
        for i, (k, v) in enumerate(kpis.items()):
            with [c1, c2, c3, c4][i % 4]:
                st.metric(k, f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) else f"{v}")

        st.markdown("---")
        st.write("**Top contribuintes (por NA10, se existir)**")
        if col_na10 in df.columns:
            top10 = df.groupby(col_na10, as_index=False)[metric].sum().sort_values(metric, ascending=False).head(10)
            st.dataframe(top10, use_container_width=True)
        else:
            st.info("Coluna NA10 não encontrada para ranking.")
