from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="ClairData • Vive le Marché", page_icon="📊", layout="wide")

OWNER  = "regis-zang"
REPO   = "streamlit-vive-le-marche"
BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"

FACT_PARQUET_PATH   = "data/fact_vive_le_marche.parquet"
DIM_MANIFEST        = "data/dimensions_manifest.json"

CAND_YEAR_COLS  = ["ano", "AN", "year", "ANO"]
CAND_GS_COLS    = ["GS"]
CAND_NA5_COLS   = ["NA5"]
CAND_NA10_COLS  = ["NA10"]
CAND_NA88_COLS  = ["NA88"]
CAND_REGLT_COLS = ["REGLT", "REG_LT", "REGION_LT"]

LANG_OPTIONS = {"Português": "pt", "English": "en", "Français": "fr"}

def _get(url: str, stream: bool = False):
    r = requests.get(url, stream=stream, timeout=60)
    r.raise_for_status()
    return r

def raw_url(path: str) -> str:
    norm = str(path).replace("\\", "/").lstrip("/")
    return f"{RAW_BASE}/{norm}"

def first_present(df_cols: pd.Index, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df_cols:
            return c
    lower = {c.lower(): c for c in df_cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def get_lang_col(df: pd.DataFrame, base_col: Optional[str], lang_code: str) -> Optional[str]:
    if base_col is None:
        return None
    candidates = [f"{base_col}_{lang_code}", f"{base_col}_{lang_code.upper()}"]
    for c in candidates:
        if c in df.columns:
            return c
    if base_col in df.columns:
        return base_col
    suffixes = [f"_{lang_code}", f"_{lang_code.upper()}"]
    for c in df.columns:
        for s in suffixes:
            if c.lower() == (base_col.lower() + s):
                return c
    return None

def guess_key_and_label(dim_df: pd.DataFrame, lang_code: str) -> Tuple[Optional[str], Optional[str]]:
    cols = list(dim_df.columns)
    if not cols:
        return None, None
    key = cols[0]
    base: Optional[str] = None
    for c in cols[1:]:
        if any(x in c.lower() for x in ["desc", "name", "label", "lib", "nom"]):
            base = c
            break
    if base is None and len(cols) >= 2:
        base = cols[1]
    label = get_lang_col(dim_df, base, lang_code) or base
    return key, label

@st.cache_data(ttl=3600, show_spinner=False)
def load_fact() -> pd.DataFrame:
    b = BytesIO(_get(raw_url(FACT_PARQUET_PATH), stream=True).content)
    return pd.read_parquet(b)

@st.cache_data(ttl=3600, show_spinner=False)
def load_dims_from_manifest() -> Tuple[Dict[str, pd.DataFrame], Dict]:
    manifest = _get(raw_url(DIM_MANIFEST)).json()
    dims: Dict[str, pd.DataFrame] = {}
    for t in manifest.get("tables", []):
        name = t["name"]
        pq_file = t["parquet_file"]
        b = BytesIO(_get(raw_url(pq_file), stream=True).content)
        dims[name] = pd.read_parquet(b)
    return dims, manifest

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

st.sidebar.header("Filtros")
lang_label = st.sidebar.selectbox("Idioma", list(LANG_OPTIONS.keys()), index=0)
LANG = LANG_OPTIONS[lang_label]

col_ano   = first_present(fact.columns, CAND_YEAR_COLS)
col_gs    = first_present(fact.columns, CAND_GS_COLS)
col_na5   = first_present(fact.columns, CAND_NA5_COLS)
col_na10  = first_present(fact.columns, CAND_NA10_COLS)
col_na88  = first_present(fact.columns, CAND_NA88_COLS)
col_reglt = first_present(fact.columns, CAND_REGLT_COLS)

if col_ano is None:
    st.warning("Coluna de Ano não encontrada (candidatas: ANO/AN/year). Alguns gráficos podem ficar indisponíveis.")

anos = sorted(fact[col_ano].dropna().unique().tolist()) if col_ano in fact.columns else []
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos[-3:] if anos else [])

def multiselect_for(colname: Optional[str], title: str) -> List:
    if colname and colname in fact.columns:
        vals = sorted(pd.Series(fact[colname].dropna().unique()).tolist())
        default_vals = vals if len(vals) <= 10 else vals[:10]
        return st.sidebar.multiselect(title, vals, default=default_vals)
    return []

sel_gs   = multiselect_for(col_gs,   "GS")
sel_na5  = multiselect_for(col_na5,  "NA5")
sel_na10 = multiselect_for(col_na10, "NA10")

df = fact.copy()
if col_ano and sel_anos:
    df = df[df[col_ano].isin(sel_anos)]
if col_gs and sel_gs:
    df = df[df[col_gs].isin(sel_gs)]
if col_na5 and sel_na5:
    df = df[df[col_na5].isin(sel_na5)]
if col_na10 and sel_na10:
    df = df[df[col_na10].isin(sel_na10)]

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
metric = st.sidebar.selectbox("Métrica (agregação = soma)", num_cols, index=0 if num_cols else None)
if not metric:
    st.error("Não encontrei nenhuma coluna numérica para usar como métrica.")
    st.stop()

tabs = st.tabs(["📈 Dashboard", "📋 Tabela", "🗺️ Mapas", "💡 Insights"])

with tabs[0]:
    st.subheader("Dashboard")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        if col_ano in df.columns:
            serie = df.groupby(col_ano, as_index=False)[metric].sum().sort_values(col_ano)
            st.markdown("**Histórico por Ano**")
            st.bar_chart(serie.set_index(col_ano)[metric])
        else:
            st.info("Sem coluna de Ano para gráfico histórico.")

    with c2:
        if col_gs in df.columns:
            top = df.groupby(col_gs, as_index=False)[metric].sum().sort_values(metric, ascending=False)
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
            fig, ax = plt.subplots()
            pd.Series(top.set_index("label")[metric]).plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Coluna GS não encontrada para gráfico de pizza.")

    st.markdown("---")
    if col_na88 in df.columns:
        top88 = df.groupby(col_na88, as_index=False)[metric].sum().sort_values(metric, ascending=False).head(20)
        st.markdown("**Top NA88 (maior → menor)**")
        st.bar_chart(top88.set_index(col_na88)[metric])
    else:
        st.info("Coluna NA88 não encontrada para gráfico horizontal.")

with tabs[1]:
    st.subheader("Tabela (fato filtrado)")
    st.dataframe(df, use_container_width=True, height=520)

with tabs[2]:
    st.subheader("Mapa – bolhas por REGLT")
    reglt_dim = dims.get("Dim_REGLT") or dims.get("Dim_REGLT".lower()) or dims.get("Dim_REGLT".upper())
    if reglt_dim is None or col_reglt not in df.columns:
        st.info("Para o mapa, preciso da coluna REGLT no fato e da dimensão 'Dim_REGLT' com latitude/longitude.")
    else:
        cand_lat = first_present(reglt_dim.columns, ["lat","latitude","LAT","Latitude"])
        cand_lon = first_present(reglt_dim.columns, ["lon","longitude","LON","Longitude"])
        if not (cand_lat and cand_lon):
            st.info("A dimensão 'Dim_REGLT' não tem colunas de latitude/longitude detectáveis.")
        else:
            agg = df.groupby(col_reglt, as_index=False)[metric].sum()
            key, label = guess_key_and_label(reglt_dim, LANG)
            labcol = label if (label and label in reglt_dim.columns) else None
            join_key = key if key else col_reglt
            m = agg.merge(reglt_dim, left_on=col_reglt, right_on=join_key, how="left")
            m = m.dropna(subset=[cand_lat, cand_lon])
            if m.empty:
                st.info("Sem coordenadas para plotar no mapa após o join.")
            else:
                m = m.rename(columns={cand_lat: "lat", cand_lon: "lon"})
                st.map(m[["lat","lon"]])
                show_cols = [col_reglt, metric, "lat", "lon"]
                if labcol and labcol in m.columns:
                    show_cols.insert(1, labcol)
                st.dataframe(m[show_cols].sort_values(metric, ascending=False).head(50),
                             use_container_width=True, height=380)

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
                if isinstance(v, (int, float, np.floating)):
                    st.metric(k, f"{v:,.2f}")
                else:
                    st.metric(k, f"{v}")
        st.markdown("---")
        st.write("**Top contribuintes (por NA10, se existir)**")
        if col_na10 in df.columns:
            top10 = df.groupby(col_na10, as_index=False)[metric].sum().sort_values(metric, ascending=False).head(10)
            st.dataframe(top10, use_container_width=True)
        else:
            st.info("Coluna NA10 não encontrada para ranking.")
