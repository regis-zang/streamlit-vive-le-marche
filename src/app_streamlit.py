# src/app_streamlit.py
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vive le Marché – Explorer", layout="wide")

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
FACT_PATH = ROOT / "data" / "fact_vive_le_marche.parquet"
DIM_DIR   = ROOT / "data" / "dimensions"
DIM_GS    = DIM_DIR / "Dim_GS.csv"

# Mapeamento de nomes de colunas das dimensões (ajuste se necessário)
DIM_COLUMNS = {
    "GS": {
        "code": "COD_GS",
        "fr":   "Desc_lbl_fr",
        "en":   "Desc_lbl_en",
        "pt":   "Desc_lbl_pt",
    }
}

# ---------- utils ----------
@st.cache_data(show_spinner=False)
def load_fact(path: Path) -> pd.DataFrame:
    """Carrega o parquet de fato e normaliza tipos essenciais."""
    df = pd.read_parquet(path)
    for c in ["Ano", "REGLT", "ZELT", "GS", "NA88"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "QTD" in df.columns:
        df["QTD"] = pd.to_numeric(df["QTD"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_dim_gs(path: Path) -> pd.DataFrame:
    """
    Lê Dim_GS.csv tentando detectar separador e encoding.
    Aceita: ',', ';', '\t', '|' e encodings ['utf-8-sig','utf-8','latin1','cp1252'].
    Garante ao final um DataFrame de strings sem NaN.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dimensão GS não encontrada: {path}")

    seps = [",", ";", "\t", "|"]
    encs = ["utf-8-sig", "utf-8", "latin1", "cp1252"]

    last_err = None
    for sep in seps:
        for enc in encs:
            try:
                df = pd.read_csv(path, dtype=str, sep=sep, encoding=enc)
                # precisa ter ao menos 2 colunas (código + 1 label)
                if df.shape[1] >= 2:
                    return df.fillna("")
            except Exception as e:
                last_err = e
                continue

    # tentativa final com engine='python'
    try:
        df = pd.read_csv(path, dtype=str, engine="python")
        return df.fillna("")
    except Exception as e:
        last_err = e

    raise ValueError(
        "Falha ao ler Dim_GS.csv. Verifique separador e encoding.\n"
        f"Tentativas: separadores {seps}, encodings {encs}.\n"
        f"Erro mais recente: {last_err}"
    )


def build_label_map(dim_df: pd.DataFrame, kind: str, lang: str) -> pd.DataFrame:
    """
    Retorna ['code','label'] para a dimensão `kind` no idioma `lang` ('fr'|'en'|'pt').
    Faz fallback para francês se a coluna do idioma não existir.
    """
    spec = DIM_COLUMNS[kind]

    # Checa presença da coluna de código
    code_col = spec["code"]
    if code_col not in dim_df.columns:
        # tenta achar algo parecido (ex.: 'cod_gs', 'CODE', etc.)
        candidates = [c for c in dim_df.columns if c.lower() in {"cod_gs", "code_gs", "code"}]
        if candidates:
            code_col = candidates[0]
        else:
            st.error(
                f"A coluna de código esperada para {kind} ('{spec['code']}') "
                f"não está em {list(dim_df.columns)}."
            )
            st.stop()

    # escolhe a coluna do idioma, com fallback para francês
    preferred = spec.get(lang, spec["fr"])
    label_col = preferred if preferred in dim_df.columns else spec["fr"]
    if label_col not in dim_df.columns:
        # tenta detectar qualquer uma das 3
        candidates = [spec["fr"], spec["en"], spec["pt"]]
        label_col = next((c for c in candidates if c in dim_df.columns), None)
        if not label_col:
            st.error(
                "Nenhuma coluna de rótulo (fr/en/pt) foi encontrada em "
                f"{list(dim_df.columns)} para a dimensão {kind}."
            )
            st.stop()

    out = dim_df[[code_col, label_col]].copy()
    out.columns = ["code", "label"]

    # se label vier vazio, usa o próprio código
    out["label"] = out["label"].where(out["label"].str.len() > 0, out["code"])
    out = out.drop_duplicates("code").sort_values("label")
    return out


# ---------- UI ----------
st.sidebar.title("Vive le Marché")
st.sidebar.caption("Filtros com suporte a rótulos multilíngues.")

# Idioma
lang = st.sidebar.radio(
    "Idioma / Language",
    options=[("fr", "🇫🇷 Français"), ("en", "🇬🇧 English (UK)"), ("pt", "🇧🇷 Português (BR)")],
    format_func=lambda x: x[1],
    index=2  # inicia PT-BR
)[0]  # pega 'fr'|'en'|'pt'

# ---------- load data ----------
if not FACT_PATH.exists():
    st.error(f"Arquivo fato não encontrado: `{FACT_PATH}`")
    st.stop()
if not DIM_GS.exists():
    st.error(f"Dimensão GS não encontrada: `{DIM_GS}`")
    st.stop()

fact = load_fact(FACT_PATH)
dim_gs_raw = load_dim_gs(DIM_GS)
dim_gs_map = build_label_map(dim_gs_raw, "GS", lang)  # ['code','label']

st.success(f"Fato: **{len(fact):,} linhas** • **{len(fact.columns)} colunas**")
st.write(f"Dimensão GS carregada: **{len(dim_gs_map):,} itens** – idioma: **{lang.upper()}**")

# ---------- filtros (Ano, GS) ----------
col_f1, col_f2 = st.sidebar.columns(2)

anos_opts = sorted(fact["Ano"].dropna().unique()) if "Ano" in fact.columns else []
anos = col_f1.multiselect("Ano", options=anos_opts, default=[])

# opções GS exibidas com label traduzido
gs_options = dim_gs_map["label"].tolist()
gs_selected_labels = col_f2.multiselect("GS", options=gs_options, default=[])

# labels -> códigos
label_to_code = dict(zip(dim_gs_map["label"], dim_gs_map["code"]))
gs_selected_codes = [label_to_code[l] for l in gs_selected_labels]

# aplica filtros
mask = pd.Series(True, index=fact.index)
if anos:
    mask &= fact["Ano"].isin(anos)
if gs_selected_codes:
    mask &= fact["GS"].isin(gs_selected_codes)

df = fact[mask].copy()

st.markdown("### Amostra da base filtrada")
st.caption(f"{len(df):,} linhas ({len(df)/len(fact):.1%} do total)")
st.dataframe(df.head(100), use_container_width=True, height=320)

# ---------- análise agregada por GS ----------
st.markdown("## Agregado de QTD por GS")

if "QTD" not in df.columns:
    st.warning("Coluna **QTD** não encontrada no fato.")
else:
    agg = df.groupby("GS", as_index=False)["QTD"].sum().sort_values("QTD", ascending=False)
    code_to_label = dict(zip(dim_gs_map["code"], dim_gs_map["label"]))
    agg["label"] = agg["GS"].map(code_to_label).fillna(agg["GS"])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Total QTD (filtro atual)", f"{agg['QTD'].sum():,}")
        st.dataframe(
            agg.rename(columns={"label": "GS (label)"}),
            use_container_width=True,
            height=420
        )

    with c2:
        fig = px.bar(
            agg.head(50),
            x="label",
            y="QTD",
            text_auto=True,
            title="Soma de QTD por GS (top 50)"
        )
        fig.update_layout(
            xaxis_title="GS",
            yaxis_title="QTD",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Notas"):
    st.write(
        "- Os rótulos de **GS** vêm de `data/dimensions/Dim_GS.csv` e mudam conforme o idioma.\n"
        "- Filtros de GS são convertidos para **códigos** GS antes de filtrar o fato.\n"
        "- O parquet lido é `data/fact_vive_le_marche.parquet`."
    )
