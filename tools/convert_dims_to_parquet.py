#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converte dimensões CSV (três idiomas) para Parquet e gera manifest.
Uso:
    python tools/convert_dims_to_parquet.py
Requisitos:
    pandas, pyarrow, chardet (opcional), python-slugify (opcional)
"""

import json, re, hashlib, sys
from pathlib import Path
from io import StringIO
import pandas as pd

try:
    import chardet  # melhora detecção de encoding; opcional
except Exception:
    chardet = None

# === Pastas do projeto ===
ROOT = Path(__file__).resolve().parents[1]
IN_DIR  = ROOT / "data" / "dimensions"
OUT_DIR = ROOT / "data" / "dimensions_parquet"
MANIFEST = ROOT / "data" / "dimensions_manifest.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Idiomas (detecta por sufixo no nome da coluna) ===
LANG_SUFFIXES = {
    "pt": ["_pt", "_ptbr", "_br", "_por"],
    "en": ["_en", "_eng"],
    "fr": ["_fr", "_fra", "_fre"],
}
# regex compilado para pegar idioma por sufixo
LANG_RE = re.compile(r"(.+?)(" + "|".join([re.escape(s) for v in LANG_SUFFIXES.values() for s in v]) + r")$", re.IGNORECASE)

def detect_encoding(sample: bytes) -> str:
    if chardet:
        res = chardet.detect(sample)
        enc = (res.get("encoding") or "utf-8").lower()
        # normaliza BOM
        if enc in ("utf-8-sig", "utf_8_sig"):
            return "utf-8-sig"
        return enc
    return "utf-8-sig"

def sniff_delimiter(text: str) -> str:
    # tenta csv.Sniffer, senão heurística simples
    import csv
    try:
        dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        # heurística por contagem
        counts = {d: text.count(d) for d in [",",";","|","\t"]}
        return max(counts, key=counts.get)

def normalize_col(col: str) -> str:
    c = col.strip()
    c = re.sub(r"\s+", "_", c)
    c = c.replace("–", "-").replace("—","-")
    c = re.sub(r"[^\w\-\.]", "_", c, flags=re.UNICODE)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def detect_languages(cols):
    langs = set()
    for c in cols:
        m = LANG_RE.match(c.lower())
        if m:
            suf = m.group(2).lower()
            for lang, variants in LANG_SUFFIXES.items():
                if suf in variants:
                    langs.add(lang)
    return sorted(langs)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # tenta converter colunas com >80% valores numéricos
    for c in df.columns:
        s = df[c]
        if s.dtype.kind in "biufc":
            continue
        # amostra
        sample = s.dropna().astype(str).str.replace(".", "", regex=False)\
                                       .str.replace(",", ".", regex=False)\
                                       .str.replace(" ", "", regex=False)
        if len(sample) == 0:
            continue
        # porcentagem que vira número
        ok = pd.to_numeric(sample, errors="coerce").notna().mean()
        if ok >= 0.8:
            df[c] = pd.to_numeric(sample, errors="coerce")
    return df

def read_csv_safely(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    enc = detect_encoding(raw[:4000])
    text = raw.decode(enc, errors="replace")
    delim = sniff_delimiter(text)
    df = pd.read_csv(StringIO(text), sep=delim, encoding=enc, dtype=str, keep_default_na=True)
    # normaliza colunas mantendo sufixos de idioma
    df.columns = [normalize_col(c) for c in df.columns]
    # tenta tipar números com segurança
    df = coerce_numeric(df)
    return df

def write_parquet(df: pd.DataFrame, out_path: Path) -> str:
    # compressão preferida
    compression = "zstd"
    try:
        df.to_parquet(out_path, index=False, engine="pyarrow", compression=compression)
    except Exception:
        compression = "snappy"
        df.to_parquet(out_path, index=False, engine="pyarrow", compression=compression)
    return compression

def main():
    if not IN_DIR.exists():
        print(f"[ERRO] Pasta de entrada não existe: {IN_DIR}")
        sys.exit(1)

    manifest = {
        "source": str(IN_DIR.relative_to(ROOT)),
        "output": str(OUT_DIR.relative_to(ROOT)),
        "tables": []
    }

    csv_files = sorted(IN_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[AVISO] Nenhum CSV encontrado em {IN_DIR}")
    total_rows = 0

    for csv_path in csv_files:
        print(f"→ Lendo {csv_path.name} ...")
        df = read_csv_safely(csv_path)
        rows, cols = df.shape
        total_rows += rows

        # detecta idiomas pelos sufixos nos nomes das colunas
        langs = detect_languages(df.columns)

        out_path = OUT_DIR / (csv_path.stem + ".parquet")
        comp = write_parquet(df, out_path)
        md5 = md5_bytes(out_path.read_bytes())

        manifest["tables"].append({
            "name": csv_path.stem,
            "csv_file": str(csv_path.relative_to(ROOT)),
            "parquet_file": str(out_path.relative_to(ROOT)),
            "n_rows": int(rows),
            "n_cols": int(cols),
            "columns": list(df.columns),
            "languages_detected": langs,  # ex.: ["pt","en","fr"]
            "compression": comp,
            "md5": md5
        })

    manifest["total_rows"] = total_rows
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOK ✓ Parquet em {OUT_DIR}")
    print(f"OK ✓ Manifest gerado em {MANIFEST}")
    print(f"Tabelas: {len(manifest['tables'])} • Linhas totais: {total_rows:,}")

if __name__ == "__main__":
    main()
