
import io
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
import yaml

st.set_page_config(page_title="SwiftComply Data Validator", layout="wide")

# ---------- Utilities ----------

def normalize_header(h: str) -> str:
    if h is None:
        return ""
    s = str(h).strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s

def compile_aliases(aliases_cfg: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for canon, alist in aliases_cfg.items():
        pats = []
        for a in alist:
            # Heuristic: treat anything with regex metachars as a regex
            if any(ch in a for ch in r".*?+^$|\()[]{}\\"):
                try:
                    pats.append(re.compile(a, re.IGNORECASE))
                except re.error:
                    pass
            else:
                norm = normalize_header(a)
                if not norm:
                    continue
                token_re = r"\W*".join(map(re.escape, norm.split("_")))
                pats.append(re.compile(rf"\b{token_re}\b", re.IGNORECASE))
        compiled[canon] = pats
    return compiled

def check_data_values(df: pd.DataFrame, canon: str, patterns: List[re.Pattern]) -> str:
    """Check if column data values indicate the column type (excluding lat/long which use header matching only)."""
    # Skip data-based matching for latitude and longitude - they use header matching only
    if canon in ["latitude", "longitude"]:
        return None
    
    # Sample up to 20 non-null values from each column for better detection
    for col in df.columns:
        sample_values = df[col].dropna().head(20).astype(str).tolist()
        valid_samples = [v.strip() for v in sample_values if v.strip() and v.strip().lower() not in ["nan", "none", ""]]
        if len(valid_samples) < 2:  # Need at least 2 valid samples
            continue
        
        # Try pattern matching on the data values themselves
        matches = 0
        for val_str in valid_samples:
            for pat in patterns:
                if pat.search(val_str) or pat.search(normalize_header(val_str)):
                    matches += 1
                    break
        
        # If a significant portion of samples match (at least 30%), consider it a match
        if matches >= max(2, len(valid_samples) * 0.3):
            return col
    return None

def map_headers(df: pd.DataFrame, aliases_cfg: Dict[str, List[str]]) -> Tuple[Dict[str, str], List[str]]:
    norm_cols = [normalize_header(c) for c in df.columns]
    norm_to_actual = {normalize_header(c): c for c in df.columns}
    compiled = compile_aliases(aliases_cfg)

    mapping: Dict[str, str] = {}
    # First pass: exact normalized matches
    for canon in compiled.keys():
        if canon in norm_to_actual:
            mapping[canon] = norm_to_actual[canon]

    # Second pass: alias patterns on headers
    for canon, patterns in compiled.items():
        if canon in mapping:
            continue
        for actual in df.columns:
            header_text = str(actual)
            for pat in patterns:
                if pat.search(header_text) or pat.search(normalize_header(header_text)):
                    mapping[canon] = actual
                    break
            if canon in mapping:
                break

    # Third pass: check data values (excluding lat/long which are header-only)
    for canon, patterns in compiled.items():
        if canon in mapping:
            continue
        matched_col = check_data_values(df, canon, patterns)
        if matched_col:
            mapping[canon] = matched_col

    return mapping, norm_cols

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(axis=1, how="all").copy()
    for col in df2.select_dtypes(include=["object"]).columns:
        df2[col] = df2[col].astype(str).str.strip().replace({"nan": pd.NA})
    return df2

def apply_header_mapping(df: pd.DataFrame, header_map: Dict[str, str], keep_unmapped: bool = True) -> pd.DataFrame:
    inv_map = {v: k for k, v in header_map.items()}
    renamed = df.rename(columns=inv_map)
    if keep_unmapped:
        return renamed
    else:
        keep_cols = list(inv_map.values())
        return renamed[keep_cols]

def validate_rows(category: str, rules: Dict[str, Any], df: pd.DataFrame, header_map: Dict[str, str]) -> pd.DataFrame:
    required = rules.get("categories", {}).get(category, {}).get("required", [])
    issues = []
    for idx, row in df.iterrows():
        row_errors = []
        for r in required:
            col = header_map.get(r, r)  # if the canonical already exists
            if col in df.columns:
                val = row.get(col)
            else:
                # maybe user has already renamed via apply_header_mapping
                val = row.get(r)
            if pd.isna(val) or (isinstance(val, str) and not str(val).strip()):
                row_errors.append(f"Missing required value: {r}")
        issues.append({"row_index": idx, "errors": "; ".join(row_errors)})
    return pd.DataFrame(issues)

def default_rules() -> Dict[str, Any]:
    return {
        "aliases": {
            "site_name": ["Site Name", r"^site\s*name$"],
            "site_type": ["Site Type", r"^site\s*type$"],
            "site_id": ["Site ID", r"\bsite\b.*\bid\b"],
            "next_inspection_date": ["Next Inspection Date"],
            "site_created_on": ["Site Created On"],
            "address_street": ["Address Street", r"^address\b.*street"],
            "address_city": ["Site - Address City", "City", r"\bcity\b"],
            "address_state": ["Site - Address State", "State", r"\bstate\b"],
            "address_zip": ["Site - Address Zip", "Zip", r"\bzip\b"],
            "latitude": ["Site - Latitude", r"\blatitude?\b"],
            "longitude": ["Site - Longitude", r"\blongitude?\b"],
            "owner_name": ["Site Owner - Name"],
            "owner_company": ["Site Owner-Company"],
            "owner_phone": ["Site Owner - Phone"],
            "owner_email": ["Site Owner - Email"],
            "primary_contact_name": ["Primary Contact - Name", "Primary Contact Name"],
            "primary_contact_email": ["Primary Contact - Email", "Primary Contact Email"],
            "primary_contact_phone": ["Primary Contact : Phone", "Primary Contact Phone"],
            "permit_npdes_id": ["npdes_pro_id", "NPDES Permit ID", "NPDES Number", "NPDES Pro Numb", r"npdes.*(permit|pro)?\s*(id|num|numb|number)"],
        },
        "categories": {
            "Construction": {
                "required": [
                    "site_name","site_type","address_street","address_city","address_state","address_zip","permit_npdes_id"
                ],
                "optional": []
            },
            "PostConstruction": {"required": ["site_name","permit_npdes_id"], "optional": []},
            "Industrial": {"required": ["site_name","permit_npdes_id","site_type"], "optional": []},
            "MunicipalFacilities": {"required": ["site_name","permit_npdes_id"], "optional": []},
            "Outfalls": {"required": ["site_name","permit_npdes_id"], "optional": []},
        }
    }

def to_excel_bytes(df_dict: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=name[:31] or "Sheet1", index=False)
    buf.seek(0)
    return buf.read()

# ---------- App UI ----------

st.title("SwiftComply Data Validator (Upload • Edit • Validate • Download)")

left, right = st.columns([2,1])

with left:
    excel_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
with right:
    rules_file = st.file_uploader("(Optional) Upload rules.yaml", type=["yml","yaml"])

# Load rules (uploaded or default)
if rules_file is not None:
    try:
        rules = yaml.safe_load(rules_file.read())
    except Exception as e:
        st.error(f"Failed to read rules: {e}")
        rules = default_rules()
else:
    rules = default_rules()

# Editable rules text
with st.expander("Rules (editable YAML)"):
    rules_text = st.text_area("Edit rules and click **Apply** to re-parse", value=yaml.safe_dump(rules, sort_keys=False), height=280)
    if st.button("Apply rules"):
        try:
            rules = yaml.safe_load(rules_text) or {}
            st.success("Rules applied.")
        except Exception as e:
            st.error(f"Invalid YAML: {e}")

if excel_file is None:
    st.info("Upload an Excel file to begin. You can also load custom rules (optional).")
    st.stop()

# Read workbook and choose sheet
try:
    xls = pd.ExcelFile(excel_file)
    sheets = xls.sheet_names
except Exception as e:
    st.error(f"Could not open Excel: {e}")
    st.stop()

sheet = st.selectbox("Choose a sheet to edit/validate", options=sheets, index=min(1, len(sheets)-1))
df_raw = pd.read_excel(xls, sheet_name=sheet)
df_raw = clean_dataframe(df_raw)

# Category selection
cat_default = "Construction" if "construct" in sheet.lower() else (
    "PostConstruction" if "post" in sheet.lower() else (
    "Industrial" if "industrial" in sheet.lower() else (
    "Outfalls" if "outfall" in sheet.lower() else "MunicipalFacilities")))
category = st.selectbox("Category (validation rule set)", options=list(rules.get("categories", {}).keys()), index=list(rules.get("categories", {}).keys()).index(cat_default) if cat_default in rules.get("categories", {}) else 0)

# Header mapping & cleaned view
header_map, norm_cols = map_headers(df_raw, rules.get("aliases", {}))
df_cleaned = apply_header_mapping(df_raw, header_map, keep_unmapped=True)

st.caption("Header mapping (canonical → actual column name)")
if header_map:
    st.json(header_map)
else:
    st.write("No header matches were found with the current aliases.")

# Editable grid
st.markdown("### Edit data")
st.caption("Tip: Scroll horizontally; you can filter the table with the search box below (client-side).")
search = st.text_input("Search (client-side filter on the grid)", "")

# Use Streamlit's data editor for inline editing
# Only show a subset of columns first to keep it manageable; allow toggling full view
basic_cols = [c for c in ["site_name","site_type","permit_npdes_id","address_street","address_city","address_state","address_zip"] if c in df_cleaned.columns]
show_all = st.checkbox("Show ALL columns", value=False)
display_cols = df_cleaned.columns.tolist() if show_all or not basic_cols else (basic_cols + [c for c in df_cleaned.columns if c not in basic_cols])

df_display = df_cleaned[display_cols].copy()

if search.strip():
    mask = df_display.apply(lambda s: s.astype(str).str.contains(search, case=False, na=False))
    df_display = df_display[mask.any(axis=1)]

edited = st.data_editor(df_display, num_rows="dynamic", use_container_width=True, key="editable_grid")
# merge edits back into df_cleaned
df_cleaned.loc[edited.index, display_cols] = edited

# Validate with current rules/category
val_df = validate_rows(category, rules, df_cleaned, header_map)

st.markdown("### Validation results")
st.dataframe(val_df, use_container_width=True, height=240)

# Downloads: cleaned CSV, validation CSV, and combined Excel
colA, colB, colC = st.columns(3)
cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")
validation_csv = val_df.to_csv(index=False).encode("utf-8")
excel_bytes = to_excel_bytes({"Cleaned": df_cleaned, "Validation": val_df})

with colA:
    st.download_button("⬇️ Download CLEANED CSV", cleaned_csv, file_name=f"{category.lower()}_cleaned.csv", mime="text/csv")
with colB:
    st.download_button("⬇️ Download VALIDATION CSV", validation_csv, file_name=f"{category.lower()}_validation.csv", mime="text/csv")
with colC:
    st.download_button("⬇️ Download Excel (Cleaned + Validation)", excel_bytes, file_name=f"{category.lower()}_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("Ready. Upload a file, edit inline, and download the adjusted outputs.")
