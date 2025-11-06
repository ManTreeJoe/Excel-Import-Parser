
import io
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
import yaml
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

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

def map_headers(df: pd.DataFrame, aliases_cfg: Dict[str, List[str]]) -> Tuple[Dict[str, str], List[str]]:
    """Map column headers to canonical names using aliases. Simplified to header matching only."""
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
    validations = rules.get("validations", {})
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
        
        # Apply validation rules
        for field_name, validation in validations.items():
            # After header mapping, columns are renamed to canonical names, so check canonical name first
            # Try canonical name first (since columns are renamed after header mapping)
            if field_name in df.columns:
                col = field_name
            else:
                # Fallback to original column name from header_map
                col = header_map.get(field_name, field_name)
            
            if col in df.columns:
                val = row.get(col)
                # Check for empty strings and None values - must check pd.isna first
                if not pd.isna(val) and val is not None:
                    val_str = str(val).strip()
                    if val_str and val_str.lower() not in ["nan", "none", ""]:
                        try:
                            val_num = float(val_str)
                            rule = validation.get("rule", "")
                            if rule == "value < 0" and val_num >= 0:
                                row_errors.append(validation.get("message", f"Longitude must be negative (North America only)"))
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric values
        
        issues.append({"row_index": idx, "errors": "; ".join(row_errors)})
    return pd.DataFrame(issues)

def get_required_optional_from_first_row(df: pd.DataFrame) -> Dict[str, bool]:
    """Reads the first row to determine which columns are REQUIRED vs Optional. Returns dict mapping column_name -> is_required."""
    required_optional_map = {}
    
    if len(df) == 0:
        return required_optional_map
    
    first_row = df.iloc[0]
    for col in df.columns:
        val = first_row.get(col)
        if pd.notna(val):
            val_str = str(val).strip().upper()
            # Check if the value contains "REQUIRED" (case insensitive)
            if "REQUIRED" in val_str:
                required_optional_map[col] = True
            elif "OPTIONAL" in val_str:
                required_optional_map[col] = False
        # Default to required if not specified
        if col not in required_optional_map:
            required_optional_map[col] = True
    
    return required_optional_map

def get_cell_issues(category: str, rules: Dict[str, Any], df: pd.DataFrame, header_map: Dict[str, str]) -> Tuple[Dict[Tuple[int, str], str], Dict[Tuple[int, str], str]]:
    """Returns two dicts: required_issues and optional_issues mapping (row_index, column_name) to error message.
    Uses the first row of data to determine required vs optional fields."""
    validations = rules.get("validations", {})
    required_issues = {}
    optional_issues = {}
    
    # Get required/optional mapping from first row
    required_optional_map = get_required_optional_from_first_row(df)
    
    # Skip the first row (row 0) as it contains the metadata
    data_rows = df.iloc[1:] if len(df) > 1 else df
    
    for idx, row in data_rows.iterrows():
        # Check all columns for missing values
        for col in df.columns:
            val = row.get(col)
            is_required = required_optional_map.get(col, True)  # Default to required
            
            if pd.isna(val) or (isinstance(val, str) and (not str(val).strip() or str(val).strip().lower() in ["nan", "none", ""])):
                key = (idx, col)
                if is_required:
                    required_issues[key] = f"Missing required: {col}"
                else:
                    optional_issues[key] = f"Missing optional: {col}"
        
        # Apply validation rules
        for field_name, validation in validations.items():
            # After header mapping, columns are renamed to canonical names
            if field_name in df.columns:
                col = field_name
            else:
                col = header_map.get(field_name, field_name)
            
            if col in df.columns:
                val = row.get(col)
                if not pd.isna(val) and val is not None:
                    val_str = str(val).strip()
                    if val_str and val_str.lower() not in ["nan", "none", ""]:
                        try:
                            val_num = float(val_str)
                            rule = validation.get("rule", "")
                            if rule == "value < 0" and val_num >= 0:
                                key = (idx, col)
                                # Use the first row to determine if this field is required
                                is_required = required_optional_map.get(col, True)
                                if is_required:
                                    required_issues[key] = validation.get("message", f"Longitude must be negative (North America only)")
                                else:
                                    optional_issues[key] = validation.get("message", f"Longitude must be negative (North America only)")
                        except (ValueError, TypeError):
                            pass
    
    return required_issues, optional_issues

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
                    "site_name","site_type","address_street","address_city","address_state","address_zip","permit_npdes_id","latitude","longitude"
                ],
                "optional": []
            },
            "PostConstruction": {"required": ["site_name","permit_npdes_id","latitude","longitude"], "optional": []},
            "Industrial": {"required": ["site_name","permit_npdes_id","site_type","latitude","longitude"], "optional": []},
            "MunicipalFacilities": {"required": ["site_name","permit_npdes_id","latitude","longitude"], "optional": []},
            "Outfalls": {"required": ["site_name","permit_npdes_id","latitude","longitude"], "optional": []},
        },
        "validations": {
            "longitude": {
                "rule": "value < 0",
                "message": "Longitude must be negative (North America only - Western Hemisphere)."
            }
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
st.caption("Tip: Scroll horizontally; you can filter the table with the search box below (client-side). Cells with validation issues are highlighted in red.")
search = st.text_input("Search (client-side filter on the grid)", "")

# Use Streamlit's data editor for inline editing
# Only show a subset of columns first to keep it manageable; allow toggling full view
basic_cols = [c for c in ["site_name","site_type","permit_npdes_id","address_street","address_city","address_state","address_zip"] if c in df_cleaned.columns]
show_all = st.checkbox("Show ALL columns", value=False)
display_cols = df_cleaned.columns.tolist() if show_all or not basic_cols else (basic_cols + [c for c in df_cleaned.columns if c not in basic_cols])

df_display = df_cleaned[display_cols].copy()

# Helper function to create styled view with colors
def display_styled_view(df_to_style, required_issues_dict, optional_issues_dict):
    df_styled = df_to_style.copy()
    df_styled.insert(0, "Row #", df_styled.index + 1)
    
    if search.strip():
        mask = df_styled.apply(lambda s: s.astype(str).str.contains(search, case=False, na=False))
        df_styled = df_styled[mask.any(axis=1)]
    
    def style_row(row):
        """Style cells with issues: red for required, orange for optional."""
        styles = [''] * len(row)
        for i, col in enumerate(row.index):
            if col != "Row #":
                if (row.name, col) in required_issues_dict:
                    styles[i] = 'background-color: #b20000'  # Dark red for required
                elif (row.name, col) in optional_issues_dict:
                    styles[i] = 'background-color: #c87f13'  # Darker orange for optional
        return styles
    
    styled_df = df_styled.style.apply(style_row, axis=1)
    return styled_df, df_styled

# Initial cell issues calculation
required_issues, optional_issues = get_cell_issues(category, rules, df_cleaned, header_map)

# Helper function to create AgGrid with cell styling
def create_aggrid_with_styling(df, required_issues_dict, optional_issues_dict, editable=False, grid_id=""):
    """Create AgGrid with cell styling and scroll sync capability."""
    df_grid = df.copy()
    df_grid.insert(0, "Row #", df_grid.index + 1)
    
    if search.strip():
        mask = df_grid.apply(lambda s: s.astype(str).str.contains(search, case=False, na=False))
        df_grid = df_grid[mask.any(axis=1)]
    
    # Add hidden columns with issue flags for styling
    for col in df.columns:
        issue_col = f"__{col}_issue"
        issue_values = []
        for idx in df_grid.index:
            if idx in df.index:
                if (idx, col) in required_issues_dict:
                    issue_values.append('required')
                elif (idx, col) in optional_issues_dict:
                    issue_values.append('optional')
                else:
                    issue_values.append('')
            else:
                issue_values.append('')
        df_grid[issue_col] = issue_values
    
    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(df_grid)
    gb.configure_default_column(editable=editable, resizable=True, sortable=True, minWidth=100)
    gb.configure_column('Row #', width=80, pinned='left', editable=False)
    
    # Hide issue flag columns
    for col in df_grid.columns:
        if col.startswith('__') and col.endswith('_issue'):
            gb.configure_column(col, hide=True)
    
    # Create cell style JavaScript function
    cell_style_js = JsCode("""
    function(params) {
        if (!params.data || params.colDef.field === 'Row #' || params.colDef.field.startsWith('__')) {
            return {'border': '1px solid #666666'};
        }
        var issueCol = '__' + params.colDef.field + '_issue';
        var issueType = params.data[issueCol];
        var baseStyle = {'border': '1px solid #666666'};
        if (issueType === 'required') {
            baseStyle['backgroundColor'] = '#b20000';
            baseStyle['color'] = 'white';
        } else if (issueType === 'optional') {
            baseStyle['backgroundColor'] = '#c87f13';
            baseStyle['color'] = 'white';
        }
        return baseStyle;
    }
    """)
    
    # Apply styling to data columns
    for col in df.columns:
        gb.configure_column(col, cellStyle=cell_style_js, minWidth=120)
    
    grid_options = gb.build()
    
    # Add global styling for cell borders
    grid_options['defaultColDef'] = grid_options.get('defaultColDef', {})
    grid_options['defaultColDef']['cellStyle'] = {'border': '1px solid #666666'}
    
    # Add scroll synchronization JavaScript
    if grid_id:
        other_grid_id = 'edit_grid' if grid_id == 'view_grid' else 'view_grid'
        grid_options['onBodyScroll'] = JsCode(f"""
        function(params) {{
            // Prevent infinite scroll loop
            if (params.api.gridOptionsWrapper.gridOptions.isScrolling) {{
                return;
            }}
            try {{
                // Find the other grid's scroll container
                var allGrids = document.querySelectorAll('.ag-root-wrapper');
                var thisGrid = params.api.getGridElement();
                var otherContainer = null;
                
                for (var i = 0; i < allGrids.length; i++) {{
                    if (allGrids[i] !== thisGrid) {{
                        var viewport = allGrids[i].querySelector('.ag-body-viewport');
                        if (viewport) {{
                            otherContainer = viewport;
                            break;
                        }}
                    }}
                }}
                
                if (otherContainer) {{
                    // Mark as syncing to prevent reverse sync
                    params.api.gridOptionsWrapper.gridOptions.isScrolling = true;
                    otherContainer.scrollLeft = params.left || 0;
                    otherContainer.scrollTop = params.top || 0;
                    setTimeout(function() {{
                        params.api.gridOptionsWrapper.gridOptions.isScrolling = false;
                    }}, 50);
                }}
            }} catch(e) {{
                // Silently fail if grids aren't ready
            }}
        }}
        """)
    
    return df_grid, grid_options

# Add custom CSS for cell borders
st.markdown("""
<style>
    .ag-cell {
        border: 1px solid #666666 !important;
    }
    .ag-header-cell {
        border: 1px solid #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# Display styled view with colors (read-only) - hidden in expander
with st.expander("📊 Color-coded view (shows validation issues) - click to expand", expanded=False):
    df_view, grid_options_view = create_aggrid_with_styling(df_display, required_issues, optional_issues, editable=False, grid_id="view_grid")
    grid_response_view = AgGrid(
        df_view,
        gridOptions=grid_options_view,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=400,
        theme='streamlit',
        key='view_grid',
        fit_columns_on_grid_load=False,
        width='100%'
    )

# Editable version with colors and scroll sync
st.markdown("### ✏️ Edit data")
st.caption("💡 **Edit cells below - cells with issues are color-coded (red = required errors, orange = optional errors).**")
df_edit, grid_options_edit = create_aggrid_with_styling(df_display, required_issues, optional_issues, editable=True, grid_id="edit_grid")
grid_response_edit = AgGrid(
    df_edit,
    gridOptions=grid_options_edit,
    update_mode=GridUpdateMode.VALUE_CHANGED,
    allow_unsafe_jscode=True,
    height=400,
    theme='streamlit',
    key='edit_grid',
    fit_columns_on_grid_load=False,
    width='100%'
)

# Get edited data from AgGrid response and merge back
edited_data = grid_response_edit.get('data')
data_changed = False
if edited_data is not None and isinstance(edited_data, pd.DataFrame) and len(edited_data) > 0:
    # Use Row # to map back to original indices (Row # is 1-indexed, so subtract 1)
    if 'Row #' in edited_data.columns:
        edited_data['__original_idx'] = edited_data['Row #'] - 1
        # Remove Row # and issue columns, keep original index mapping
        edited_clean = edited_data.drop(columns=[c for c in edited_data.columns if c == "Row #" or (c.startswith('__') and c.endswith('_issue'))])
        
        # Merge edits back into df_cleaned using the original index
        if '__original_idx' in edited_clean.columns:
            for _, row in edited_clean.iterrows():
                orig_idx = int(row['__original_idx'])
                if orig_idx in df_cleaned.index:
                    for col in edited_clean.columns:
                        if col != '__original_idx' and col in df_cleaned.columns:
                            old_val = df_cleaned.loc[orig_idx, col]
                            new_val = row[col]
                            if pd.isna(old_val) and pd.isna(new_val):
                                continue
                            if pd.isna(old_val) or pd.isna(new_val) or str(old_val) != str(new_val):
                                df_cleaned.loc[orig_idx, col] = new_val
                                data_changed = True

# Recalculate cell issues AFTER merging edits
required_issues, optional_issues = get_cell_issues(category, rules, df_cleaned, header_map)

# Update df_display with latest data
df_display = df_cleaned[display_cols].copy()


# Validate with current rules/category
val_df = validate_rows(category, rules, df_cleaned, header_map)

# Hide validation results in a collapsible section
with st.expander("📋 Validation results (click to expand)", expanded=False):
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
