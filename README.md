
# SwiftComply Data Validator (Upload → Edit → Validate → Download)

A minimal Streamlit app that lets you:
- **Upload** an Excel file (.xlsx) and optional **rules.yaml**
- **Edit** fields inline in a spreadsheet-like grid
- **Validate** using required fields per category (Construction, PostConstruction, Industrial, MunicipalFacilities, Outfalls)
- **Download** an updated **Cleaned CSV**, **Validation CSV**, or **Excel** (both sheets)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Usage
1. Upload your Excel workbook.
2. (Optional) Upload a custom `rules.yaml`. If omitted, defaults are used.
3. Select the sheet and the category (which determines required fields).
4. Edit cells inline; the validation table updates as you go.
5. Download the **Cleaned CSV**, **Validation CSV**, or an **Excel** with both.

## Rules format

```yaml
aliases:
  permit_npdes_id:
    - "npdes_pro_id"
    - "NPDES Number"
    - "npdes.*(permit|pro)?\\s*(id|num|numb|number)"
categories:
  Construction:
    required: ["site_name","site_type","address_street","address_city","address_state","address_zip","permit_npdes_id"]
    optional: []
  Outfalls:
    required: ["site_name","permit_npdes_id"]
    optional: []
```

You can also edit the rules within the app (expand the Rules panel) and click **Apply rules** to re-parse.
