import pandas as pd
import tempfile
import os

def read_excel_file(file, sheet_name=None):
    """Read Excel file from file-like object, file path, or Gradio NamedString."""
    print(f"[DEBUG] read_excel_file called with type: {type(file)}")
    print(f"[DEBUG] file attributes: {dir(file)}")
    
    # Gradio NamedString: has .name attribute (file path)
    if hasattr(file, "name"):
        print(f"[DEBUG] Using .name attribute: {file.name}")
        return pd.read_excel(file.name, sheet_name=sheet_name) if sheet_name else pd.read_excel(file.name)
    elif hasattr(file, "read"):
        print("[DEBUG] Using .read() for file-like object")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        df = pd.read_excel(tmp_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(tmp_path)
        os.unlink(tmp_path)
        return df
    elif isinstance(file, str):
        print(f"[DEBUG] Using string file path: {file}")
        return pd.read_excel(file, sheet_name=sheet_name) if sheet_name else pd.read_excel(file)
    else:
        print("[DEBUG] Unsupported file type for Excel reading.")
        raise ValueError("Unsupported file type for Excel.")
    

def execute_query(query, engine):
    # Executes the SQL query using the engine and returns the result as a DataFrame
    return pd.read_sql_query(query, engine)