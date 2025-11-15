"""
Streamlit notebook runner / UI

How to use:
1. pip install streamlit nbformat
2. streamlit run streamlit_notebook_app.py
3. In the app: upload a .ipynb file (or use the example path variable below if running in same environment)
4. Use the UI to view and run cells.

Security note: This app executes arbitrary Python from the notebook. ONLY run notebooks you trust.
"""

# pip install nbformat
# pip install streamlit nbformat



import streamlit as st
import nbformat
import io
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

st.set_page_config(page_title="Notebook -> Streamlit runner", layout="wide")

st.title("Notebook → Streamlit runner")
st.markdown(
    "Upload a Jupyter notebook (`.ipynb`) and run its cells interactively. "
    "**Warning:** running notebook code executes arbitrary Python — only run notebooks you trust."
)

# Sidebar controls
st.sidebar.header("Controls")
st.sidebar.markdown("Upload a notebook, then expand code cells and Run.")
auto_show_code = st.sidebar.checkbox("Show code cells expanded by default", value=False)
run_whole = st.sidebar.button("Run entire notebook sequentially")
st.sidebar.markdown("---")

# Notebook upload
uploaded_nb = st.file_uploader("Upload a .ipynb file", type=["ipynb"])
# Optional: use a notebook path present on server when running locally (helpful for dev)
local_nb_path = st.sidebar.text_input("Or enter server-side notebook path (optional)", value="")

nb = None
if uploaded_nb is not None:
    try:
        nb = nbformat.reads(uploaded_nb.getvalue().decode("utf-8"), as_version=4)
        st.success("Notebook loaded from upload.")
    except Exception as e:
        st.error(f"Failed to read uploaded notebook: {e}")
elif local_nb_path:
    p = Path(local_nb_path)
    if p.exists():
        try:
            nb = nbformat.read(str(p), as_version=4)
            st.success(f"Notebook loaded from server path: {local_nb_path}")
        except Exception as e:
            st.error(f"Failed to read notebook at path: {e}")

if nb is None:
    st.info("Upload or provide a notebook path to get started.")
    st.stop()

# Convert cells into a safer in-memory structure
cells = [{"cell_type": c.cell_type, "source": c.source, "outputs": getattr(c, "outputs", [])} for c in nb.cells]

# Simple execution namespace and helper to capture stdout/stderr
exec_namespace: Dict[str, Any] = {}
# provide a default placeholder variable for uploaded file input in notebook code
uploaded_data = st.file_uploader("Optional: upload a data file to inject as variable `uploaded_file` (any type)", type=None)

def run_code_cell(code: str):
    """
    Executes code in exec_namespace and returns a dict with 'stdout', 'stderr', 'error' (exception str or None)
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    try:
        # If an uploaded file exists, inject it
        if uploaded_data is not None:
            exec_namespace["uploaded_file"] = uploaded_data
        exec(compile(code, "<notebook-cell>", "exec"), exec_namespace)
        err = None
    except Exception as e:
        err = str(e)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return {"stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue(), "error": err}

# Layout: two columns - notebook display and variable inspector
left_col, right_col = st.columns([3, 1])

with left_col:
    st.header("Notebook cells")
    for idx, cell in enumerate(cells):
        ct = cell["cell_type"]
        header = f"Cell {idx+1} — {ct.capitalize()}"
        if ct == "markdown":
            with st.expander(header, expanded=True):
                st.markdown(cell["source"])
        else:
            expanded = auto_show_code
            with st.expander(header, expanded=expanded):
                st.code(cell["source"], language="python")
                run_btn = st.button(f"Run cell {idx+1}", key=f"run_{idx}")
                if run_btn or run_whole:
                    result = run_code_cell(cell["source"])
                    if result["error"] is None:
                        if result["stdout"]:
                            st.text("Standard output:")
                            st.text(result["stdout"])
                        if result["stderr"]:
                            st.text("Standard error:")
                            st.text(result["stderr"])
                        st.success(f"Cell {idx+1} executed successfully.")
                    else:
                        st.error(f"Error executing cell {idx+1}: {result['error']}")
                        if result["stderr"]:
                            st.text("Captured stderr:")
                            st.text(result["stderr"])

with right_col:
    st.header("Execution namespace")
    st.markdown("Keys available after runs (non-`__`):")
    keys = [k for k in sorted(exec_namespace.keys()) if not k.startswith("__")]
    st.write(keys)

    if keys:
        sel = st.selectbox("Inspect variable", options=[""] + keys)
        if sel:
            val = exec_namespace.get(sel)
            # Try to render common types nicely
            try:
                import pandas as pd
                if isinstance(val, pd.DataFrame):
                    st.write("Pandas DataFrame preview (head):")
                    st.dataframe(val.head())
                elif isinstance(val, (list, tuple, set)):
                    st.write(f"{type(val).__name__} (length {len(val)}):")
                    st.write(val if len(str(val)) < 10000 else repr(val)[:10000] + "...")
                elif isinstance(val, dict):
                    st.write("dict (preview):")
                    st.json({k: repr(v) for k, v in list(val.items())[:100]})
                else:
                    st.write(repr(val))
            except Exception:
                st.write(repr(val))

st.markdown("---")
st.caption(
    "Notes:\n"
    "- This app executes notebook code in-process. Long-running code will block the UI.\n"
    "- If the notebook relies on specific packages, make sure to install them in the environment running Streamlit.\n"
    "- Use the server-side notebook path for local development (avoid re-uploading large notebooks)."
)


