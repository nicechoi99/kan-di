"""KAN-DI descriptor importance in the browser.

    pip install -r requirements-app.txt
    streamlit run app/streamlit_app.py

Upload a table of experiments (CSV or Excel), pick the response column and the descriptor columns,
and the app runs the same analysis as `analyze.py` (it calls `analyze.analyze`). Nothing is stored:
the table stays in the session and the results are offered as downloads.
"""
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

R2_WARN = 0.5  # same threshold as the warning in analyze.py's report


def read_upload(name, data):
    """Read an uploaded file (name, bytes) into a DataFrame; CSV unless the name ends in .xlsx/.xls."""
    buf = io.BytesIO(data)
    if name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(buf)
    return pd.read_csv(buf)


def numeric_columns(df):
    """Columns with at least one numeric value (candidates for target and descriptors)."""
    return [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().any()]


def default_target(columns):
    """Last column, the usual place for the response."""
    return columns[-1] if columns else None


def default_features(columns, target):
    return [c for c in columns if c != target]


def r2_warning(r2):
    """Warning text when the expression explains less than half of the variance, else None."""
    med = float(np.nanmedian(r2)) if len(r2) else float('nan')
    if not med >= R2_WARN:
        return ('The closed-form expression explains less than half of the variance (median R² = %.2f). '
                'Treat the ranking as unreliable.' % med)
    return None


def run(df, target, features, minimize, fits, log_inputs, simulate=False, seeds=3, log=print):
    """Call analyze.analyze on an in-memory table without writing files."""
    from analyze import analyze
    return analyze(df, target, features=list(features), minimize=minimize, fits=int(fits),
                   log_inputs=log_inputs, simulate=simulate, seeds=int(seeds), out=None, name='upload', log=log)


def summary_png(res):
    """The summary figure of analyze.py as PNG bytes."""
    from analyze import plot_summary
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'summary.png')
        plot_summary(res['table'], res['sim'], path)
        with open(path, 'rb') as f:
            return f.read()


def main():
    import streamlit as st

    st.set_page_config(page_title='KAN-DI descriptor importance', layout='wide')
    st.title('KAN-DI: which descriptors does the response depend on?')
    st.markdown(
        'Upload a table of experiments you have already run: one row per experiment, numeric descriptor '
        'columns and one response column. A Kolmogorov–Arnold Network is fitted to the rows, pruned to a '
        'closed-form expression, and each descriptor is scored by its average gradient energy (AGE) under '
        'that expression. This is the same analysis as `analyze.py`.')

    use_example = st.sidebar.checkbox('Use the bundled example dataset', value=False,
                                      help='Synthetic: yield depends only on temperature and amine_ratio.')
    up = st.sidebar.file_uploader('Table of experiments (CSV or Excel)', type=['csv', 'xlsx', 'xls'])
    if use_example:
        df = pd.read_csv(os.path.join(ROOT, 'examples', 'example_dataset.csv'))
    elif up is not None:
        try:
            df = read_upload(up.name, up.getvalue())
        except Exception as e:
            st.error('Could not read the file: %s' % e)
            return
    else:
        st.info('Upload a CSV or Excel file in the sidebar, or tick "Use the bundled example dataset".')
        return

    st.subheader('Data')
    st.write('%d rows, %d columns' % df.shape)
    st.dataframe(df.head(20), use_container_width=True)

    cols = numeric_columns(df)
    if len(cols) < 2:
        st.error('The table needs at least two numeric columns (descriptors and a response).')
        return
    target = st.sidebar.selectbox('Response (target) column', cols, index=cols.index(default_target(cols)))
    others = default_features(cols, target)
    features = st.sidebar.multiselect('Descriptor columns', others, default=others)
    minimize = st.sidebar.checkbox('Lower response is better (minimize)', value=False)
    fits = st.sidebar.number_input('KAN fits averaged', min_value=1, max_value=10, value=1,
                                   help='Each fit trains one more KAN with LBFGS, which adds to the run time.')
    log_inputs = st.sidebar.radio('Log-transform KAN inputs', ['auto', 'on', 'off'], index=0,
                                  help='auto fits both ways on an 80/20 split and keeps the better held-out R².')
    simulate = st.sidebar.checkbox('Also replay campaigns (KAN-DI vs ZERO-EI)', value=False,
                                   help='Uses the table as the candidate pool. Several minutes per seed.')
    seeds = st.sidebar.number_input('Replay seeds', min_value=1, max_value=20, value=3, disabled=not simulate)
    if simulate:
        st.sidebar.warning('The replay takes several minutes per seed, and its iteration counts describe '
                           'this pool, not experiments you have not run.')

    if not features:
        st.warning('Select at least one descriptor column.')
        return
    if not st.sidebar.button('Run analysis', type='primary'):
        if 'res' in st.session_state:
            show(st, st.session_state['res'])
        return

    with st.status('Fitting KANs...', expanded=True) as status:
        lines = []
        def log(msg):
            lines.append(msg)
            st.text(msg.strip('\n'))
        try:
            res = run(df, target, features, minimize, fits, log_inputs, simulate, seeds, log=log)
        except Exception as e:
            status.update(label='Failed', state='error')
            st.error(str(e))
            return
        status.update(label='Done', state='complete', expanded=False)
    st.session_state['res'] = res
    show(st, res)


def show(st, res):
    st.subheader('Descriptor importance')
    msg = r2_warning(res['r2'])
    if msg:
        st.warning(msg)
    st.write('KAN input transform: **%s**; R² of the expression on the rows: %s'
             % ('log' if res['log_inputs'] else 'none', ', '.join('%.2f' % v for v in res['r2'])))
    t = res['table'].copy()
    t['AGE share (%)'] = 100 * t['AGE_share_mean']
    st.bar_chart(t.set_index('descriptor')['AGE share (%)'], horizontal=True)
    st.dataframe(t, use_container_width=True, hide_index=True)
    if res['sim'] is not None:
        st.subheader('Campaign replay (iterations to the top 10% of the response range)')
        st.dataframe(res['sim'], use_container_width=True, hide_index=True)
    try:
        st.image(summary_png(res), caption='summary.png, as written by analyze.py', width=600)
    except Exception as e:
        st.caption('figure skipped: %s' % e)
    st.text(res['report'])
    st.download_button('descriptor_importance.csv', res['table'].to_csv(index=False), 'descriptor_importance.csv')
    st.download_button('report.txt', res['report'] + '\n', 'report.txt')


if __name__ == '__main__':
    main()
