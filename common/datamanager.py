import os
import pickle
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer, LabelEncoder

from config import DATASET, DATASET_DIR
from utils import plt, cprint, datadir, imgdir, set_colorbar, get_carray, checkexists


def _resolve_dir(dataset_name):
    """Map dataset_name to actual subdirectory under dat/."""
    return DATASET_DIR.get(dataset_name, dataset_name)


def load_results(dataset_name, combinations, force=False):
    filedir = os.path.join(datadir, _resolve_dir(dataset_name), 'BO_results_' + dataset_name + '.pkl')
    
    if checkexists(filedir) and not force:
        df = pd.read_pickle(filedir)
        return df
    else:
        results = {}
        combinations = [(m.upper(), a.upper()) for m, a in combinations]
        
        data_names = DATASET[dataset_name]
        
        jobs = [(m, acquisition, data_name) for m, acquisition in combinations for data_name in data_names]
        
        tempdir = os.path.join(datadir, _resolve_dir(dataset_name), 'temp')
        files = os.listdir(tempdir)
        
        df = []
        for job in jobs:
            (m, acquisition, data_name) = job
            prefix = data_name + '_' + m + '_' + acquisition
            
            if prefix == 'dilute_solute_diffusion_KAN_DI-UCB-H':
                a = 1
            
            for f in files:
                filename = f.split('.pkl')[0]
                filename_prefix = '_'.join(filename.split('_')[:-1])
                
                if filename_prefix != prefix:
                    continue
                
                _filedir = os.path.join(tempdir, f)
                ds = pd.read_pickle(_filedir)
                df.append(ds)
        
        df = pd.concat(df)
        
        # Exclude failed cases
        indices = df['__error__'].isna()
        df = df[indices]
        
        # Save
        df.to_pickle(filedir)
        return df
    
    
def load_dataset(dataset_name, data_name=None):
    _dir = _resolve_dir(dataset_name)
    if data_name and type(data_name) == str:  # single data
        filedir = os.path.join(datadir, _dir, 'data_' + data_name + '.pkl')
        df = pd.read_pickle(filedir)
        return df
    else:
        datasets = {}
        for data_name in DATASET[dataset_name]:
            filedir = os.path.join(datadir, _dir, 'data_' + data_name + '.pkl')
            df = pd.read_pickle(filedir)
            
            df.attrs['data_name'] = data_name
            datasets[data_name] = df
        return datasets


def encode_non_numeric(df):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


def scale_data(data_name, df, scaling_methods, feature_range=(0.1, 0.9)):
    df = deepcopy(df)
    scalers = {}
    for target in ['x', 'y']:  # for inputs and objective columns, respectively
        if target == 'x':
            values = df.iloc[:,:-1].values
            method = scaling_methods[target]
        else:
            values = df.iloc[:,-1].values
            method = scaling_methods[target]
        
        if method == 'normalization':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'standardization':
            scaler = StandardScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(n_quantiles=4, output_distribution='normal', random_state=0)
        elif method == 'box-cox':
            scaler = MinMaxScaler(feature_range=feature_range)  # box-cox requires strict positivity
            values = scaler.fit_transform(values)
            scaler = PowerTransformer(method='box-cox')
        else:
            raise NotImplementedError
        
        values = np.array(values, dtype=np.float64)
        
        if target == 'x':
            svalues = scaler.fit_transform(values)
            df.iloc[:,:-1] = svalues
        else:
            svalues = scaler.fit_transform(values.reshape(-1, 1))
            df.iloc[:,-1] = svalues

        scalers[target] = scaler
    return df, scalers
    
    
def get_random_indices(y, lower=0.5, n_sample=20, seed=0, sampling_method='span'):
    np.random.seed(seed)
    n = len(y)
    
    if sampling_method == 'split':
        indices = np.argsort(-y)
        start = int(n*lower)
        indices = indices[start:]
    elif sampling_method == 'span':
        y_min, y_max = y.min(), y.max()
        y_lowerspan = y_min + (y_max - y_min) * lower
        indices = np.where(y <= y_lowerspan)[0]
    elif sampling_method == 'random':
        indices = np.arange(n)
    else:
        raise NotImplementedError
    
    indices_sample = np.random.choice(indices, n_sample, replace=False).tolist()
    return indices_sample


def gen_dataset_few_initial(X, y, indices):
    assert y.ndim == 1 or (y.ndim == 2 and y.shape[-1] == 1)
    
    _indices = np.array(range(X.shape[0]))
    indices = np.array(indices)
    indices_ = np.setdiff1d(_indices, indices)
    
    dataset = {}
    dataset['train_input'] = X[indices, :]
    dataset['test_input'] = X[indices_, :]
    
    dataset['train_label'] = y[indices]
    dataset['test_label'] = y[indices_]
    return dataset


def type_conversion(x, y):
    x = np.array(list(x.values()))
    y = y.values
    return x.reshape(-1, 1), y


def get_z_index(df):
    return np.where(df.columns == 'Z')[0][0]


def get_X(df, indices=None, tensor=True):
    if indices:
        values = np.vstack(df.loc[indices, 'X'].values)
    else:
        values = np.vstack(df.loc[:, 'X'].values)
    
    if tensor:
        return torch.tensor(values).float()
    else:
        return values


def get_Y(df, indices=None, tensor=True):
    if indices:
        values = df.loc[indices, 'Y'].values
    else:
        values = df.loc[:, 'Y'].values
    
    if tensor:
        return torch.tensor(values).float()
    else:
        return values


def get_Ymax(df, indices, tensor=True):
    values = get_Y(df, indices, tensor=tensor)
    
    if tensor:
        return torch.max(values).float()
    else:
        return np.max(values)


def get_Z(df, indices=None, tensor=True):
    if indices:
        values = np.vstack(df.iloc[indices, np.where(df.columns == 'Z')[0][0]])
    else:
        values = np.vstack(df.iloc[:, np.where(df.columns == 'Z')[0][0]])
    
    if tensor:
        return torch.tensor(values).float()
    else:
        return values


def subtract_samples(X, y, X_samples, y_samples, rtol=1e-8, atol=1e-12):
    mask = np.ones(len(X), dtype=bool)
    for xs, ys in zip(X_samples, y_samples):
        eq_x = np.all(np.isclose(X, xs, rtol=rtol, atol=atol), axis=1)
        eq_y = np.isclose(y, ys, rtol=rtol, atol=atol)
        mask &= ~(eq_x & eq_y)
    return X[mask], y[mask]


def get_subset(df, m, acquisition, data_name):
    indices = (df['m'] == m) & (df['acquisition'] == acquisition) & (df['data_name'] == data_name)
    ds = df[indices]
    ds.drop(columns=['__error__', 'error_type', 'error_msg', 'traceback', 'data_name', 'acquisition', 'm'], inplace=True)
    return ds


def dtree_regression(df, features, objective, show_just_path=False, max_depth=50, min_samples_split=3, 
                   min_samples_leaf=3, min_impurity_decrease=0.0, criterion='squared_error', random_state=0):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import accuracy_score
    
    X = np.vstack(df[features].values)
    Y = df[objective].values
    
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state, 
                                  min_samples_split=min_samples_split, 
                                  min_samples_leaf=min_samples_leaf,
                                  min_impurity_decrease=min_impurity_decrease)
    
    model = model.fit(X, Y)
    Yhat = model.predict(X)
    score = model.score(X, Y)
    
    cprint('Model score is', np.round(score, 2), color='g')
    
    indices = np.argsort(model.feature_importances_)[::-1]
    indices_top = indices[:max_depth]
    features_top = np.array(features)[indices_top].tolist()
    return model, features_top


def reduce_features(data_name, df, max_depth):
    features = list(df.columns)[:-1]
    objective = list(df.columns)[-1]
    model, features_top = dtree_regression(df, max_depth=max_depth, features=features, objective=objective)
    return df[features_top + [objective]]


def preprocessing(dataset_name, scaling_methods, model='tSNE', n_max_feature=50, inspect=False):
    """Preprocess raw CSV datasets: encode, scale, embed, and save as pickle."""
    from sklearn.manifold import TSNE
    from pacmap import PaCMAP
    from umap import UMAP
    from plotter import plot_2D_contour_simple

    for data_name in DATASET[dataset_name]:
        if data_name not in ['MOF_Td', 'polymer_Cp']:  #! temporary
            continue
        
        # Load dataset
        _filedir = os.path.join(datadir, dataset_name, data_name + '.csv')
        df = pd.read_csv(_filedir)
        
        # Encode non-numeric values
        df = encode_non_numeric(df)
        
        # Remove unnecessary features (columns that have the same value for all samples)
        df = df.loc[:, df.nunique() > 1]
        
        # Reduce # of features for large feature dataset
        if len(df.columns) > n_max_feature:
            df = reduce_features(data_name, df, max_depth=n_max_feature)
        
        # Extract features
        feature_name = list(df.columns)[:-1]
        objective_name = list(df.columns)[-1]
        
        # Average duplicate points
        ds_grouped = df.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
        df = (ds_grouped.to_frame()).reset_index()
        
        # Normalize input features and objective values
        if data_name in ['Perovskite', 'AgNP', 'dilute_solute_diffusion', 'perovskite_stability', 'metallic_glass_forming', 'polymer_Cp']:
            inversed_objective = True
        else:
            inversed_objective = False
        
        if data_name == 'Perovskite':
            df[objective_name] = np.log(df[objective_name].values/1e6)
            log_transformed = True
        else:
            log_transformed = False
            
        df, scalers = scale_data(data_name, df, scaling_methods=scaling_methods)
        
        # Set X and Y
        X = np.vstack(df.loc[:, feature_name].values)
        df['X'] = np.vsplit(X, X.shape[0])
        df['X'] = df['X'].apply(lambda x: x.flatten())
        df['Y'] = df[objective_name]
        
        # Dimension reduction for visualization
        cprint('Evaluating low dimensional representation for', data_name, color='cyan')
        if model == 'PaCMAP':
            embedder = PaCMAP(n_components=2, n_neighbors=30, random_state=0)
        elif model == 'tSNE':
            embedder = TSNE(n_components=2, perplexity=30, random_state=0)
        elif model == 'UMAP':
            embedder = UMAP(n_components=2, n_neighbors=30, random_state=0)
        else:
            raise NotImplementedError
        
        Z = embedder.fit_transform(X)
        # Z = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0) + 1e-12)  # Normalize to [0, 1] for numerical stability in interpolation
        
        if inspect:
            Y = df['Y'].values
            indices = np.isnan(Y)
            _Z = Z[~indices, :]
            _Y = Y[~indices]
            plot_2D_contour_simple(_Z, _Y, xlabel=model)
            
        df['Z'] = np.split(Z, Z.shape[0])
        df['Z'] = df['Z'].apply(lambda x: np.squeeze(x))
        df.attrs['embedder'] = model
        
        # Save
        df.attrs['data_name'] = data_name
        df.attrs['scalers'] = scalers
        df.attrs['log_transformed'] = log_transformed
        filedir = os.path.join(datadir, dataset_name, 'data_' + data_name + '.pkl')
        df.to_pickle(filedir)
    return


def extract_XY(df, col_y, cols_skip):
    X = []
    for column in df.columns:
        if column in cols_skip:
            continue
        elif column == col_y:
            Y = df[column].values
            continue
        
        x = df[column].values
        
        if not pd.api.types.is_numeric_dtype(x):
            encoder = LabelEncoder()
            x = encoder.fit_transform(x)  # ensure consistent type

        X.append(x)
    X = np.concat(X)
    
    df['X'] = np.split(X, len(df), axis=0)
    df['Y'] = np.split(Y, len(df), axis=0)
    df['Y'] = df['Y'].apply(lambda x: x.squeeze())
    return df


def gen_demo_data(n=300, biased=False):
    """Generate synthetic demo data for testing."""
    np.random.seed(0)
    x1 = np.random.uniform(0.1, 5.0, n)
    # base distribution for x2 (uniform small domain)
    x2 = np.random.uniform(0.0, 0.1, n)
    y = gamma.pdf(x1, a=2.0, scale=1.0) + np.random.randn(n) * x2

    if biased:
        # Determine top 10% threshold
        threshold = np.percentile(y, 90)
        idx_top = np.where(y >= threshold)[0]
        idx_rest = np.where(y < threshold)[0]

        n_top = int(2 * n / 3)
        n_rest = n - n_top

        # Sample indices
        top_sample = np.random.choice(idx_top, size=n_top, replace=len(idx_top) < n_top)
        rest_sample = np.random.choice(idx_rest, size=n_rest, replace=len(idx_rest) < n_rest)

        idx_selected = np.concatenate([top_sample, rest_sample])
        np.random.shuffle(idx_selected)

        # Re-sample x2 with bias toward smaller values (e.g., quadratic bias)
        # This makes the x2 distribution right-skewed (more dense near 0)
        x2_biased = np.random.beta(a=1.0, b=5.0, size=n) * 0.1  # skewed toward 0

        # Apply selection
        x1, y = x1[idx_selected], y[idx_selected]
        x2 = x2_biased

    return x1, x2, y


def gen_benchmark_functions():
    """Construct symbolic benchmark functions for Bayesian optimization."""
    import sympy as sp
    # Common symbol definitions (up to 6D)
    x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6', real=True)

    # Rastrigin (5D scalable)
    x = sp.symbols('x1 x2 x3 x4 x5', real=True)
    A = 10
    rastrigin_expr = A * 5 + sum(xi**2 - A * sp.cos(2 * sp.pi * xi) for xi in x)

    # Hartmann-6
    alpha = [1.0, 1.2, 3.0, 3.2]
    A_mat = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ]
    P = 1e-4 * sp.Matrix([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    hartmann_expr = -sum(
        alpha[k] * sp.exp(-sum(A_mat[k][j] * (sp.Symbol(f'x{j+1}') - P[k, j])**2 for j in range(6)))
        for k in range(4)
    )

    # Griewank (5D)
    sum_term = sum(xi**2 for xi in x) / 4000
    prod_term = sp.prod(sp.cos(x[i] / sp.sqrt(i + 1)) for i in range(5))
    griewank_expr = sum_term - prod_term + 1

    # Rosenbrock (5D)
    rosen_expr = sum(100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(4))

    # Ackley (5D scalable)
    a, b, c = 20, 0.2, 2 * sp.pi
    n = 5
    sum_sq = sum(xi**2 for xi in x)
    sum_cos = sum(sp.cos(c * xi) for xi in x)
    ackley_expr = -a * sp.exp(-b * sp.sqrt(sum_sq / n)) - sp.exp(sum_cos / n) + a + sp.E

    # Collect into dictionary
    benchmarks = {
        'rastrigin': {
            'expr': rastrigin_expr,
            'domain': {
                'x1': (-5.12, 5.12),
                'x2': (-5.12, 5.12),
                'x3': (-5.12, 5.12),
                'x4': (-5.12, 5.12),
                'x5': (-5.12, 5.12)
            }
        },

        'hartmann6': {
            'expr': hartmann_expr,
            'domain': {
                'x1': (0.0, 1.0),
                'x2': (0.0, 1.0),
                'x3': (0.0, 1.0),
                'x4': (0.0, 1.0),
                'x5': (0.0, 1.0),
                'x6': (0.0, 1.0)
            }
        },

        'griewank': {
            'expr': griewank_expr,
            'domain': {
                'x1': (-600.0, 600.0),
                'x2': (-600.0, 600.0),
                'x3': (-600.0, 600.0),
                'x4': (-600.0, 600.0),
                'x5': (-600.0, 600.0)
            }
        },

        'rosenbrock': {
            'expr': rosen_expr,
            'domain': {
                'x1': (-2.0, 2.0),
                'x2': (-2.0, 2.0),
                'x3': (-2.0, 2.0),
                'x4': (-2.0, 2.0),
                'x5': (-2.0, 2.0)
            }
        },

        'ackley': {
            'expr': ackley_expr,
            'domain': {
                'x1': (-32.768, 32.768),
                'x2': (-32.768, 32.768),
                'x3': (-32.768, 32.768),
                'x4': (-32.768, 32.768),
                'x5': (-32.768, 32.768)
            }
        }
    }

    return benchmarks
