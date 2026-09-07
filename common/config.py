

DATASET = {  # dataset_name: data_names
    'small_feature': ['AgNP', 'AutoAM', 'P3HT', 'Perovskite', 'Crossed barrel'],
    'large_feature': ['dilute_solute_diffusion', 'metallic_glass_forming', 'MOF_Td', 'polymer_Cp'],
    'campaign': ['STEAM'],          # prospective amine screening campaign
    'amine_screening': ['STEAM'],   # backward compatibility alias
}

# Map dataset_name to dat/ subdirectory name (handles aliases)
DATASET_DIR = {
    'small_feature': 'small_feature',
    'large_feature': 'large_feature',
    'campaign': 'campaign',
    'amine_screening': 'campaign',  # amine_screening data lives in dat/campaign/
}

ALIAS = {
    'AgNP': 'AgNP size/shape',
    'AutoAM': 'Extruder shape',
    'P3HT': 'P3HT-CNT\nelec. conductivity',
    'Perovskite': 'Perovskite\nstability',
    'Crossed barrel': '3D structure\ntoughness',
    
    'dilute_solute_diffusion': 'Solute $E_{diffusion}$',
    'metallic_glass_forming': 'Metal $T_{glass}$',
    'MOF_Td': 'MOF $T_{d}$',
    'polymer_Cp': 'Polymer $C_{p}$',
}


DESCRIPTOR_INDICES = {
    'AgNP': [1, 4],
    'AutoAM': [3, 4],
    'P3HT': [2],
    'Perovskite': [3],
    'Crossed barrel': [],
    
    'dilute_solute_diffusion': [17],
    'metallic_glass_forming': [17],
    'MOF_Td': [25, 8, 28, 27, 13, 22, 10],
    'polymer_Cp': [14, 22],
}


COMBINATIONS = [('ZERO', 'EI'), ('ZERO', 'UCB'), ('ZERO', 'TS'), 
                ('KAN', 'EI'), ('KAN', 'DI-EI'), ('KAN', 'DI-UCB-L'), ('KAN', 'DI-UCB-H'), ('KAN', 'DI-TS'), ('KAN', 'DI')]


FEATURE_RANGE = (0.1, 0.9)

OUTPUT_RANGE = (0.1, 0.9)

# Derived constants for stopping criterion
FEATURE_SPAN = FEATURE_RANGE[1] - FEATURE_RANGE[0]   # 0.8
Y_BAR = 0.9                                           # fraction of optimum used as threshold
Y_BAR_ = FEATURE_RANGE[0] + Y_BAR * FEATURE_SPAN     # 0.82


REG_PARAMS = {  # regressors to compare and their sparsity control parameters
    'KAN': {
        'lamb': [1e-3, 0.01, 0.1],
        'lamb_entropy': [0.01, 0.1, 1.0],
        'lamb_coef': [0.01, 0.1, 1.0],
        'lamb_coefdiff': [0.01, 0.1, 1.0],
        'pruning_th': [0.1, 0.2, 0.5],
    },
    
    'SVR': {
        'C': [0.1, 1, 10],
        'epsilon': [0.001, 0.01, 0.1],
    },
    'Lasso': {
        'alpha': [0.01, 0.1, 1.0],
        'max_iter': [5000]
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1.0],
        'max_iter': [5000]
    },
    'Elastic': {
        'alpha': [0.001, 0.01, 0.1],
        'l1_ratio': [0.1, 0.2, 0.5],
        'max_iter': [5000]
    },
    'DT': {
        'max_depth': [3, 5, 10],
        'ccp_alpha': [0.0, 1e-3, 1e-2, 1e-1], # larger → stronger post-pruning → sparser tree
    },
    'XGBoost': {
        'max_depth': [3, 5, 10],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5, 1.0],
    },
    'MLP': {
        'hidden_layer_sizes': [
            (50,), (100,),
            (50, 25), (100, 50),
            (100, 50, 25)  # optional deeper but tapered
        ],
        'alpha': [0.001, 0.01, 0.1],
        'max_iter': [5000],          # default 200
    },
    'RF': {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 10],
        'max_features': [0.3, 0.5, 1.0]
    },
}