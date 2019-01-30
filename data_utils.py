import os
import numpy as np
import pandas as pd

from sklearn import preprocessing


def load_clean_augment(csv_file, mirror=False, mean_diff=True, valence_feats=False):
    """
    Full data loading/cleaning/augmentation stack
    
    Returns
    ------
    df: pandas DataFrame with all samples and columns
    X: pandas DataFrame with all samples, only quantitative input features
    y: pandas DataFrame with only the truncated stability vectors
    """
    
    df = basic_load_and_clean(csv_file)
    
    if mirror:
        df = augment_mirror(df)
    if mean_diff:
        df = expand_mean_diff(df)
    if valence_feats:
        df = expand_valence_feats(df)
        
    df = truncate_stability(df)
    
    columnsX = [col for col in df.columns
                if ('stabilityVec' not in col and
                    col not in ['formulaA','formulaB']
                   )
               ]
    columnY = 'stabilityVec_trunc'
    
    X = df[columnsX]
    y = df[columnY]
    
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    
    return df, X, y


def basic_load_and_clean(csv_file):
    """
    Load the training data and perform basic cleaning steps
    
    Inputs
    ------
    csv_file: path to training_data.csv
    
    Outputs
    ------
    df: pandas DataFrame (samples x features)
    """
    
    df = pd.read_csv(csv_file)

    # Convert the stabilityVec to Numpy Array
    df['stabilityVec'] = df['stabilityVec'].apply(lambda s: np.array(eval(s)).astype(int))

    # Fix the 'Valance' spelling error
    df.columns = df.columns.str.replace('Valance','Valence')

    # Fix the last few features that don't start with 'formulaA/B'
    for col in df.columns:
        if col[-2:] in ['_A', '_B']:
            df = df.rename({col:'formula{}_{}'.format(col[-1], col[:-2])},
                           axis='columns')

    # Store the original X columns
    X_cols_orig = [col for col in df.columns
                   if col not in ['formulaA','formulaB','stabilityVec']]

    print('{} samples X {} features'.format(df.shape[0], len(X_cols_orig)))
    
    return df


def truncate_stability(df):
    """
    Truncate the stability vector to remove the 0 and 100% endpoints (they're always stable)
    
    Input is DataFrame, Output is DataFrame
    """
    
    df['stabilityVec_trunc'] = df['stabilityVec'].apply(lambda A: A[1:-1])
    
    print('Test truncated stability:')
    print(df['formulaA'].iloc[0], df['formulaB'].iloc[0], ':', df['stabilityVec_trunc'].iloc[0])
    
    return df


def augment_mirror(df):
    """
    Augment the dataset by mirroring all of the A/B pairs and their stability info.
    """
    
    df_mirror = pd.DataFrame()

    ### SWAP ALL A/B COLUMNS ###
    columnsA = [col for col in df.columns if 'formulaA' in col]
    
    for colA in columnsA:
        colB = colA.replace('formulaA','formulaB')
        df_mirror[colA] = df[colB].copy()
        df_mirror[colB] = df[colA].copy()

    ### FLIP PHASE VECTOR ###
    df_mirror['stabilityVec'] = pd.Series([vec[::-1] for vec in df['stabilityVec']])

    ### RETURN AUGMENTED DATASET ###
    df_aug = pd.concat([df, df_mirror])
    
    print('Test mirror:')
    print(df_aug.loc[0,['formulaA','formulaB','stabilityVec']])
    
    return df_aug


def expand_mean_diff(df):
    """
    Expand the feature set with the differences and means between A/B properties
    """

    ### CALCULATE MEANS AND DIFFS ###
    columnsA = [col for col in df.columns
                if ('formulaA' in col and
                    col not in ['formulaA','formulaB','stabilityVec']
                   )
               ]
    
    for colA in columnsA:
        
        colB = colA.replace('formulaA','formulaB')
        diff_col = colA.replace('formulaA', 'diff')
        mean_col = colA.replace('formulaA', 'mean')
        
        df[diff_col] = df[colB].copy()-df[colA].copy()
        df[mean_col] = (df[colB].copy()+df[colA].copy())/2

    print('Test diff and mean:')
    print(df.loc[0,['formulaA',
                    'formulaB',
                    'formulaA_elements_AtomicVolume',
                    'formulaB_elements_AtomicVolume',
                    'diff_elements_AtomicVolume',
                    'mean_elements_AtomicVolume',
                   ]
                ])
    return df


def expand_valence_feats(df):
    """
    Compute the sums and ratios between all of the valance electron features
    """
    
    ABBA = ['A','B','B','A'] # There was a reason to have it this way in the earlier exploratory analysis...
    fill = ['Valence','Unfilled','Valence','Unfilled']

    shell_cols = []
    for block in ['','s','p','d','f']:
        for ab,filled in zip(ABBA,fill):
            shell_cols.append('formula{}_elements_N{}{}'.format(ab,block,filled))

    ii,jj = np.triu_indices(len(shell_cols), k=1)
    
    for i,j in zip(ii,jj):
        
        coli = shell_cols[i]; colj = shell_cols[j]
        coli_parts = coli.split('_'); colj_parts = colj.split('_')
        sum_col = '_'.join(['sum', coli_parts[0], coli_parts[2], colj_parts[0], colj_parts[2]])
        ratio_col = '_'.join(['ratio', colj_parts[0], colj_parts[2], coli_parts[0], coli_parts[2]])

        df[sum_col] = df[coli].copy()+df[colj].copy()
        df[ratio_col] = df[colj].copy()/df[coli].copy()

        # Clean up divide-by-zeros by making them -1........
        df[ratio_col] = df[ratio_col].replace([np.inf, -np.inf], np.nan)
        df[ratio_col] = df[ratio_col].fillna(-1)
        
    return df


def standardized_arrays(X_df, y_df):
    """
    Standardize X and convert to array, convert y into an array
    Return: two Numpy arrays
    """
    
    std = preprocessing.StandardScaler()
    std.fit(X_df)
    X_std_array = std.transform(X_df)
    y_array = np.vstack(y_df.values)
    
    return X_std_array, y_array


def load_standardize_test_data(test_csv, X_df, mean_diff=True, valence_feats=False):
    """
    Load the test data and augment and standardize the features
    in the same way as the training data
    
    Inputs
    ------
    test_csv: path to test_data.csv
    X_df: X_df returned by load_clean_augment - TRAINING DATA
    mean_diff: if True, compute means and differences of A/B pairs
    valence_feats: compute sums and ratios of all valence shell electrons
    
    Returns
    ------
    X_test_array: Numpy array of augmented and standardized test features
    df_test_orig: Original, raw test data loaded from csv to input predictions
    """
    
    df_test_orig = pd.read_csv(test_csv)
    df = df_test_orig.copy()
    
    # Fix the 'Valance' spelling error
    df.columns = df.columns.str.replace('Valance','Valence')

    # Fix the last few features that don't start with 'formulaA/B'
    for col in df.columns:
        if col[-2:] in ['_A', '_B']:
            df = df.rename({col:'formula{}_{}'.format(col[-1], col[:-2])},
                           axis='columns')

    # Augment features if desired
    if mean_diff:
        df = expand_mean_diff(df)
    if valence_feats:
        df = expand_valence_feats(df)
    
    columnsX = [col for col in df.columns
                if ('stabilityVec' not in col and
                    col not in ['formulaA','formulaB']
                   )
               ]
    X_test_df = df[columnsX]
    print('Are columns from test same as train?', X_df.columns==X_test_df.columns)
    
    std = preprocessing.StandardScaler()
    std.fit(X_df) # Fit on training data
    X_test_array = std.transform(X_test_df) # But transform test data
    
    return X_test_array, df_test_orig