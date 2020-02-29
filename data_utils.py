import re
import numpy as np
import pandas as pd
import pymatgen as mg

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from matminer.featurizers import composition as composition_features


def load_clean(xls_file='data/Hydrogen Storage DataBase Full.xlsx'):
    """ Load and clean the data, return dataframe """
    
    df = basic_load_and_clean(xls_file)
    df = clean_Hf(df)
    
    # Prepare for featurization
    df['composition'] = df['clean_composition'].apply(mg.Composition)

    return df


def basic_load_and_clean(xls_file):
    """
    Load the training data and perform basic cleaning steps
    """
    
    # Load in data
    df = pd.read_excel(xls_file)
    
    ### DEFINE CLEANING FUNCTIONS ###
    clean_funcs = {
                'variable_comp': (lambda x: 'x' in x or
                                  '-' in x or
                                  '+' in x or
                                  ('y' in x and 'Dy' not in x)
                                  ),
                'has_mischmetal': lambda x: 'Mm' in x,
                'has_mischmetal_pct': lambda x: 'wt.% Mm' in x or 'w/o Mm' in x or 'M?' in x,
                'has_oxygen': lambda x: ('O' in x) and ('Os' not in x),
                'formula_has_R': lambda x: 'RNi' in x,
                'formula_has_Lm': lambda x: 'Lm' in x,
                'has_parens': lambda x: re.search(r'\((.*),+(.*)\)', x) is not None,
                }

    # remove (M) from composition string...
    remove_m = lambda x: re.compile(r'\s*\(M\)').sub('', x)
    df['clean_composition'] = df['Composition Formula'].apply(remove_m)

    # Remove Complex and Mg classes
    df = df[~df['Material Class'].isin(['Complex', 'Mg'])]

    # Perform all other cleaning ops
    for key,func in clean_funcs.items():
        df = df[~df['clean_composition'].apply(func)]
    
    return df


def clean_Hf(df):
    """ Further cleaning of training data by filtering Heat of Formation data"""
    
    df['Heat of Formation (kJ/mol H2)'] = \
        df['Heat of Formation (kJ/mol H2)'].apply(average_range)
    df = df[df['Heat of Formation (kJ/mol H2)'].apply(is_numeric)]
    df = df[~pd.isna(df['Heat of Formation (kJ/mol H2)'])]

    return df


def featurize(df):

    # use the standard magpie feature set...
    # e.g. min, max, range, and mean for a collection of elemental properties (e.g. atomic number, covalent radius, space group number...)
    f = composition_features.ElementProperty.from_preset('magpie')
    X = f.featurize_dataframe(df, col_id='composition', inplace=False)

    # matminer adds columns to the input dataframe...
    # so drop the original (metadata and target) columns from the new dataframe
    n_metadata = len(df.keys())
    X = X.iloc[:,n_metadata:]

    return X


def standardize(X):
    """ Use only with training set """
    std = preprocessing.StandardScaler()
    std.fit(X)
    X_std = std.transform(X)
    X_std = pd.DataFrame(data=X_std, index=X.index, columns=X.columns)
    return X_std


def average_range(x):
    """ find enthalpies reported as ranges and convert that entry to the average of the reported range """
    if type(x) is str and '-' in x:
        return np.mean(list(map(float, x.split(' - '))))
    else:
        return x


def is_numeric(x):
    """ throw out any remaining datetime or string entries """
    try:
        float(x);
        return True;
    except (TypeError, ValueError):
        return False


def load_test_data(X_train,
                   standardize=False,
                   test_xls='data/test_set_jhs.xlsx'):
    """
    Load the test data and augment and standardize the features
    in the same way as the training data
    
    Inputs
    ------
    X_train: X returned by featurize - TRAINING DATA
    test_csv: path to test data
    
    Returns
    ------
    X_test_std: dataframe of featurized, standardized test data
    df_test: Original, raw test data loaded from csv to input predictions
    """
    
    df_test = pd.read_excel(test_xls)
    df_test['composition'] = df_test['comp'].apply(mg.Composition)
    X_test = featurize(df_test)
    
    # Standardize test X based on training X
    if standardize:
        std = preprocessing.StandardScaler()
        std.fit(X_train)
        X_test_std = std.transform(X_test)
        X_test = pd.DataFrame(data=X_test_std,
                              index=X_test.index,
                              columns=X_train.columns)
    
    return X_test, df_test
