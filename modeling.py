import os
import numpy as np
import pandas as pd

from sklearn import metrics, model_selection, preprocessing, feature_selection
from sklearn import tree, ensemble, neural_network, linear_model, svm
from sklearn.base import clone
# from skmultilearn.model_selection import IterativeStratification
from scipy import stats

from nevergrad import instrumentation as instru
from nevergrad.optimization import optimizerlib
from concurrent import futures

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
plt.style.use('seaborn')
rcParams.update({'figure.autolayout': True,
                 'xtick.top': True,
                 'xtick.direction': 'in',
                 'ytick.right': True,
                 'ytick.direction': 'in',
                 'font.sans-serif': 'Arial',
                 'font.size': 16,
                 'savefig.dpi': 300,
                 'figure.dpi': 96
                })


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


class Regressor():
    """
    Class for the evaluation of models and hyperparameters
    for regression problems
    """
    
    def __init__(self, estimator, name='multi-label-classifier'):
        """
        estimator must be an instance of sklearn model e.g.
        ensemble.RandomForestClassifier()
        """
        
        self.estimator = estimator
        self.name = name
        self.last_scores = None
        self.last_hypes = {}
        self.best_hypes = {}
        self.last_splits = []
        
    
    def evaluate(self, X, y, hypes={}, n_splits=5, shuffle=True, standardize=True, groups=None):

        """
        Evaluate the estimator on X and y with given hyperparameters,
        return a dataframe of cross-validation results including
        mean absolute error, mean absolute percentage error, pearson and spearman
        for both training and validation of each fold

        Inputs:
        ------
        estimator: an estimator class (callable) from sklearn,
        X: an array with shape = (samples, features), and
        y: an array with shape = (samples,), where each entry is a 1-D array of class labels

        hypes: a dictionary of estimator hyperparameters
        n_splits: the number of cross-validation splits to use
        standardize: whether to standardize data at each fold
        groups: must contain a list-like of shape (samples,) with some type of group label for each sample. If passed, LOGO-CV will be used by default

        Returns:
        ------
        scores: a DataFrame with training and validation scores from every split of cross validation
        """
        
        ### SET HYPERPARAMETERS ###
        model = clone(self.estimator)  # Gotta do this otherwise funky things happen
        model.set_params(**hypes)
        
        ### INITIALIZE SCORING DATAFRAME ###
        fractions = ['train', 'val']
        scoring_metrics = ['mae', 'mape', 'medape', 'pearson', 'spearman']
        score_columns = pd.MultiIndex.from_product([fractions, scoring_metrics])  # This sets up a heirarchical index for the results dataframe
        score = pd.DataFrame(columns=score_columns)

        ### SET UP X-VALIDATION ###
        
        if groups is not None:
            cv = model_selection.LeaveOneGroupOut()
            splitter = enumerate(cv.split(X,y,groups))
        else:
            cv = model_selection.KFold(n_splits=n_splits, shuffle=shuffle)
            splitter = enumerate(cv.split(X,y))

        ### RUN CV AND SCORE MODEL ###
        last_splits = []  # Keep track of split indices for forensics
        for idx, (train, val) in splitter:

            X_train = X.iloc[train,:]; y_train = y.iloc[train]
            X_val = X.iloc[val,:]; y_val = y.iloc[val]
            
            if standardize:
                std = preprocessing.StandardScaler()
                std.fit(X_train)
                X_train, X_val = std.transform(X_train), std.transform(X_val)

    #         if idx==0:
    #             for v in ['X_train','y_train','X_val','y_val']:
    #                 print('{} shape: {}'.format(v, eval('{}.shape'.format(v))))

            ### INSTANTIATE AND FIT MODEL ###
            last_splits.append((train, val))
            model.fit(X_train, y_train)

            for frac in ['train','val']:
                
                # y_true will either be y_train or y_val depending on what 'frac' is. Kind of hacky.
                y_true = eval('y_'+frac)
                y_pred = model.predict(eval('X_'+frac))
                
                # Calculate MAE
                score.loc[idx, (frac,'mae')] = \
                    metrics.mean_absolute_error(y_true, y_pred)
                        
                # Calculate MAPE
                score.loc[idx, (frac,'mape')] = \
                    mean_absolute_percentage_error(y_true, y_pred)
                    
                # Calculate MedAPE
                score.loc[idx, (frac,'medape')] = \
                    median_absolute_percentage_error(y_true, y_pred)

                # Calculate pearson
                score.loc[idx, (frac,'pearson')] = \
                    stats.pearsonr(y_true, y_pred)[0]

                # Calculate spearman
                score.loc[idx, (frac,'spearman')] = \
                    stats.spearmanr(y_true, y_pred)[0]

        self.estimator = model
        self.last_scores = score
        self.last_hypes = hypes
        self.last_splits = last_splits

        return score
        
        
    def parity_plot(self, X, y, hypes='current', shuffle=True, standardize=True, test_size=0.2,
                    ax=None, figsize=(5,5), lim=450, title=''):
        """
        Perform train/val split, fit and predict, plot y_pred vs. y_true
        with a dashed line at y=x for both train and val fractions

        Inputs:
        ------
        X: an array with shape = (samples, features), and
        y: an array with shape = (samples,), where each entry is a 1-D array of class labels

        hypes: 'current' (use whatever was used when evaluate() was last called), 'best' (use whatever was obtained by optimize_hyperparameters), or just a dictionary of valid hyperparameters for the estimator
        shuffle: whether to shuffle data before splitting
        standardize: whether to standardize data after splitting
        test_size: fraction of data to use for validation
        ax: pass a matplotlib Axes onto which the plot will be drawn - if None, will create a new figure
        figsize: (inches width, inches height)
        lim: upper limit for the x/y axes (zoom in on higher-density region)
        title: title for the plot

        Returns:
        ------
        ax: the Axes object on which the plot was drawn
        """
        
        ### SPLIT + STANDARDIZE DATA ###
        X_train, X_val, y_train, y_val = \
            model_selection.train_test_split(X, y, shuffle=shuffle, test_size=test_size)
        
        if standardize:
            std = preprocessing.StandardScaler()
            std.fit(X_train)
            X_train, X_val = std.transform(X_train), std.transform(X_val)
        
        ### FIT + PREDICT ###
        model = clone(self.estimator)
        if hypes=='current':
            model.set_params(**self.last_hypes)
        elif hypes=='best':
            model.set_params(**self.best_hypes)
        else:
            try:
                model.set_params(**hypes)
            except:
                print('Passed hypes were invalid')
        
        model.fit(X_train, y_train)

        ### BUILD PLOTS ###
        if ax is None:
            plt.figure(figsize=figsize)
            ax=plt.gca()

        for frac in ['train','val']:
            y_true = eval('y_'+frac)
            y_pred = model.predict(eval('X_'+frac))
            ax.scatter(y_true, y_pred, alpha=0.7)

        ax.plot((0,lim), (0,lim), linestyle='--', color='xkcd:gray')
        ax.set_aspect('equal','datalim')
        ax.set_xlim([-10,lim]); ax.set_ylim([-10,lim])
        ax.set_xlabel(r'True $\Delta H$')
        ax.set_ylabel(r'Predicted $\Delta H$')
        ax.legend(['y=x', 'train', 'val'])
        plt.title(title)

        return ax
    
    
    def optimize_hyperparameters(self, hype_instru_dict, X, y, n_splits=5,
                                 metric='mape', lower_better=True, budget=100, n_workers=4):
        """
        Optimize the hyperparameters of the estimator within specified
        hyperparameter ranges, using training data X and y

        Inputs
        ------
        hype_instru_dict: needs to look like this, using RF as an example:
        {
        'n_estimators' = instru.variables.OrderedDiscrete(list(range(5,200,5))),
        'max_depth' = instru.variables.OrderedDiscrete(list(range(1,30,1))),
        'min_samples_leaf' = instru.variables.OrderedDiscrete(list(range(1,11,1)))
        }
        
        X: feature array, (samples, feats)
        y: target values, (samples, )
        n_splits: number of folds for CV at each objective function evaluation
        metric: 'mae', 'mape', 'pearson', 'spearman'
        budget: number of allowable objective function evaluations
        n_workers: number of parallel workers to allocate

        Outputs
        ------
        best_hypes: dictionary of the best hyperparameter values found during optimization
        """
        
        ### MAKE AN INSTRUMENTED OBJECTIVE FUNCTION ###
        
        hype_names = [k for k in hype_instru_dict.keys()]
        hype_args = [v for v in hype_instru_dict.values()]
        eval_args = {'hype_names':hype_names,
                     'X':X,
                     'y':y,
                     'n_splits':n_splits,
                     'metric':metric}
        
        # The tricky part here is that obj_fun must accept the instrumented
        # hyperparameter variables as its first positional arguments, but then
        # the evaluate() method takes them as a kwarg, so we have to re-build the
        # hypes kwarg dict inside the lambda........
        if lower_better:
            obj_fun = (lambda *hype_args, **eval_args:
                       self.evaluate(eval_args['X'], eval_args['y'],
                                     hypes = {k:v for k,v in zip(eval_args['hype_names'],hype_args)},
                                     n_splits = eval_args['n_splits'])['val',eval_args['metric']].mean()
                      )
        else:
            obj_fun = (lambda *hype_args, **eval_args:
                       -self.evaluate(eval_args['X'], eval_args['y'],
                                      hypes = {k:v for k,v in zip(eval_args['hype_names'],hype_args)},
                                      n_splits = eval_args['n_splits'])['val',eval_args['metric']].mean()
                      )
        
        ifunc = instru.InstrumentedFunction(obj_fun, *hype_args, **eval_args)
        
        ### RUN THE OPTIMIZATION ###
        # Particle Swarm Optimizer
        optimizer = optimizerlib.PSO(dimension=ifunc.dimension,
                                     budget=budget,
                                     num_workers=n_workers)

        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommended = optimizer.optimize(ifunc,
                                             executor=executor,
                                             batch_mode=False)
            
        ### CONVERT OUTPUTS BACK TO HP-SPACE ###
        best_hype_values, other_kwargs = ifunc.convert_to_arguments(recommended)
        self.best_hypes = {k:v for k,v in zip(hype_names, best_hype_values)}
        self.last_hypes = self.best_hypes
        print('Recommended hyperparameters: ', self.best_hypes)
        
        return self.best_hypes
        
