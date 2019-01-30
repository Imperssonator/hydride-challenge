import os
import numpy as np
import pandas as pd

from sklearn import metrics, model_selection, preprocessing, feature_selection
from sklearn import tree, ensemble, neural_network, linear_model, svm
from skmultilearn.model_selection import IterativeStratification

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
                 'font.size': 14,
                 'savefig.dpi': 300,
                 'figure.dpi': 96
                })


class MultiLabelClassifier():
    """
    Class for the evaluation of models and hyperparameters
    for multi-label classification problems
    """
    
    def __init__(self, estimator, name='multi-label-classifier'):
        """
        estimator must be a sklearn callable e.g.
        ensemble.RandomForestClassifier
        """
        
        self.estimator = estimator
        self.name = name
        self.last_scores = None
        self.best_hypes = {}
        
    
    def evaluate(self, X, y, hypes={}, n_splits=5, average=None):

        """
        Evaluate the estimator on X and y with given hyperparameters,
        return a dataframe of cross-validation results including precision,
        recall, F1 score, and class surpports for both training and validation
        of each fold

        Inputs:
        ------
        estimator: an estimator class (callable) from sklearn,
        X: an array with shape = (samples, features), and
        y: an array with shape = (samples,), where each entry is a 1-D array of class labels

        hypes: a dictionary of estimator hyperparameters
        n_splits: the number of cross-validation splits to use
        average: type of averaging (over classes) to use for scoring metrics (if any) (don't use this)

        Returns:
        ------
        scores: a DataFrame with training and validation scores from every split of cross validation
        model_i: an instance of "estimator" fit during the last fold of cross-validation
        """

        ### INITIALIZE SCORING DATAFRAME ###
        fractions = ['train', 'val']
        scoring_metrics = ['hamming', 'precision', 'recall', 'F1', 'support']
        score_columns = pd.MultiIndex.from_product([fractions, scoring_metrics])
        scores = pd.DataFrame(columns=score_columns)

        ### SET UP X-VALIDATION ###
        # cv = model_selection.KFold(n_splits=n_splits,
        #                                shuffle=True)
        cv = IterativeStratification(n_splits=n_splits, order=1)

        ### RUN CV AND SCORE MODEL ###
        for idx, (train, val) in enumerate(cv.split(X,y)):

            X_train = X[train,:]; y_train = y[train,:]
            X_val = X[val,:]; y_val = y[val,:]

    #         if idx==0:
    #             for v in ['X_train','y_train','X_val','y_val']:
    #                 print('{} shape: {}'.format(v, eval('{}.shape'.format(v))))

            ### INSTANTIATE AND FIT MODEL ###
            model_i = self.estimator(**hypes)
            model_i.fit(X_train, y_train)

            for frac in ['train','val']:

                # Calculate Hamming loss
                scores.loc[idx, (frac,'hamming')] = \
                    metrics.hamming_loss(eval('y_'+frac),
                                         model_i.predict(eval('X_'+frac))
                                        )

                # Calculate Precision, Recall, F1
                scores.loc[idx, (frac,'precision')], \
                scores.loc[idx, (frac,'recall')], \
                scores.loc[idx, (frac,'F1')], \
                scores.loc[idx, (frac,'support')] = \
                    metrics.precision_recall_fscore_support(eval('y_'+frac),
                                                            model_i.predict(eval('X_'+frac)),
                                                            average=average
                                                           )
        
        self.last_scores = scores

        return scores
    
    
    def plot_metric_byclass(self, fraction='val', metric='F1', ax=None, figsize=(5,4),
                            classes=np.arange(10,100,10).astype(int)):
        """
        Plot the results of model evaluation
        Y-axis is the chosen metric,
        X-axis is the classes from 10-90% Element A
        Values are the mean metric across n-fold cross validation,
        Shaded area is the standard deviation across n-fold cross validation

        Inputs
        ------
        fraction: 'train' or 'val'
        metric: the name of a metric to plot
        ax: axes to plot on - if None will create new figure
        figsize: kwarg for plt.figure()

        Outputs
        ------
        Axes object with the new series on it
        """

        if self.last_scores is None:
            raise AttributeError('No scores to plot; evaluate first')
        df = self.last_scores
        name = self.name

        ### CALCULATE VALIDATION STATS ###
        means = df.agg(lambda col: np.mean(col.values))['val',metric]
        stds = df.agg(lambda col: np.std(col.values))['val',metric]

        if classes is None:
            classes = list(range(len(means)))

        ### BUILD PLOTS ###
        if ax is None:
            plt.figure(figsize=figsize)
            ax=plt.gca()

        ax.plot(classes, means, '-', alpha=0.8, label=name)
        ax.fill_between(classes, means-stds, means+stds, alpha=0.2)

        ax.set_ylabel('{} ({}-fold CV)'.format(metric, df.shape[0]))
        ax.set_xlabel('% Element A (class)')
        ax.set_ylim([0,1])
        ax.set_xticks(classes)

        return ax
    
    
    def optimize_hyperparameters(self, hype_instru_dict, X, y, n_splits=5,
                                 average='weighted', metric='F1',
                                 budget=100, n_workers=4):
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
        y: class labels, (samples, classes)
        n_splits: number of folds for CV at each objective function evaluation
        average: what type of averaging to use for metric ('weighted','micro','macro')
        metric: 'F1', 'precision', 'recall'
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
                     'average':average,
                     'metric':metric}
        
        # The tricky part here is that obj_fun must accept the instrumented
        # hyperparameter variables as its first positional arguments, but then
        # the evaluate() method takes them as a kwarg, so we have to re-build the
        # hypes kwarg dict inside the lambda........
        obj_fun = (lambda *hype_args, **eval_args:
                   -self.evaluate(eval_args['X'], eval_args['y'],
                                  hypes = {k:v for k,v in zip(eval_args['hype_names'],hype_args)},
                                  n_splits = eval_args['n_splits'],
                                  average=eval_args['average'])['val',eval_args['metric']].mean().mean()
                  )
        # At the moment, assumes we're optimizing a metric where higher is better
        
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
        print('Recommended hyperparameters: ', self.best_hypes)
        
        return self.best_hypes
        