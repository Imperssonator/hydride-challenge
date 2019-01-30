import os
import numpy as np
import pandas as pd

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


def plot_f1_bymodel(models, ax=None, figsize=(5,4)):
    """
    Plot the results of model evaluation
    Y-axis is the support-weighted F1 score,
    X-axis is the model name
    Values are the mean metric across n-fold cross validation,
    Error bars are the standard deviation across n-fold cross validation
    
    results is a dictionary {model_name: scores DataFrame}
    """
    results_df = pd.DataFrame()
    
    ### CALCULATE VALIDATION STATS ###
    for model in models:
        
        val_f1 = model.last_scores[[('val','F1'), ('val','support')]].copy()
        val_f1.columns=['f1','support']
        val_f1['support_weights'] = val_f1['support'].apply(lambda s: s/s.sum())
        val_f1['f1_weighted'] = val_f1.apply(lambda r: (r['f1']*r['support_weights']).sum(),
                                             axis=1)
        results_df[model.name] = val_f1['f1_weighted']

    ### BUILD PLOTS ###
    if ax is None:
        plt.figure(figsize=figsize)
        ax=plt.gca()
    
    sns.barplot(data=results_df, ax=ax, ci='sd', errwidth=1, capsize=0.25)
    plt.xticks(rotation=30, ha='right')
    ax.set_ylabel('Support-weighted $F_1$ Score')
    ax.set_ylim([0,1])
    
    return ax