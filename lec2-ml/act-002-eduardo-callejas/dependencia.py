import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pygam.utils import generate_X_grid

def dependencia_parcial(modelo, X_train):

    
    x_grid = generate_X_grid(modelo)
    plt.figure(figsize=(10,5))
    attribute = X_train.columns
    
    cols = 3; rows = int(len(attribute) / cols)

    
    for i, n in enumerate(range(len(attribute))):
    
        plt.subplot(rows, cols, i + 1)
        
        partial_dep, confidence_intervals = modelo.partial_dependence(x_grid, feature = i + 1, width=.95)
        
        plt.plot(x_grid[:, n], partial_dep, color='tomato')
        
        plt.fill_between(x_grid[:, n],
                        confidence_intervals[0][:, 0],
                        confidence_intervals[0][:, 1],
                        color='tomato', alpha=.25)
        
        plt.title(attribute[n])
        plt.plot(X_train[attribute[n]], 
                [plt.ylim()[0]] * len(X_train[attribute[n]]),
                '|', color='orange', alpha=.5)
    plt.tight_layout()