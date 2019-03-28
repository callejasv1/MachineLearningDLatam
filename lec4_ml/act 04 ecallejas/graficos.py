
import pandas as pd
import matplotlib.pyplot as plt





def classif_report(df):
    class_report = pd.DataFrame(df).drop(columns=['micro avg','macro avg']).T
    for i in ['f1-score', 'precision','recall']:
        plt.figure()
        plt.title(i)
        class_report.sort_values(by=i, inplace=True)
        plt.barh(class_report.index, class_report[i])
    