import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_plot(df):
    corr = df.corr()
    bool_mask = np.zeros_like(corr, dtype=np.bool)
    bool_mask[np.triu_indices_from(bool_mask)] = True
    f, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, mask=bool_mask, cmap=sns.diverging_palette(0,255, as_cmap=True))

def pairwise_corr(df, target=None):
    if target != None:
        corr = df.drop(target,axis=1).corr()
    else:
        corr = df.corr()
    corr_df = pd.DataFrame()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            row=pd.DataFrame({'feature1':corr.columns[i], 'feature2':corr.columns[j], 
                              'corr_coef':corr.iloc[i,j], 'abs_corr_coef':np.abs(corr.iloc[i,j])},
                              index=[i])
            corr_df = corr_df.append(row, ignore_index=True)

    corr_df = corr_df.sort_values('abs_corr_coef', ascending=False).reset_index(drop=True)
    return corr_df

def feature_reduction(n_features_to_drop, df, target):
    
    pc = pairwise_corr(df, target)
    pct = pairwise_corr(df)

    i = 0
    to_drop=[]
    while len(to_drop) < n_features_to_drop:
        f1 = pc.iloc[i,:]['feature1']
        f2 = pc.iloc[i,:]['feature2']

        corr1 = pct[(pct['feature1'] == f1) & (pct['feature2'] == target)]['abs_corr_coef'].values
        corr2 = pct[(pct['feature1'] == f2) & (pct['feature2'] == target)]['abs_corr_coef'].values

        if corr1 < corr2:
            to_drop.append(f1)
        elif corr2 < corr1:
            to_drop.append(f2)
        else:
            print("help")

        i += 1
    
    return to_drop