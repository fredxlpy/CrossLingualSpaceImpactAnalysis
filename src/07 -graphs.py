import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

"""
HEATMAP OF CORRELATIONS BETWEEN LINGUISTIC DISTANCE AND IMPACT
ON REPRESENTATION SPACE (PER LAYER)
"""
data = pd.read_excel('./Output/correlation_lang2vec_vs_impact.xlsx')

corr_dfs = []  # Heatmap values
annot_text_dfs = []  # Heatmap annotation

lang_distances = ['syntactic', 'geographic', 'inventory', 'genetic', 'phonological']

for dist in lang_distances:
    corr = data.loc[data['Linguistic distance'] == dist, ['Layer', 'Pearson (corr)']].reset_index(drop=True).set_index('Layer')
    corr = corr.rename(columns={'Pearson (corr)':dist})
    corr_dfs.append(corr)

    # Add annotation text for heatmap to show significant p-values
    annot_text = data.loc[data['Linguistic distance'] == dist, ['Layer', 'Pearson (corr)', 'Pearson (pvalue)']].reset_index(drop=True).set_index('Layer')
    annot_text = annot_text.rename(columns={'Pearson (corr)': dist, 'Pearson (pvalue)':'pvalue'})
    annot_text = annot_text.apply(lambda x: '{}**'.format(np.round(x[dist],3)) if x['pvalue'] <=0.01 else '{}*'.format(np.round(x[dist],3)) if x['pvalue'] <=0.05 else str(np.round(x[dist],3)), axis=1)
    annot_text_dfs.append(annot_text)

short_table = pd.concat(corr_dfs, axis=1)
annotation = pd.concat(annot_text_dfs, axis=1)

sns.set(rc={'figure.figsize':(9,10)})
sns.set(font_scale=1.5)
fig = sns.heatmap(short_table, annot=annotation, cbar=False, fmt='', annot_kws={'fontsize': 20}) # vmin=-1, vmax=1, cmap='rocket_r', annot=annotation
fig.set_xticklabels(['SYN','GEO','INV','GEN','PHON'])
fig.tick_params(axis='y', rotation=0)
fig.hlines([12], *fig.get_xlim(), colors='black', linewidth=5)
fig.figure.savefig('./Output/Plots/corr_heatmap_en.pdf', dpi=600, bbox_inches='tight')
plt.close('all')


"""
HEATMAP OF ZERO-SHOT CROSS-LINGUAL TRANSFER PERFORMANCE
"""
source_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
target_langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

data = pd.read_excel('./Output/ZS_results.xlsx')
data = pd.crosstab(index = data['Source'], columns= data['Target'],
                   values = data['Accuracy'], aggfunc='mean')

# Add average transfer performance
data.loc['AVG',:] = data.mean(axis=0)
data.loc[:,'AVG'] = data.mean(axis=1)
data = data*100
data.iloc[-1,-1] = np.nan

sns.set(rc={'figure.figsize':(12,12)})
sns.set(font_scale=1)
fig = sns.heatmap(data, annot=True, cbar=False, fmt=".2f")
plt.xlabel('Target Language')
plt.xlabel('Source Language')
fig.hlines([15], *fig.get_xlim(), colors='black')
fig.vlines([15], *fig.get_ylim(), colors='black')
fig.figure.savefig('./Output/Plots/ZS_performance.pdf', dpi=600, bbox_inches='tight')
plt.close()
