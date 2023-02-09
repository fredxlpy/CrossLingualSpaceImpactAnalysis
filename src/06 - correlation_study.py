import pandas as pd
import scipy.stats

"""
CORRELATION BETWEEN IMPACT ON REPRESENTATION SPACE AND LANGUAGE DISTANCE
"""

linguistic_distances = ['syntactic', 'geographic', 'inventory', 'genetic', 'phonological']

cka = pd.read_excel('./Output/CKA.xlsx')
cka['Impact'] = 1 - cka['CKA']
# cka = cka[cka['Source'] == 'en']  # Choose specific source language

correlations = []
for ling_dist in linguistic_distances:

    lang_sim = pd.read_excel('./Output/language_similarity.xlsx', sheet_name=ling_dist)
    data = pd.merge(cka, lang_sim, how='inner', on=['Source', 'Target'])
    data = data[data['Source'] != data['Target']]  # Remove same source-target pair data points

    for layer in range(1,13):

        # CORRELATION PER LAYER
        layer_data = data[data['Layer']==layer]

        pearson = scipy.stats.pearsonr(layer_data['Impact'], layer_data['value'])
        spearman = scipy.stats.spearmanr(layer_data['Impact'], layer_data['value'])

        correlations.append([
            layer, ling_dist,
            pearson.statistic, spearman.correlation,
            pearson.pvalue, spearman.pvalue
        ])

    # CORRELATION ACROSS ALL LAYERS
    data = data.groupby(['Source','Target']).mean().reset_index()
    pearson = scipy.stats.pearsonr(data['Impact'], data['value'])
    spearman = scipy.stats.spearmanr(data['Impact'], data['value'])

    correlations.append([
        'AVG', ling_dist,
        pearson.statistic, spearman.correlation,
        pearson.pvalue, spearman.pvalue
    ])

correlations = pd.DataFrame(correlations, columns=['Layer', 'Linguistic distance', 'Pearson (corr)',
                                                   'Spearman (corr)', 'Pearson (pvalue)', 'Spearman (pvalue)'])

correlations.to_excel('./Output/correlation_lang2vec_vs_impact.xlsx', index=False)

############################################################################################
############################################################################################
############################################################################################

"""
CORRELATION BETWEEN IMPACT ON THE REPRESENTATION SPACE AND TRANSFER PERFORMANCE
"""
print('-'*50 + '\nCORR. BETWEEN REPRESENTATION SPACE IMPACT AND TRANSFER PERFORMANCE\n' + '-'*50)

# Load CKA and transfer performance values
cka = pd.read_excel('./Output/CKA.xlsx')
cka['Impact'] = 1 - cka['CKA']
# cka = cka[cka['Source'] == 'en']  # Choose specific source language
transfer_results = pd.read_excel('./Output/ZS_results.xlsx')

correlations = []

# Correlation for each layer
for layer in range(1,13):
    print(f'Layer {layer}:')

    cka_layer = cka[cka['Layer']==layer]
    data = pd.merge(cka_layer, transfer_results, on=['Source','Target'])

    # Correlations
    pearson = scipy.stats.pearsonr(data['Impact'], data['Accuracy'])
    spearman = scipy.stats.spearmanr(data['Impact'], data['Accuracy'])

    print(f'Pearson: {pearson.statistic} (pvalue: {pearson.pvalue})')
    print(f'Spearman: {spearman.statistic} (pvalue: {spearman.pvalue})')

    correlations.append([layer, pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue])

# Correlation across all layers
cka = cka.groupby(['Source', 'Target']).mean().reset_index()
data = pd.merge(cka, transfer_results, on=['Source', 'Target'])
pearson = scipy.stats.pearsonr(data['Impact'], data['Accuracy'])
spearman = scipy.stats.spearmanr(data['Impact'], data['Accuracy'])

print(f'Average:')
print(f'Pearson: {pearson.statistic} (pvalue: {pearson.pvalue})')
print(f'Spearman: {spearman.statistic} (pvalue: {spearman.pvalue})')
correlations.append(['ALL', pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue])

correlations = pd.DataFrame(correlations, columns=['Layer', 'Pearson (corr)', 'Spearman (corr)',
                                                   'Pearson (pvalue)', 'Spearman (pvalue)'])

correlations.to_excel('./Output/correlation_impact_vs_performance.xlsx')

############################################################################################
############################################################################################
############################################################################################

"""
CORRELATION BETWEEN TRANSFER PERFORMANCE AND LANGUAGE DISTANCE
"""
print('-'*50 + '\nCORR. BETWEEN TRANSFER PERFORMANCE AND LANGUAGE DISTANCE\n' + '-'*50)

# Load Zero-Shot Cross-Lingual Transfer Accuracies
performance = pd.read_excel('./Output/ZS_results.xlsx')

linguistic_distances = ['syntactic', 'geographic', 'inventory', 'genetic', 'phonological']

correlations = []
for dist_type in linguistic_distances:
    ling_dist = pd.read_excel('./Output/language_similarity.xlsx', sheet_name=dist_type)

    data = pd.merge(performance, ling_dist, how='inner', on=['Source','Target'])

    # Compute correlations
    pearson = scipy.stats.pearsonr(data['value'], data['Accuracy'])
    spearman = scipy.stats.spearmanr(data['value'], data['Accuracy'])

    correlations.append([dist_type, pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue])

    print(dist_type)
    print(f'Pearson: {pearson.statistic} (pvalue: {pearson.pvalue})')
    print(f'Spearman: {spearman.statistic} (pvalue: {spearman.pvalue})')

correlations = pd.DataFrame(correlations, columns=['Distance metric', 'Pearson (corr)', 'Spearman (corr)',
                                                       'Pearson (pvalue)', 'Spearman (pvalue)'])

correlations.to_excel('./Output/correlation_lang2vec_vs_performance.xlsx')