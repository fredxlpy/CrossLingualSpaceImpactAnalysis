import lang2vec.lang2vec as l2v
import pandas as pd

"""
Computing 5 different language similarity metrics between language pairs.
We uses the pre-computed lang2vec distances.
"""

languages = {'ar':'ara', 'bg':'bul', 'de':'deu', 'el':'ell', 'en':'eng', 'es':'spa', 'fr':'fra',
             'hi':'hin', 'ru':'rus', 'sw':'swa', 'th':'tha', 'tr':'tur', 'ur':'urd', 'vi':'vie', 'zh':'zho'}


writer = pd.ExcelWriter('./Output/language_similarity.xlsx', engine='xlsxwriter')

for distance_type in ['syntactic', 'geographic', 'inventory', 'genetic', 'phonological']:
    dist_array = l2v.distance(distance_type, list(languages.values()))

    # Long Table
    df = pd.DataFrame(dist_array, columns=languages.keys(), index=languages.keys())
    df['Source'] = df.index
    df = pd.melt(df, id_vars='Source').rename(columns={'variable':'Target'})
    df.to_excel(writer, sheet_name=distance_type, index=False)

writer.close()