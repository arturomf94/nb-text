import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from io import StringIO
import csv

all_texts = pd.read_csv('../all_texts.csv')
result_classification = pd.read_csv('../result_classification.csv', index_col=0)
print(all_texts.head())
print(result_classification.head())

data = pd.merge(all_texts, result_classification, how = 'inner', on = 'file_name')

data = data[data['category'] != 'myster']
data = data[data['category'] != 'satire']
data = data[data['category'] != 'travel']
data = data[data['category'] != 'history']
data = data[data['category'] != 'children']

print(data.shape)
print(data.head())

data['category_id'] = data['category'].factorize()[0]
category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

print(data.head())

fig = plt.figure(figsize = (8,6))
data.groupby('category').content.count().plot.bar(ylim = 0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

features = tfidf.fit_transform(data.content).toarray()
labels = data.category_id
print(features.shape)

correlations = []

N = 2
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  correlations.append([category, ' and '.join(unigrams[-N:]), ' and '.join(bigrams[-N:])])

with open('ngram_correlations.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['category', 'unigrams', 'bigrams'])
    for elem in correlations:
		wr.writerow(elem)


# Second 

all_texts = pd.read_csv('../all_texts.csv')
result_classification = pd.read_csv('../result_classification.csv', index_col=0)
print(all_texts.head())
print(result_classification.head())

data = pd.merge(all_texts, result_classification, how = 'inner', on = 'file_name')

data = data[data['category'] != 'myster']
data = data[data['category'] != 'satire']
data = data[data['category'] != 'travel']
data = data[data['category'] != 'history']
data = data[data['category'] != 'children']
data = data[data['category'] != 'romance']
data = data[data['category'] != 'religion']
data = data[data['category'] != 'humor']
data = data[data['category'] != 'crime']

print(data.shape)
print(data.head())

data['category_id'] = data['category'].factorize()[0]
category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

print(data.head())

fig = plt.figure(figsize = (8,6))
data.groupby('category').content.count().plot.bar(ylim = 0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

features = tfidf.fit_transform(data.content).toarray()
labels = data.category_id
print(features.shape)

correlations = []

N = 2
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  correlations.append([category, ' and '.join(unigrams[-N:]), ' and '.join(bigrams[-N:])])

with open('ngram_correlations2.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['category', 'unigrams', 'bigrams'])
    for elem in correlations:
		wr.writerow(elem)