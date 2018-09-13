import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from io import StringIO
import seaborn as sns
import csv
from imblearn.under_sampling import RandomUnderSampler

# Prepare data

all_texts = pd.read_csv('../all_texts.csv')
result_classification = pd.read_csv('../result_classification.csv', index_col=0)

data = pd.merge(all_texts, result_classification, how = 'inner', on = 'file_name')
data = data[data['category'] != 'myster']
data = data[data['category'] != 'satire']
data = data[data['category'] != 'travel']
data = data[data['category'] != 'history']
data = data[data['category'] != 'children']
data['category_id'] = data['category'].factorize()[0]
category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

features = tfidf.fit_transform(data.content).toarray()
labels = data.category_id
features.shape

X_train, X_test, y_train, y_test = train_test_split(data['content'], data['category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB(alpha = .001).fit(X_train_tfidf, y_train)

test_texts = pd.read_csv('../test_texts.csv')

test_results = []
for value in test_texts.values:
	print('Category:\n')
	print(value[1].replace('.txt',''))
	print('Predicted:\n')
	print(clf.predict(count_vect.transform([value[0]]))[0])
	print(clf.predict_proba(count_vect.transform([value[0]]))[0])
	test_results.append([value[1].replace('.txt',''),clf.predict(count_vect.transform([value[0]]))[0]])

with open('test_results.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['file_name', 'prediction'])
    for elem in test_results:
		wr.writerow(elem)

accuracies = cross_val_score(MultinomialNB(alpha = .001), features, labels, scoring='accuracy', cv=5)

print(accuracies.mean())

model = MultinomialNB(alpha = .001)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size = 0.2, random_state=0)
clf = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(metrics.classification_report(y_test, y_pred, target_names=data['category'].unique()))


# Resampling/Undersampling 

rus = RandomUnderSampler(ratio = .8, random_state=0)
rus.fit(features, labels)
X_resampled, y_resampled = rus.sample(features, labels)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.1, random_state=0)

clf = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(metrics.classification_report(y_test, y_pred, target_names=data['category'].unique()))


# Segundo intento

# Prepare data

all_texts = pd.read_csv('../all_texts.csv')
result_classification = pd.read_csv('../result_classification.csv', index_col=0)

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
data['category_id'] = data['category'].factorize()[0]
category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

features = tfidf.fit_transform(data.content).toarray()
labels = data.category_id
features.shape

X_train, X_test, y_train, y_test = train_test_split(data['content'], data['category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB(alpha = .001).fit(X_train_tfidf, y_train)

test_texts = pd.read_csv('../test_texts.csv')

test_results = []
for value in test_texts.values:
	print('Category:\n')
	print(value[1].replace('.txt',''))
	print('Predicted:\n')
	print(clf.predict(count_vect.transform([value[0]]))[0])
	print(clf.predict_proba(count_vect.transform([value[0]]))[0])
	test_results.append([value[1].replace('.txt',''),clf.predict(count_vect.transform([value[0]]))[0]])

with open('test_results2.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['file_name', 'prediction'])
    for elem in test_results:
		wr.writerow(elem)

accuracies = cross_val_score(MultinomialNB(alpha = .001), features, labels, scoring='accuracy', cv=5)

print(accuracies.mean())

model = MultinomialNB(alpha = .001)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size = 0.2, random_state=0)
clf = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(metrics.classification_report(y_test, y_pred, target_names=data['category'].unique()))