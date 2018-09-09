import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from io import StringIO

# Prepare data

all_texts = pd.read_csv('../all_texts.csv')
result_classification = pd.read_csv('../result_classification.csv', index_col=0)

data = pd.merge(all_texts, result_classification, how = 'inner', on = 'file_name')
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

clf = MultinomialNB().fit(X_train_tfidf, y_train)


print(clf.predict(count_vect.transform(["The disturbing piece of information is that the evil Hutts, criminal warlords of the galaxy, are building a secret superweapon: a reconstruction of the original Death Star, to be named Darksaber. This planet-crushing power will be in the ruthless hands of Durga the Hutt -- a creature without conscience or mercy.But there is worse news yet: the Empire lives. The beautiful Admiral Daala, still very much alive and more driven than ever to destroy the Jedi, has joined forces with the defeated Pellaeon, former second in command to Grand Admiral Thrawn. Together they are marshaling Imperial forces to wipe out the New Republic.Now, as Luke, Han, Leia, Chewbacca, Artoo and Threepio regroup to face these threats, they are joined by new Jedi Knights and Callista. Together they must fight on two fronts, outshooting and outsmarting the most formidable enemies in the galaxy. In Darksaber the Jedi are heading for the ultimate test of their power--a test in which all the temptations of the dark side beckon. And Luke Skywalker must draw upon his innermost resources to fight for a world in which he can not only live, but dare to love."])))