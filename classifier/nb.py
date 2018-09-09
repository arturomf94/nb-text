import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from io import StringIO
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from stemming.porter2 import stem
import string
import re

# Cleaning functions 
nltk.download('averaged_perceptron_tagger')

tagger = PerceptronTagger()
tagset = None
stop = nltk.corpus.stopwords
wordnet_lemmatizer = WordNetLemmatizer()

grammar = '''REMOVE: {<PRP><VBP>?<VBG><TO>?}
                         {<PRP><MD><VB><TO>}
                         {<VBZ><DT><JJ>}
                         {<MD><DT><NN>}
                         {<NNP><PRP><VBP>}
                         {<MD><PRP>}
                         {<NNP><PRP><VBP>}
                         {<WDT><MD>}
                         {<PRP><VBP><VBG><VB><DT>}
                         {<VBZ><DT><JJ>}
                         {<VBZ><EX><NN><PRP><VBP><TO><VB>}
                         {<DT><VBZ>}
                         {<PRP><VBP><VBG><TO>}
                         {<MD><VB><TO><VB>}
                         {<VBZ><EX><DT>}
                         {<VB><TO>}
                         {<VBZ>}
                         {<DT>}
                         {<EX>}
                         {<PRP><VBP>}
                         {<CD>}
                         {<PRP\$>}
                         {<PRP>}
                         {<TO>}
                         {<IN>}
                         {<VBP>}
                         {<CC>}
              '''

def stem_doc(x):
    red_text = [stem(word.strip()) for word in x.split(" ") if word.strip()!='']
    return ' '.join(red_text)

def lem(x):
    try:
        return wordnet_lemmatizer.lemmatize(x,pos='v')
    except:
        return x
        
def remove_url(x):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x)

def cleanse_text(text):
    if text:
        text = remove_url(text)
        addl_txt = addl_clean_words(text)
        red_text = clean_words(addl_txt)
        
        no_gram = red_text
        try:
            no_gram = remove_grammar(red_text)
        except:
            no_gram = red_text
    
        #clean = ' '.join([i for i in no_gram.split() if i not in stop])
        if no_gram:
            clean = ' '.join([i for i in no_gram.split()])
            red_text = [lem(word) for word in clean.split(" ")]
            red_text = [stem(word) for word in clean.split(" ")]
            return clean_words(' '.join(red_text))
        else:
            return no_gram
    else:
        return text

        
def addl_clean_words(words):
    # any additional data pre-processing
    words = words.replace('can\'t','cannot')
    words = words.replace('won\'t','would not')
    words = words.replace('doesn\'t','does not')
    return words
    
def clean_words(words):
    if words:
        words = remove_email(words)
        words = words.replace('\t',' ')
        words = words.replace(',',' ')
        words = words.replace(':',' ')
        words = words.replace(';',' ')
        words = words.replace('=',' ')
        #words = words.replace('\x92','') # apostrophe encoding
        words = words.replace('\x08','\\b') # \b is being treated as backspace
        #words = ''.join([i for i in words if not i.isdigit()])
        words = words.replace('_',' ')
        words = words.replace('(',' ')
        words = words.replace(')',' ')
        words = words.replace('+',' ')
        words = words.replace('-',' ')
        words = words.replace('`',' ')
        words = words.replace('\'',' ')
        words = words.replace('.',' ')
        words = words.replace('#',' ')
        words = words.replace('/',' ')
        words = words.replace('_',' ')
        words = words.replace('"',' ')
        return words.strip()
    return words

    
def remove_grammar(review):
    sentences = nltk.sent_tokenize(review)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    result_review = []
    for sentence in sentences:
        if sentences.strip():
            tagged_review = nltk.tag._pos_tag(sentence, tagset, tagger)
            cp = nltk.RegexpParser(grammar)
            result = cp.parse(tagged_review)
            result_review.append(traverseTree(result))
    return ''.join([word for word in result_review])
    
# Remove email
def remove_email(words):
    mod_words = ''
    if words:
        if words.strip():
            for word in words.split(' '):
                if (word.strip().lower()=='email') or (word.strip().lower()=='phn') or (word.strip().lower()=='phone') or (len(word.strip())<=1):
                    continue
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", word.lower()):
                    mod_words = mod_words+' '+word
                #else:   
    else:
        return words
    return mod_words.strip()
    
def traverseTree(tree):
    imp_words = []
    for n in tree:
        if not isinstance(n, nltk.tree.Tree):               
            if isinstance(n, tuple):
                imp_words.append(n[0])
            else:
                continue
    return ' '.join([word for word in imp_words])


# Prepare data

all_texts = pd.read_csv('../all_texts.csv')
result_classification = pd.read_csv('../result_classification.csv', index_col=0)

data = pd.merge(all_texts, result_classification, how = 'inner', on = 'file_name')
data['category_id'] = data['category'].factorize()[0]
category_id_df = data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

data['clean_sum'] = data['content'].apply(lambda x: cleanse_text(x))

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1, 2), stop_words = 'english')

features = tfidf.fit_transform(data.clean_sum).toarray()
labels = data.category_id
features.shape

X_train, X_test, y_train, y_test = train_test_split(data['clean_sum'], data['category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)


print(clf.predict(count_vect.transform(["The disturbing piece of information is that the evil Hutts, criminal warlords of the galaxy, are building a secret superweapon: a reconstruction of the original Death Star, to be named Darksaber. This planet-crushing power will be in the ruthless hands of Durga the Hutt -- a creature without conscience or mercy.But there is worse news yet: the Empire lives. The beautiful Admiral Daala, still very much alive and more driven than ever to destroy the Jedi, has joined forces with the defeated Pellaeon, former second in command to Grand Admiral Thrawn. Together they are marshaling Imperial forces to wipe out the New Republic.Now, as Luke, Han, Leia, Chewbacca, Artoo and Threepio regroup to face these threats, they are joined by new Jedi Knights and Callista. Together they must fight on two fronts, outshooting and outsmarting the most formidable enemies in the galaxy. In Darksaber the Jedi are heading for the ultimate test of their power--a test in which all the temptations of the dark side beckon. And Luke Skywalker must draw upon his innermost resources to fight for a world in which he can not only live, but dare to love."])))