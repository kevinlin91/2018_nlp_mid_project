from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from gensim import corpora, models
import numpy as np
import json

# only deal with  [joy, sadness, anger neutral] labels
#wordsFiltered = [wordnet_lemmatizer.lemmatize(w) for w in words]
#wordsFiltered = [porter_stemmer.stem(w) for w in words]
#words = word_tokenize(data[0][0]['utterance'])#.replace('\u0092', "'"))
#tag.pos_tag(wordsFiltered)
def loading_data(path):
    data = json.load(open(path, 'r'))
    return data

def preprocessing_all(data):
    target_labels = ['joy', 'sadness', 'anger', 'neutral']
    #loading stopwords
    stopWords = set(stopwords.words('english'))
    #tokenize
    words = [ (word_tokenize(sentence['utterance']), sentence['emotion']) for dialogue in data for sentence in dialogue ]
    #stemming
    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    #processing
    output = list()
    stopWords = set(stopwords.words('english'))
    for word in words:
        sentence_token = word[0]
        label = word[1]
        if label not in target_labels:
            continue
        wordsFiltered = [wordnet_lemmatizer.lemmatize(w) for w in sentence_token if w not in stopWords]
        output.append( (' '.join(wordsFiltered), label) )
    return output
def feature_transformation_all(preprocessing_data, method = 'tfidf'):
    sentences = [x[0] for x in preprocessing_data]
    labels = [x[1] for x in preprocessing_data]
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        X = X.toarray()
    elif method == 'count':
        vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
        X = vectorizer.fit_transform(sentences)
        X = X.toarray()
    return X, labels

def feature_transformation_topic(preprocessing_data, topic = 100):
    sentences = [x[0] for x in preprocessing_data]
    sentences = [x.split(',') for x in sentences]
    labels = [x[1] for x in preprocessing_data]
    dictionary = corpora.Dictionary(sentences)
    corpus = [dictionary.doc2bow(text) for text in sentences]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics=topic)
    corpus_lsi = lsi[corpus_tfidf]
    features = list()
    for doc in corpus_lsi:
        features.append( [x[1] for x in doc] )
    return features, labels
#LSI mapping
#new_doc = "Human computer interaction"
#vec_bow = dictionary.doc2bow(doc.lower().split())
#vec_lsi = lsi[vec_bow] 

def pipeline_test(preprocessing_data):
    sentences = [x[0] for x in preprocessing_data]
    labels = [x[1] for x in preprocessing_data]
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC()),
    ])
    parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    }
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(sentences, labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    

def preprocessing_sep(data):
    target_labels = ['joy', 'sadness', 'anger', 'neutral']
    joy_uttre = list()
    sadness_uttre = list()
    anger_uttre = list()
    neutral_uttre = list()
    for dialog in data


    
    pass

def feature_transformation_sep(preprocessing_data):
    pass

if __name__ == '__main__':
    path = './Friends/friends_train.json'
    #path = './EmotionPush/emotionpush_train.json'
    preprocessing_data = preprocessing_all(loading_data(path))
    #feature_transformation_all(preprocessing_data)
    #feature_transformation_topic(preprocessing_data)
    #pipeline_test(preprocessing_data)



