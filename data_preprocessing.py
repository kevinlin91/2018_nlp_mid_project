from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
import numpy as np
import json
import pickle

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

def preprocessing_sep(data):
    target_labels = ['joy', 'sadness', 'anger', 'neutral']
    joy_uttre = list()
    joy_uttre_start = list()
    sadness_uttre = list()
    sadness_uttre_start = list()
    anger_uttre = list()
    anger_uttre_start = list()
    neutral_uttre = list()
    neutral_uttre_start = list()
    for dialogue in data:
        if dialogue[0]['emotion'] == target_labels[0]:
            joy_uttre.append([ (word_tokenize(sentence['utterance']), sentence['emotion']) for sentence in dialogue])
            joy_uttre_start.append( (word_tokenize(dialogue[0]['utterance']), dialogue[0]['emotion']) )
        elif dialogue[0]['emotion'] == target_labels[1]:
            sadness_uttre.append([ (word_tokenize(sentence['utterance']), sentence['emotion']) for sentence in dialogue])
            sadness_uttre_start.append( (word_tokenize(dialogue[0]['utterance']), dialogue[0]['emotion']) )
        elif dialogue[0]['emotion'] == target_labels[2]:
            anger_uttre.append([ (word_tokenize(sentence['utterance']), sentence['emotion']) for sentence in dialogue])
            anger_uttre_start.append( (word_tokenize(dialogue[0]['utterance']), dialogue[0]['emotion']) )                                    
        elif dialogue[0]['emotion'] == target_labels[3]:
            neutral_uttre.append([ (word_tokenize(sentence['utterance']), sentence['emotion']) for sentence in dialogue])
            neutral_uttre_start.append( (word_tokenize(dialogue[0]['utterance']), dialogue[0]['emotion']) )
    new_data = {'joy_uttre': joy_uttre, 'joy_uttre_start': joy_uttre_start, 'sadness_uttre': sadness_uttre, 'sadness_uttre_start': sadness_uttre_start, 'anger_uttre': anger_uttre,
                'anger_uttre_start': anger_uttre_start, 'neutral_uttre': neutral_uttre, 'neutral_uttre_start': neutral_uttre_start}
    for emotion in target_labels:
        words = new_data[emotion+'_uttre']
        words = [x for word in words for x in word]
        wordnet_lemmatizer = WordNetLemmatizer()
        output = list()
        stopWords = set(stopwords.words('english'))
        for word in words:
            sentence_token = word[0]
            label = word[1]
            if label not in target_labels:
                continue
            wordsFiltered = [wordnet_lemmatizer.lemmatize(w) for w in sentence_token if w not in stopWords]
            output.append( (' '.join(wordsFiltered), label) )
        new_data[emotion+'_uttre'] = output


        start_words = new_data[emotion+'_uttre_start']
        output_start = list()
        for word in start_words:
            sentence_token = word[0]
            label = word[1]
            if label not in target_labels:
                continue
            wordsFiltered = [wordnet_lemmatizer.lemmatize(w) for w in sentence_token if w not in stopWords]
            output_start.append( (' '.join(wordsFiltered), label) )
        new_data[emotion+'_uttre_start'] = output_start
    return new_data

def feature_transformation_all(preprocessing_data, method = 'tfidf'):
    sentences = [x[0] for x in preprocessing_data]
    labels = [x[1] for x in preprocessing_data]
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        X = X.toarray()
        pickle.dump(vectorizer, open('./data/tfidf_all.pkl', 'wb'))
    elif method == 'count':
        vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
        X = vectorizer.fit_transform(sentences)
        X = X.toarray()
    return X, labels

def feature_transformation_sep(preprocessing_data, method = 'tfidf'):
    target_labels = ['joy', 'sadness', 'anger', 'neutral']
    features = dict()
    for emotion in target_labels:
        data = preprocessing_data[emotion+'_uttre']
        sentences = [x[0] for x in data]
        labels = [x[1] for x in data]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        X = X.toarray()
        features[emotion] = X
        features[emotion+'_label'] = labels
        
    #data_start = preprocessing_data[emotion + '_uttre_start']
    data_start = [ x for emotion in target_labels for x in preprocessing_data[emotion+'_uttre_start'] ]
    sentences_start = [x[0] for x in data_start]
    labels_start = [x[1] for x in data_start]
    vectorizer_start = TfidfVectorizer()
    X_start = vectorizer_start.fit_transform(sentences_start)
    X_start = X_start.toarray()
    features['start'] = X
    features['label_start'] = labels_start
    pickle.dump( features, open('./tfidf_sep.pkl', 'wb'))
    
def feature_transformation_topic(preprocessing_data, method = 'lsa', topic = 100):
    sentences = [x[0] for x in preprocessing_data]
    sentences = [x.split(' ') for x in sentences]
    labels = [x[1] for x in preprocessing_data]
    dictionary = corpora.Dictionary(sentences)
    dictionary.save('./data/dict_all.pkl')
    #dictionary.load
    corpus = [dictionary.doc2bow(text) for text in sentences]
    corpora.MmCorpus.serialize( './data/corpus_all.pkl' ,corpus) 
    if method == 'lsa':
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics=topic)
        lsi.save('./data/lsi_all.pkl')
        corpus_lsi = lsi[corpus_tfidf]
        features = list()
        for doc in corpus_lsi:
            features.append( [x[1] for x in doc] )
        return features, labels
    elif method == 'lda':
        lda = models.ldamodel.LdaModel(corpus, num_topics = topic)
        lda.save('./data/lda_all.pkl')
        lda_features = [lda.get_document_topics(cor, minimum_probability=0.0) for cor in corpus]
        features = list()
        for lda_feature in lda_features:
            feature = [x[1] for x in lda_feature]
            features.append(feature)
        return features, labels
#LSI mapping
#new_doc = "Human computer interaction"
#vec_bow = dictionary.doc2bow(doc.lower().split())
#vec_lsi = lsi[vec_bow] 

#LDA mapping
#other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
#unseen_doc = other_corpus[0]
#vector = lda[unseen_doc]

def feature_transformation_sep_topic(preprocessing_data, method = 'lsa', topic = 100):
    target_labels = ['joy', 'sadness', 'anger', 'neutral']
    topic_features = dict()
    for emotion in target_labels:
        data = preprocessing_data[emotion+'_uttre']
        sentences = [x[0] for x in data]
        sentences = [x.split(' ') for x in sentences]
        labels = [x[1] for x in data]
        data_start = preprocessing_data[emotion+'_uttre_start']
        sentences_start = [x[0] for x in data_start]
        sentences_start = [x.split(' ') for x in sentences_start]
        labels_start = [x[1] for x in data_start]
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(text) for text in sentences]
        dictionary_start = corpora.Dictionary(sentences_start)
        corpus_start = [dictionary.doc2bow(text) for text in sentences_start]
        if method == 'lsa':
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]
            lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics=topic)
            corpus_lsi = lsi[corpus_tfidf]
            features = list()
            for doc in corpus_lsi:
                features.append( [x[1] for x in doc] )
            tfidf_start = models.TfidfModel(corpus_start)
            corpus_tfidf_start = tfidf_start[corpus_start]
            lsi_start = models.LsiModel(corpus_tfidf_start, num_topics=topic)
            corpus_lsi_start = lsi_start[corpus_tfidf_start]
            features_start = list()
            for doc in corpus_lsi_start:
                features_start.append( [x[1] for x in doc] )        
            topic_features[emotion] = features
            topic_features[emotion+'_label'] = labels
            topic_features[emotion+'_start'] = features_start
            topic_features[emotion+'_label_start'] = labels_start
            
        elif method == 'lda':
            lda = models.ldamodel.LdaModel(corpus, num_topics = topic)
            lda_features = [lda.get_document_topics(cor, minimum_probability=0.0) for cor in corpus]
            features = list()
            for lda_feature in lda_features:
                feature = [x[1] for x in lda_feature]
                features.append(feature)
            lda_start = models.ldamodel.LdaModel(corpus_start, num_topics = topic)
            lda_features_start = [lda.get_document_topics(cor, minimum_probability=0.0) for cor in corpus_start]
            features_start = list()
            for lda_feature in lda_features_start:
                feature = [x[1] for x in lda_feature]
                features_start.append(feature)
            topic_features[emotion] = features
            topic_features[emotion+'_label'] = labels
            topic_features[emotion+'_start'] = features_start
            topic_features[emotion+'_label'] = labels_start
            
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
    
if __name__ == '__main__':
    path = './Friends/friends_train.json'
    #path = './EmotionPush/emotionpush_train.json'
    #preprocessing_data = preprocessing_all(loading_data(path))
    preprocessing_data = preprocessing_sep(loading_data(path))
    #feature_transformation_all(preprocessing_data)
    #feature_transformation_topic(preprocessing_data, method = 'lda')
    feature_transformation_sep(preprocessing_data)
    #feature_transformation_sep_topic(preprocessing_data, method = 'lsa', topic = 10)
    #pipeline_test(preprocessing_data)



