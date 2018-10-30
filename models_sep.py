from data_preprocessing import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
import os
from tqdm import tqdm

class setiment_analysis_sep():
    def __init__(self, train, valid, test, _type, method='lsa', topic = 0):
        self.train_path = train
        self.valid_path = valid
        self.test_path = test
        self.data_type = _type
        self.emotions = ['joy', 'sadness', 'anger', 'neutral']
        self.method = method
        self.topic = topic
        self.train_joy, self.train_joy_label, self.train_sadness, self.train_sadness_label, self.train_anger, self.train_anger_label, self.train_neutral, self.train_neutral_label, self.train_start, self.train_start_label, self.label_mapping = self.get_training_data()
        self.test_joy, self.test_joy_label, self.test_sadness, self.test_sadness_label, self.test_anger, self.test_anger_label, self.test_neutral, self.test_neutral_label, self.test_start, self.test_start_label = self.get_testing_data('test')
        self.valid_joy, self.valid_joy_label, self.valid_sadness, self.valid_sadness_label, self.valid_anger, self.valid_anger_label, self.valid_neutral, self.valid_neutral_label, self.valid_start, self.valid_start_label = self.get_testing_data('valid')
    def get_training_data(self):
        if self.data_type == 'sep':
            preprocessing_data = preprocessing_sep(loading_data(self.train_path))
            feature_transformation_sep(preprocessing_data)
            features = pickle.load(open('./data/tfidf_sep.pkl', 'rb'))
            le = LabelEncoder()
            le.fit(self.emotions)
            return features['joy'], le.transform(features['joy_label']), features['sadness'], le.transform(features['sadness_label']), features['anger'], le.transform(features['anger_label']), features['neutral'], le.transform(features['neutral_label']), features['start'], le.transform(features['start_label']), le
        elif self.data_type == 'sep_topic':
            preprocessing_data = preprocessing_sep(loading_data(self.train_path))
            feature_transformation_sep_topic(preprocessing_data, method = self.method, topic = self.topic)
            features = pickle.load(open('./data/sep_%s_%s.pkl' % (self.method, self.topic), 'rb'))
            le = LabelEncoder()
            le.fit(self.emotions)
            return features['joy'], le.transform(features['joy_label']), features['sadness'], le.transform(features['sadness_label']), features['anger'], le.transform(features['anger_label']), features['neutral'], le.transform(features['neutral_label']), features['start'], le.transform(features['start_label']), le
    
        elif self.data_type == 'sep_word2vec':
            preprocessing_data = preprocessing_sep(loading_data(self.train_path))
            feature_transformation_sep_word2vec(preprocessing_data, dim = self.topic)
            features = pickle.load(open('./data/word2vec_sep.pkl', 'rb'))
            le = LabelEncoder()
            le.fit(self.emotions)
            return features['joy'], le.transform(features['joy_label']), features['sadness'], le.transform(features['sadness_label']), features['anger'], le.transform(features['anger_label']), features['neutral'], le.transform(features['neutral_label']), features['start'], le.transform(features['start_label']), le


            
    def get_testing_data(self, path_type):
        if path_type == 'test':
            path = self.test_path
        elif path_type == 'valid':
            path = self.valid_path
        if self.data_type == 'sep':
            features = dict()
            preprocessing_data = preprocessing_sep(loading_data(path))
            for emotion in self.emotions:
                data = preprocessing_data[emotion+'_uttre']
                sentences = [x[0] for x in data]
                labels = [x[1] for x in data]
                vectorizer = pickle.load(open('./data/tfidf_%s.pkl' % emotion ,'rb'))
                testing = vectorizer.transform(sentences)
                testing = testing.toarray()
                labels = self.label_mapping.transform(labels)
                features[emotion] = testing
                features[emotion+'_label'] = labels
                
            data_start = [ x for emotion in self.emotions for x in preprocessing_data[emotion+'_uttre_start'] ]
            sentences_start = [x[0] for x in data_start]
            labels_start = [x[1] for x in data_start]
            vectorizer = pickle.load(open('./data/tfidf_start.pkl','rb'))
            testing = vectorizer.transform(sentences_start)
            testing = testing.toarray()
            labels_start = self.label_mapping.transform(labels_start)
            features['start'] = testing
            features['start_label'] = labels_start            
            return features['joy'], features['joy_label'], features['sadness'], features['sadness_label'], features['anger'], features['anger_label'], features['neutral'], features['neutral_label'], features['start'], features['start_label']
        
        elif self.data_type == 'sep_topic':
            topic_features = dict()
            preprocessing_data = preprocessing_sep(loading_data(path))
            for emotion in self.emotions:
                data = preprocessing_data[emotion+'_uttre']
                sentences = [x[0] for x in data]
                labels = [x[1] for x in data]
                dictionary = corpora.Dictionary.load('./data/dict_sep_%s.pkl' % emotion)
                labels = self.label_mapping.transform(labels)
                if self.method == 'lsa':                
                    lsi = models.LsiModel.load('./data/lsi_sep_%s.pkl' % emotion)    
                    features = list()
                    for sentence in sentences:
                        vec_bow = dictionary.doc2bow(sentence.split(' '))
                        vec_lsi = lsi[vec_bow]
                        if len(vec_lsi) != self.topic:
                            vec_lsi = [(x,0) for x in range(self.topic)]
                        features.append([x[1] for x in vec_lsi])
                    topic_features[emotion] = features
                    topic_features[emotion + '_label'] = labels
                elif self.method =='lda':
                    lda = models.ldamodel.LdaModel.load('./data/lda_sep_%s.pkl' % emotion)
                    features = list()
                    for sentence in sentences:
                        vec_bow = dictionary.doc2bow(sentence.split(' '))
                        vec_lda = lda[vec_bow]
                        if len(vec_lda) != self.topic:
                            vec_lda = [(x,0) for x in range(self.topic)]
                        features.append([x[1] for x in vec_lda])
                    topic_features[emotion] = features
                    topic_features[emotion + '_label'] = labels

            data_start = [ x for emotion in self.emotions for x in preprocessing_data[emotion+'_uttre_start'] ]
            sentences_start = [x[0] for x in data_start]
            labels_start = [x[1] for x in data_start]
            labels_start = self.label_mapping.transform(labels_start)
            if self.method == 'lsa':                
                lsi_start = models.LsiModel.load('./data/lsi_start.pkl')    
                features = list()
                for sentence in sentences_start:
                    vec_bow = dictionary.doc2bow(sentence.split(' '))
                    vec_lsi = lsi[vec_bow]
                    if len(vec_lsi) != self.topic:
                        vec_lsi = [(x,0) for x in range(self.topic)]
                    features.append([x[1] for x in vec_lsi])
                topic_features['start'] = features
                topic_features['start_label'] = labels_start
            elif self.method =='lda':
                lda_start = models.ldamodel.LdaModel.load('./data/lda_start.pkl')
                features = list()
                for sentence in sentences_start:
                    vec_bow = dictionary.doc2bow(sentence.split(' '))
                    vec_lda = lda[vec_bow]
                    if len(vec_lda) != self.topic:
                        vec_lda = [(x,0) for x in range(self.topic)]
                    features.append([x[1] for x in vec_lda])
                topic_features['start'] = features
                topic_features['start_label'] = labels_start
            return topic_features['joy'], topic_features['joy_label'], topic_features['sadness'], topic_features['sadness_label'], topic_features['anger'], topic_features['anger_label'], topic_features['neutral'], topic_features['neutral_label'], topic_features['start'], topic_features['start_label']
        elif self.data_type == 'sep_word2vec':
            w2v_features = dict()
            preprocessing_data = preprocessing_sep(loading_data(path))
            for emotion in self.emotions:
                data = preprocessing_data[emotion+'_uttre']
                sentences = [x[0] for x in data]
                labels = [x[1] for x in data]
                word2vec_sentences = [x.split(' ') for x in sentences]
                labels = self.label_mapping.transform(labels)
                word2vec = models.word2vec.Word2Vec.load('./data/word2vec_%s.pkl' % emotion)
                tfidf = pickle.load(open('./data/tfidf_word2vec_%s.pkl' % emotion, 'rb'))
                features = list()
                for sentence in word2vec_sentences:
                    count = 0
                    a = np.array([ 0 for x in range(self.topic)])
                    for word in sentence:
                        try:
                            word_vec = np.array(word2vec[word])
                            tfidf_value = tfidf[word]
                            count +=1
                            a = a + (word_vec * tfidf_value)
                        except:
                            pass
                    features.append(a/count)
                features = np.nan_to_num(features).tolist()
                w2v_features[emotion] = features
                w2v_features[emotion + '_label'] = labels
                
            data_start = [ x for emotion in self.emotions for x in preprocessing_data[emotion+'_uttre_start'] ]
            sentences_start = [x[0] for x in data_start]
            labels_start = [x[1] for x in data_start]
            word2vec_sentences = [x.split(' ') for x in sentences_start]
            labels_start = self.label_mapping.transform(labels_start)
            word2vec = models.word2vec.Word2Vec.load('./data/word2vec_start.pkl')
            tfidf = pickle.load(open('./data/tfidf_word2vec_start.pkl', 'rb'))
            features_start = list()
            for sentence in word2vec_sentences:
                count = 0
                a = np.array([ 0 for x in range(self.topic)])
                for word in sentence:
                    try:
                        word_vec = np.array(word2vec[word])
                        tfidf_value = tfidf[word]
                        count +=1
                        a = a + (word_vec * tfidf_value)
                    except:
                        pass
                features_start.append(a/count)            
            features_start = np.nan_to_num(features_start).tolist()
            w2v_features['start'] = features_start
            w2v_features['start_label'] = labels_start            
            return w2v_features['joy'], w2v_features['joy_label'], w2v_features['sadness'], w2v_features['sadness_label'], w2v_features['anger'], w2v_features['anger_label'], w2v_features['neutral'], w2v_features['neutral_label'], w2v_features['start'], w2v_features['start_label']


    def svm(self):
        if self.data_type == 'sep':
            clf_joy = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
            clf_sadness = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
            clf_anger = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
            clf_neutral = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
            clf_start = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
        elif self.data_type == 'sep_topic':
            if self.method == 'lsa':
                clf_joy = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
                clf_sadness = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
                clf_anger = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
                clf_neutral = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
                clf_start = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
            elif self.method == 'lda':
                clf_joy = SVC(C = 0.001, kernel = 'rbf', gamma = 0.01)
                clf_sadness = SVC(C = 0.001, kernel = 'rbf', gamma = 0.01)
                clf_anger = SVC(C = 0.001, kernel = 'rbf', gamma = 0.01)
                clf_neutral = SVC(C = 0.001, kernel = 'rbf', gamma = 0.01)
                clf_start = SVC(C = 0.001, kernel = 'rbf', gamma = 0.01)
        elif self.data_type == 'sep_word2vec':
            clf_joy = SVC()
            clf_sadness = SVC()
            clf_anger = SVC()
            clf_neutral = SVC()
            clf_start = SVC()
            
        clf_start.fit(self.train_start, self.train_start_label)
        clf_joy.fit(self.train_joy, self.train_joy_label)
        clf_sadness.fit(self.train_sadness, self.train_sadness_label)
        clf_anger.fit(self.train_anger, self.train_anger_label)
        clf_neutral.fit(self.train_neutral, self.train_neutral_label)

        
        test_score_start = clf_start.score(self.test_start, self.test_start_label)
        test_score_joy = clf_joy.score(self.test_joy, self.test_joy_label)
        test_score_sadness = clf_sadness.score(self.test_sadness, self.test_sadness_label)
        test_score_anger = clf_anger.score(self.test_anger, self.test_anger_label)
        test_score_neutral = clf_neutral.score(self.test_neutral, self.test_neutral_label)
        
        valid_score_start = clf_start.score(self.valid_start, self.valid_start_label)
        valid_score_joy = clf_joy.score(self.valid_joy, self.valid_joy_label)
        valid_score_sadness = clf_sadness.score(self.valid_sadness, self.valid_sadness_label)
        valid_score_anger = clf_anger.score(self.valid_anger, self.valid_anger_label)
        valid_score_neutral = clf_neutral.score(self.valid_neutral, self.valid_neutral_label)
        
        print ('SVM %s %s' % (self.data_type, self.method))
        print ('testing accuracy: ', test_score_start, test_score_joy, test_score_sadness, test_score_anger, test_score_neutral)
        print ('validing accuracy: ', valid_score_start, valid_score_joy, valid_score_sadness, valid_score_anger, valid_score_neutral)
        

    def rf(self):
        if self.data_type == 'sep':
            clf_start = RandomForestClassifier(n_estimators= 101, max_features = 'auto')
            clf_joy = RandomForestClassifier(n_estimators= 101, max_features = 'auto')
            clf_sadness = RandomForestClassifier(n_estimators= 101, max_features = 'auto')
            clf_anger = RandomForestClassifier(n_estimators= 101, max_features = 'auto')
            clf_neutral = RandomForestClassifier(n_estimators= 101, max_features = 'auto')
        elif self.data_type == 'sep_topic':
            if self.method == 'lsa':
                clf_start = RandomForestClassifier(n_estimators= 51, max_features = 'auto')
                clf_joy = RandomForestClassifier(n_estimators= 51, max_features = 'auto')
                clf_sadness = RandomForestClassifier(n_estimators= 51, max_features = 'auto')
                clf_anger = RandomForestClassifier(n_estimators= 51, max_features = 'auto')
                clf_neutral = RandomForestClassifier(n_estimators= 51, max_features = 'auto')
            elif self.method == 'lda':
                clf_start = RandomForestClassifier(n_estimators= 201, max_features = 'log2')
                clf_joy = RandomForestClassifier(n_estimators= 201, max_features = 'log2')
                clf_sadness = RandomForestClassifier(n_estimators= 201, max_features = 'log2')
                clf_anger = RandomForestClassifier(n_estimators= 201, max_features = 'log2')
                clf_neutral = RandomForestClassifier(n_estimators= 201, max_features = 'log2')
        elif self.data_type == 'sep_word2vec':
            clf_start = RandomForestClassifier(n_estimators= 101, max_features = 'log2')
            clf_joy = RandomForestClassifier(n_estimators= 101, max_features = 'log2')
            clf_sadness = RandomForestClassifier(n_estimators= 101, max_features = 'log2')
            clf_anger = RandomForestClassifier(n_estimators= 101, max_features = 'log2')
            clf_neutral = RandomForestClassifier(n_estimators= 101, max_features = 'log2')
        
        clf_start.fit(self.train_start, self.train_start_label)
        clf_joy.fit(self.train_joy, self.train_joy_label)
        clf_sadness.fit(self.train_sadness, self.train_sadness_label)
        clf_anger.fit(self.train_anger, self.train_anger_label)
        clf_neutral.fit(self.train_neutral, self.train_neutral_label)

        test_score_start = clf_start.score(self.test_start, self.test_start_label)
        test_score_joy = clf_joy.score(self.test_joy, self.test_joy_label)
        test_score_sadness = clf_sadness.score(self.test_sadness, self.test_sadness_label)
        test_score_anger = clf_anger.score(self.test_anger, self.test_anger_label)
        test_score_neutral = clf_neutral.score(self.test_neutral, self.test_neutral_label)
        
        valid_score_start = clf_start.score(self.valid_start, self.valid_start_label)
        valid_score_joy = clf_joy.score(self.valid_joy, self.valid_joy_label)
        valid_score_sadness = clf_sadness.score(self.valid_sadness, self.valid_sadness_label)
        valid_score_anger = clf_anger.score(self.valid_anger, self.valid_anger_label)
        valid_score_neutral = clf_neutral.score(self.valid_neutral, self.valid_neutral_label)
        
        print ('Random Forest %s %s' % (self.data_type, self.method))
        print ('testing accuracy: ', test_score_start, test_score_joy, test_score_sadness, test_score_anger, test_score_neutral)
        print ('validing accuracy: ', valid_score_start, valid_score_joy, valid_score_sadness, valid_score_anger, valid_score_neutral)
    
    def adaboost(self):
        if self.data_type == 'all':
            clf = AdaBoostClassifier(learning_rate = 0.001, n_estimators = 10)
        elif self.data_type == 'all_topic':
            if self.method == 'lsa':
                clf = AdaBoostClassifier(learning_rate = 0.5, n_estimators = 80)
            elif self.method == 'lda':
                clf = AdaBoostClassifier(learning_rate = 0.5, n_estimators = 70)
        elif self.data_type == 'all_word2vec':
            clf = AdaBoostClassifier()
            
        clf.fit(self.training, self.label)
        test_score = clf.score(self.testing, self.test_label)
        valid_score = clf.score(self.valid, self.valid_label)
        print ('Adaboost %s %s' % (self.data_type, self.method))
        print ('testing accuracy: ', test_score)
        print ('valid accuracy: ', valid_score)

    def gbdt(self):
        #clf = GradientBoostingClassifier()
        if self.data_type == 'all':
            clf = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 60)
        elif self.data_type == 'all_topic':
            if self.method == 'lsa':
                clf = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 90)
            elif self.method == 'lda':
                clf = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 80)
        elif self.data_type == 'all_word2vec':
            clf = GradientBoostingClassifier()
            
        clf.fit(self.training, self.label)
        test_score = clf.score(self.testing, self.test_label)
        valid_score = clf.score(self.valid, self.valid_label)
        print ('GBDT %s %s' % (self.data_type, self.method))
        print ('testing accuracy: ', test_score)
        print ('valid accuracy: ', valid_score)

def main():
    start_time = time.time()
    model = setiment_analysis_sep(train = './Friends/friends_train.json', valid = './Friends/friends_dev.json', test = './Friends/friends_test.json',
                              _type = 'sep_topic',method = 'lda', topic = 10)
    #print (model.train_joy[0], model.test_joy_label[0])
    #print (model.test_joy[0], model.test_joy_label[0])
    #print (len(model.train_joy), len(model.train_joy_label), len(model.test_joy), len(model.test_joy_label), len(model.valid_joy), len(model.valid_joy_label), len(model.train_start), len(model.train_start_label))
    #print ( len(model.train_start), len(model.train_start_label), len(model.test_start), len(model.test_start_label), len(model.valid_start), len(model.valid_start_label) )
                    
    #model.svm()
    model.rf()
    #model.adaboost()
    #model.gbdt()
    #print (time.time() - start_time)

if __name__ == '__main__':
    main()

