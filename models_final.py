from data_preprocessing import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from collections import Counter
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
import json

class setiment_analysis_final():
    def __init__(self, train, test, _type, method='lsa', topic = 0):
        self.train_path = train
        self.test_path = test
        self.data_type = _type
        self.method = method
        self.topic = topic
        self.training, self.label, self.label_mapping = self.get_training_data()
        self.testing = self.get_testing_data('test')

    def get_training_data(self):
        if self.data_type == 'all':
            preprocessing_data = preprocessing_all(loading_data(self.train_path))
            training, label = feature_transformation_all(preprocessing_data)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, le
        elif self.data_type == 'all_topic':
            preprocessing_data = preprocessing_all(loading_data(self.train_path))
            training, label = feature_transformation_topic(preprocessing_data, method = self.method, topic = self.topic)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, le
        elif self.data_type == 'all_word2vec':
            preprocessing_data = preprocessing_all(loading_data(self.train_path))
            training, label = feature_transformation_word2vec(preprocessing_data, dim = self.topic)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, le

            
    def get_testing_data(self, path_type):
        path = self.test_path
        if self.data_type == 'all':
            preprocessing_data = preprocessing_test(loading_data(path))
            sentences = [x[0] for x in preprocessing_data]
            labels = [x[1] for x in preprocessing_data]
            vectorizer = pickle.load(open('./data/tfidf_all.pkl','rb'))
            testing = vectorizer.transform(sentences)
            testing = testing.toarray()
            return testing
        elif self.data_type == 'all_topic':
            preprocessing_data = preprocessing_test(loading_data(path))
            sentences = [x[0] for x in preprocessing_data]
            dictionary = corpora.Dictionary.load('./data/dict_all.pkl')
            if self.method == 'lsa':                
                lsi = models.LsiModel.load('./data/lsi_all.pkl')    
                features = list()
                for sentence in sentences:
                    vec_bow = dictionary.doc2bow(sentence.split(' '))
                    vec_lsi = lsi[vec_bow]
                    if len(vec_lsi) != self.topic:
                        vec_lsi = [(x,0) for x in range(self.topic)]
                    features.append([x[1] for x in vec_lsi])
                return features
            elif self.method =='lda':
                lda = models.ldamodel.LdaModel.load('./data/lda_all.pkl')
                features = list()
                for sentence in sentences:
                    vec_bow = dictionary.doc2bow(sentence.split(' '))
                    vec_lda = lda[vec_bow]
                    if len(vec_lda) != self.topic:
                        vec_lda = [(x,0) for x in range(self.topic)]
                    features.append([x[1] for x in vec_lda])
                return features
        elif self.data_type == 'all_word2vec':
            preprocessing_data = preprocessing_test(loading_data(path))
            sentences = [x[0] for x in preprocessing_data]
            word2vec_sentences = [x.split(' ') for x in sentences]
            word2vec = models.word2vec.Word2Vec.load('./data/word2vec.pkl')
            tfidf = pickle.load(open('./data/tfidf_word2vec.pkl', 'rb'))
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
            return features
    def svm(self):
        if self.data_type == 'all':
            clf = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
        elif self.data_type == 'all_topic':
            clf = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
        clf.fit(self.training, self.label)
        result = clf.predict(self.testing)
        result = self.label_mapping.inverse_transform(result)
        return result

    def save_json(self, result):
        test_data = json.load(open(self.test_path))
        count = 0
        print (Counter(result))
        for dialogue in test_data:
            for sentence in dialogue:
                sentence['emotion'] = result[count]
                count +=1
        output_path = self.test_path.replace('tigp', 'result')
        with open(output_path, 'w') as f:
            json.dump(test_data, f)
                
if __name__ == '__main__':
    test_path = ['./tigp/DailyDialog_chopped_test.json', './tigp/iemocap_emotion_test.json', './tigp/ubuntu_emotion_test.json']
    for path in test_path:
        #model = setiment_analysis_final(train = './Friends/friends_train.json', test = path, _type = 'all_topic',method = 'lsa', topic = 50)
        model = setiment_analysis_final(train = './EmotionPush/emotionpush_train.json', test = path, _type = 'all')
        result = model.svm()
        model.save_json(result)
        
