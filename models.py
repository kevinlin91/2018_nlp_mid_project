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

class setiment_analysis():
    def __init__(self, train, valid, test, _type, method='lsa', topic = 0):
        self.train_path = train
        self.valid_path = valid
        self.test_path = test
        self.data_type = _type
        self.method = method
        self.topic = topic
        self.training, self.label, self.label_mapping = self.get_training_data()
        self.testing, self.test_label = self.get_testing_data('test')
        self.valid, self.valid_label = self.get_testing_data('valid')

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
        if path_type == 'test':
            path = self.test_path
        elif path_type == 'valid':
            path = self.valid_path
        if self.data_type == 'all':
            preprocessing_data = preprocessing_all(loading_data(path))
            sentences = [x[0] for x in preprocessing_data]
            labels = [x[1] for x in preprocessing_data]
            vectorizer = pickle.load(open('./data/tfidf_all.pkl','rb'))
            testing = vectorizer.transform(sentences)
            testing = testing.toarray()
            labels = self.label_mapping.transform(labels)
            return testing, labels
        elif self.data_type == 'all_topic':
            preprocessing_data = preprocessing_all(loading_data(path))
            sentences = [x[0] for x in preprocessing_data]
            dictionary = corpora.Dictionary.load('./data/dict_all.pkl')
            labels = [x[1] for x in preprocessing_data]
            labels = self.label_mapping.transform(labels)
            if self.method == 'lsa':                
                lsi = models.LsiModel.load('./data/lsi_all.pkl')    
                features = list()
                for sentence in sentences:
                    vec_bow = dictionary.doc2bow(sentence.split(' '))
                    vec_lsi = lsi[vec_bow]
                    if len(vec_lsi) != self.topic:
                        vec_lsi = [(x,0) for x in range(self.topic)]
                    features.append([x[1] for x in vec_lsi])
                return features, labels
            elif self.method =='lda':
                lda = models.ldamodel.LdaModel.load('./data/lda_all.pkl')
                features = list()
                for sentence in sentences:
                    vec_bow = dictionary.doc2bow(sentence.split(' '))
                    vec_lda = lda[vec_bow]
                    if len(vec_lda) != self.topic:
                        vec_lda = [(x,0) for x in range(self.topic)]
                    features.append([x[1] for x in vec_lda])
                return features, labels
        elif self.data_type == 'all_word2vec':
            preprocessing_data = preprocessing_all(loading_data(path))
            sentences = [x[0] for x in preprocessing_data]
            word2vec_sentences = [x.split(' ') for x in sentences]
            labels = [x[1] for x in preprocessing_data]
            labels = self.label_mapping.transform(labels)
            word2vec = models.word2vec.Word2Vec.load('./data/word2vec.pkl')
            tfidf = pickle.load(open('./data/tfidf_word2vec.pkl', 'rb'))
            features = list()
            for sentence in word2vec_sentences[11:13]:
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
            return features, labels

    def svm(self):
        if self.data_type == 'all':
            clf = SVC(C = 50, kernel = 'rbf', gamma = 0.01)
        elif self.data_type == 'all_topic':
            if self.method == 'lsa':
                clf = SVC(C = 1000, kernel = 'rbf', gamma = 0.01)
            elif self.method == 'lda':
                clf = SVC(C = 0.001, kernel = 'rbf', gamma = 0.01)
        elif self.data_type == 'all_word2vec':
            clf = SVC()
            
        clf.fit(self.training, self.label)
        test_score = clf.score(self.testing, self.test_label)
        valid_score = clf.score(self.valid, self.valid_label)
        print ('SVM %s %s' % (self.data_type, self.method))
        print ('testing accuracy: ', test_score)
        print ('valid accuracy: ', valid_score)
        

    def svm_tuning(self):
        tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
        ]
        clf = GridSearchCV(SVC(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

    def rf(self):
        if self.data_type == 'all':
            clf = RandomForestClassifier(n_estimators= 101, max_features = 'auto')
        elif self.data_type == 'all_topic':
            if self.method == 'lsa':
                clf = RandomForestClassifier(n_estimators= 51, max_features = 'auto')
            elif self.method == 'lda':
                clf = RandomForestClassifier(n_estimators= 201, max_features = 'log2')
        elif self.data_type == 'all_word2vec':
            clf = RandomForestClassifier(n_estimators= 101, max_features = 'log2')
            
        clf.fit(self.training, self.label)
        testing_data = np.array(self.testing)
        test_score = clf.score(self.testing , self.test_label)
        valid_score = clf.score(self.valid, self.valid_label)
        print ('Random Forest %s %s' % (self.data_type, self.method))
        print ('testing accuracy: ', test_score)
        print ('valid accuracy: ', valid_score)
    
    def rf_tuning(self):
        tuned_parameters = [ {'n_estimators': [11, 51, 101, 151, 201], 'max_features': ['auto', 'log2']} ]
        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

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

    def adaboost_tuning(self):
        tuned_parameters = [ {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]} ]
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

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

    def gbdt_tuning(self):
        tuned_parameters = [ {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]} ]
        clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

def main(tuning = 0):
    start_time = time.time()
    model = setiment_analysis(train = './Friends/friends_train.json', valid = './Friends/friends_dev.json', test = './Friends/friends_test.json',
                              _type = 'all_topic',method = 'lda', topic = 10)
    #print (model.testing[0], model.test_label[0])
    if tuning == 1:
        print ('svm')
        model.svm_tuning()
        print ('random forest')
        model.rf_tuning()
        print ('adaboost')
        model.adaboost_tuning()
        print ('gbdt')
        model.gbdt_tuning()
        print (time.time() - start_time)
    elif tuning == 0:
        model.svm()
        model.rf()
        model.adaboost()
        model.gbdt()
        print (time.time() - start_time)

if __name__ == '__main__':
    main(tuning = 0)

