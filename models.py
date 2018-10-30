from data_preprocessing import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import time

class setiment_analysis():
    def __init__(self, train, valid, test, _type, method='lsa', topic = 0):
        self.train_path = train
        self.valid_path = valid
        self.test_path = test
        self.data_type = _type
        self.method = method
        self.topic = topic
        self.training, self.label, self.label_mapping = self.get_training_data()
        self.testing, self.test_label = self.get_testing_data()

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
            return training, label
        elif self.data_type == 'all_word2vec':
            preprocessing_data = preprocessing_all(loading_data(self.train_path))
            training, label = feature_transformation_word2vec(preprocessing_data, dim = self.topic)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, le
        elif self.data_type == 'sep':
            preprocessing_data = preprocessing_sep(loading_data(self.train_path))
            training, label = feature_transofrmation_sep(preprocessing_data)

    def get_testing_data(self):
        if self.data_type == 'all':
            preprocessing_data = preprocessing_all(loading_data(self.test_path))
            sentences = [x[0] for x in preprocessing_data]
            labels = [x[1] for x in preprocessing_data]
            vectorizer = pickle.load(open('./data/tfidf_all.pkl','rb'))
            testing = vectorizer.transform(sentences)
            labels = self.label_mapping.transform(labels)
            return testing, labels
            
        
        pass
        

    def svm(self):
        #clf = SVC(C = 1.25, kernel = 'rbf', gamma = 0.75)
        pass

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
        #clf = RandomForestClassifier(n_estimators= 101)
        pass
    
    def rf_tuning(self):
        tuned_parameters = [ {'n_estimators': [11, 51, 101, 151, 201], 'max_features': ['auto', 'log2']} ]
        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

    def adaboost(self):
        #clf = AdaBoostClassifier()
        pass

    def adaboost_tuning(self):
        tuned_parameters = [ {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]} ]
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

    def gbdt(self):
        #clf = GradientBoostingClassifier()
        pass

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
                              _type = 'all',method = ' ', topic = 100)
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
        print ('svm')
        model.svm()
        print ('random forest')
        model.rf()
        print ('adaboost')
        model.adaboost()
        print ('gbdt')
        model.gbdt()
        print (time.time() - start_time)

if __name__ == '__main__':
    main(tuning = 0)

