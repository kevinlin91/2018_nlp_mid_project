from data_preprocessing import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
import time

class setiment_analysis():
    def __init__(self, path, _type, method, topic = 0):
        self.path = path
        self.data_type = _type
        self.method = method
        self.topic = topic
        self.training, self.label, self.label_mapping = self.get_data()


    def get_data(self):
        if self.data_type == 'all':
            preprocessing_data = preprocessing_all(loading_data(self.path))
            training, label = feature_transformation_all(preprocessing_data)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, label_mapping
        elif self.data_type == 'all_topic':
            preprocessing_data = preprocessing_all(loading_data(self.path))
            training, label = feature_transformation_topic(preprocessing_data, method = self.method, topic = self.topic)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, label_mapping
        elif self.data_type == 'sep':
            preprocessing_data = preprocessing_sep(loading_data(self.path))
            training, label = feature_transofrmation_sep(preprocessing_data)

    def svm(self):
        #clf = SVC(C = 1.25, kernel = 'rbf', gamma = 0.75)
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
        tuned_parameters = [ {'n_estimators': [11, 51, 101, 151, 201], 'max_features': ['auto', 'log2']} ]
        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

    def adaboost(self):
        #clf = AdaBoostClassifier()
        tuned_parameters = [ {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]} ]
        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

    def gbdt(self):
        #clf = GradientBoostingClassifier()
        tuned_parameters = [ {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]} ]
        clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

def main():
    start_time = time.time()
    model = setiment_analysis('./Friends/friends_train.json', 'all_topic', 'lsa', 100)
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
    main()

