from data_preprocessing import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import Counter


class setiment_analysis():
    def __init__(self, path, _type):
        self.path = path
        self.data_type = _type
        self.training, self.label, self.label_mapping = self.get_data()
        

    def get_data(self):        
        if self.data_type == 'all':
            preprocessing_data = preprocessing_all(loading_data(self.path))
            training, label = feature_transformation_all(preprocessing_data)
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, label_mapping
        elif self.data_type == 'sep':
            preprocessing_data = preprocessing_sep(loading_data(self.path))
            training, label = feature_transofrmation_sep(preprocessing_data)

    def svm(self):
        clf = SVC(C = 1.25, kernel = 'rbf', gamma = 0.75)
        tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
        ]
        #clf = GridSearchCV(SVC(), tuned_parameters, scoring = 'accuracy', cv=2)
        clf.fit(self.training, self.label)
        #print("Best parameters set found on development set:")
        #print()
        #print(clf.best_params_)
        #cross_val_score(clf, self.training, self.label, cv=2)
        #print (corss_val_score)
            
                                



def main():
    model = setiment_analysis('./Friends/friends_train.json', 'all')
    model.svm()


if __name__ == '__main__':
    main()

