from data_preprocessing import *
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
class rnn():
    def __init__(self, path, _type):
        self.path = path
        self.data_type = _type
        self.training, self.label, self.label_mapping = self.get_data()
        
    def get_data(self):        
        if self.data_type == 'all':
            preprocessing_data = preprocessing_all(loading_data(self.path))
            training = [x[0] for x in preprocessing_data]
            label = [x[1] for x in preprocessing_data]
            le = LabelEncoder()
            label = le.fit_transform(label)
            label_mapping = le.classes_
            return training, label, label_mapping
        elif self.data_type == 'sep':
            preprocessing_data = preprocessing_sep(loading_data(self.path))
            training, label = feature_transofrmation_sep(preprocessing_data)
    def get_features(self):
        max_fatures = 2000
        tokenizer = Tokenizer(num_words=max_fatures, split=' ')
        tokenizer.fit_on_texts(data['text'].values)
        X = tokenizer.texts_to_sequences(data['text'].values)
        X = pad_sequences(X)
    
        
def main():
    model = rnn('./Friends/friends_train.json', 'all')

if __name__ == '__main__':
    main()
