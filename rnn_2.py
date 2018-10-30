from data_preprocessing import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#friend max word length: 59
#emotionpush max word length: 179

class rnn():
    def __init__(self, train, valid, test, _type):
        self.train_path = train
        self.valid_path = valid
        self.test_path = test
        self.data_type = _type
        self.max_features = 100
        self.emotion = ['joy', 'sadness', 'anger', 'neutral']
        self.training, self.label, self.label_mapping, self.tokenizer, self.length = self.get_training_data()
        self.testing, self.test_label = self.get_testing_data('test')
        self.valid, self.valid_label = self.get_testing_data('valid')
        
    def get_training_data(self):
        data = loading_data(self.train_path)
        preprocessing_data = [ (sentence['utterance'], sentence['emotion']) for dialogue in data for sentence in dialogue if sentence['emotion'] in self.emotion]
        training = [x[0] for x in preprocessing_data]
        label = [x[1] for x in preprocessing_data]
        ohe = OneHotEncoder()
        label = ohe.fit_transform(np.array(label).reshape(-1, 1)).toarray()        
        tokenizer = Tokenizer(num_words=self.max_features, split=' ')
        tokenizer.fit_on_texts(training)
        X = tokenizer.texts_to_sequences(training)
        X = pad_sequences(X)
        
        return X, label, ohe, tokenizer, len(X[0])
        
    def get_testing_data(self, path_type):
        if path_type == 'test':
            preprocessing_data = loading_data(self.test_path)
        elif path_type == 'valid':
            preprocessing_data = loading_data(self.valid_path)
        preprocessing_data = [ (sentence['utterance'], sentence['emotion']) for dialogue in preprocessing_data for sentence in dialogue if sentence['emotion'] in self.emotion ]
        data = [x[0] for x in preprocessing_data]
        label = [x[1] for x in preprocessing_data]
        label = self.label_mapping.transform(np.array(label).reshape(-1, 1)).toarray()
        X = self.tokenizer.texts_to_sequences(data)
        X = pad_sequences(X, maxlen = self.length)
        return X, label

        
        
    def rnn_model(self):
        
        embed_dim = 32
        lstm_out = 64

        model = Sequential()
        model.add(Embedding(60, embed_dim,input_length = self.length))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(4,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

        X_train = self.training
        Y_train = self.label
        
        batch_size = 32
        model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 2)
        score_test, acc_test = model.evaluate(self.testing, self.test_label, verbose = 2, batch_size = batch_size)
        score_valid, acc_valid = model.evaluate(self.valid, self.valid_label, verbose = 2, batch_size = batch_size)
        print ("test")
        print (score_test, acc_test)
        print ("valid")
        print (score_valid, acc_valid)
        
        model.save('rnn_model.h5')
        
def main():
    model = rnn(train = './Friends/friends_train.json', valid = './Friends/friends_dev.json', test = './Friends/friends_test.json', _type = 'all')
    model.rnn_model()
if __name__ == '__main__':
    main()
