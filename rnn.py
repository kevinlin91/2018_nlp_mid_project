from data_preprocessing import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#friend max word length: 59
#emotionpush max word length: 179

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
            #le = LabelEncoder()
            ohe = OneHotEncoder()
            #label = ohe.fit_transform(le.fit_transform(label).reshape(-1, 1))
            label = ohe.fit_transform(np.array(label).reshape(-1, 1)).toarray()
            label_mapping = list(ohe.categories_[0])
            return training, label, label_mapping
        elif self.data_type == 'sep':
            preprocessing_data = preprocessing_sep(loading_data(self.path))
            training, label = feature_transofrmation_sep(preprocessing_data)
    def get_features(self):
        max_features = 100
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(self.training)
        X = tokenizer.texts_to_sequences(self.training)
        X = pad_sequences(X)

        embed_dim = 32
        lstm_out = 64

        model = Sequential()
        model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(4,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

        X_train, X_test, Y_train, Y_test = train_test_split(X,np.array(self.label), test_size = 0.33, random_state = 42)
        batch_size = 32
        model.fit(X_train, Y_train, epochs = 100, batch_size=batch_size, verbose = 2)
        score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
        model.save('rnn_model.h5')
def main():
    model = rnn('./Friends/friends_train.json', 'all')
    model.get_features()
if __name__ == '__main__':
    main()
