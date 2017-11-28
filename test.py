%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import boto
import tflearn
import numpy as np
# import seaborn as sns
from tflearn.data_utils import to_categorical, pad_sequences, load_csv
from string import punctuation
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Defaults for pandas
pd.set_option('display.max_colwidth',1000)

# Load data from S3
original_review = pd.read_csv('s3a://amazon-fine-food-dataset/Reviews.csv')

# Analysis of data
s = pd.Series(original_review.Score)
s.describe()
s_counts = s.value_counts()
plt.bar(s_counts.index.tolist(), s_counts.values, align='center')
plt.title('Score Distribution')

# Data Preprocessing
temp = original_review.loc[:,["Text","Score"]]
temp = temp[temp.Score != 3]

def partition(x):
    if x < 3:
        return 0
    return 1

# Partition the data by 1-3 and 4-5 
Score = temp['Score']
Score = Score.map(partition)
Text = temp['Text']
Score.head()

Score_counts = pd.Series(Score).value_counts()
plt.bar(Score_counts.index.tolist(), Score_counts.values, align='center')
plt.title('Sentiment Positive vs Negative Distribution')

# Removes punctuations in texts
reviews = temp.Text.values
labels = np.array(Score.values)
reviews_cleaned = []

for i in range(len(reviews)):
    reviews_cleaned.append(''.join([c.lower() for c in reviews[i] if c not in punctuation]))

# new code start here

X_train, X_test, y_train, y_test = train_test_split(reviews_cleaned, Score, test_size=0, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
# Don't understand this token pattern
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')
vect.fit(X_train)
vocab = vect.vocabulary_


def convert_X_to_X_word_ids(X):
    return X.apply(lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab])

X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids = convert_X_to_X_word_ids(X_test)

# See diff b/w X_train and X_train_words_ids
X_train.head()
X_train_words_ids.head()

X_train_word_ids.shape
X_test_word_ids.shape

max_length = 20
X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=max_length, value=0)
X_test_padded_seqs = pad_sequences(X_test_word_ids, maxlen=max_length, value=0)

X_train_padded_seqs.shape
X_test_padded_seqs.shape

pd.DataFrame(X_train_padded_seqs).head()
pd.DataFrame(X_test_padded_seqs).head()

# Convert y to labels to vectors

unique_y_labels = list(y_train.value_counts().index)
unique_y_labels
len(unique_y_labels)


# need to see unique_y_labels here our data here is only binary, the rest of code was changing the 
# 10 categorical labels to vectors

size_of_each_vector = X_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)





'''
# Store all the text from each review in a text variable
text = ' '.join(reviews_cleaned)

# List all the vocabulary contained in the reviews
vocabulary = set(text.split(' '))

# Map each word to an integer
vocabulary_to_int = {word:i for i,word in enumerate(vocabulary,0)}

def reviews_to_integers(reviews):
    reviews_to_int = []
    for i in range(len(reviews)):
        to_int = [vocabulary_to_int[word] for word in reviews[i].split()]
        reviews_to_int.append(to_int)
    return reviews_to_int

reviews_to_int = reviews_to_integers(reviews_cleaned)

# Set max length of text and add padding if necessary
max_length = 10
print(len(reviews_to_int))

features = np.zeros(shape=(len(reviews_to_int),max_length),dtype=int)
for i in range(len(reviews_to_int)):
    nb_words = len(reviews_to_int[i])
    features[i] = [0]*(max_length -nb_words) + reviews_to_int[i][:max_length]

vocab_size = len(features)
print(vocab_size)
'''

'''

Old code
# Randomly shuffle the data into train and test groups
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5435)
splitter = sss.split(features, labels)
train_index, validation_index = next(splitter)
test_index = validation_index[:int(len(validation_index)/2)]
validation_index = validation_index[int(len(validation_index)/2):]

# Create train, validation, and test sets
train_x, train_y = features[train_index], labels[train_index]
val_x, val_y = features[test_index], labels[test_index]
test_x, test_y = features[validation_index], labels[validation_index]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
'''

# Create Recurrant Neural Network Model
# input dim = vocab size
# fully connected size = no of unique y labels
net = tflearn.input_data([None, max_length])
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)

# Train the RNN with train and test data
model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True)

'''
# Manually save the model
# need to make folder SavedModels
model.save('SavedModels/model.tfl')
print(colored('Model Saved!', 'red'))

model.load('SavedModels/model.tfl')
print(colored('Model Loaded!', 'red'))

from sklearn import metrics
pred_classes = [np.argmax(i) for i in model.predict(X_test_padded_seqs)]
true_classes = [np.argmax(i) for i in y_test]

print('\nRNN Classifier\'s Accuracy: %0.5f\n' % metrics.accuracy_score(true_classes, pred_classes))
