import pandas as pd
import boto
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, load_csv

pd.set_option('display.max_colwidth',1000)
original_review = pd.read_csv('s3a://amazon-fine-food-dataset/Reviews.csv')
temp = original_review.loc[:,["Text","Score"]]
temp = temp[temp.Score != 3]
def partition(x):
    if x < 3:
        return 0
    return 1

Score = temp['Score']
Score = Score.map(partition)
Text = temp['Text']
Score.head()


reviews = temp.Text.values
import numpy as np
labels = np.array(Score.values)
from string import punctuation
reviews_cleaned = []
for i in range(len(reviews)):
    reviews_cleaned.append(''.join([c.lower() for c in reviews[i] if c not in punctuation]))
   


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






max_length = 10
print(len(reviews_to_int))

features = np.zeros(shape=(len(reviews_to_int),max_length),dtype=int)
for i in range(len(reviews_to_int)):
    nb_words = len(reviews_to_int[i])
    features[i] = [0]*(max_length -nb_words) + reviews_to_int[i][:max_length]
print(len(features))



from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5435)

splitter = sss.split(features, labels)
train_index, validation_index = next(splitter)
test_index = validation_index[:int(len(validation_index)/2)]
validation_index = validation_index[int(len(validation_index)/2):]

train_x, train_y = features[train_index], labels[train_index]
val_x, val_y = features[test_index], labels[test_index]
test_x, test_y = features[validation_index], labels[validation_index]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))


net = tflearn.input_data([None, max_length])
net = tflearn.embedding(net, input_dim=240772, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 1, activation='linear')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True)

