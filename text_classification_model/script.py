import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

filepath_dict = { "yelp" : "sentiment_sentences/yelp_labelled.txt",
                  "amazon" : "sentiment_sentences/amazon_cells_labelled.txt",
                  "imdb"   : "sentiment_sentences/imdb_labelled.txt"
                }

#creating an empty list 
df_list = []
#we are creating a loop asking it to read the items as a csv file 
# and creating a column known as sentence and label, using the sentence and label in the dataset 
# creating another column called source using source 
for source, filepath in filepath_dict.items():
    sentences = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    sentences['source'] = source 
    df_list.append(sentences) # appending it to the empty list 
 
#concatenating the list and creating a dataframe
sentences = pd.concat(df_list)

# we are going to create a loop that will go through the source and predict all the scores

for source in sentences["source"].unique():
    sentence_source = sentences[sentences["source"] == source]
    sentence_words = sentence_source['sentence'].values
    label = sentence_source['label'].values

    # split into trains and test 
    # test size always need to be smaller than train
    # you always need a large validation set (train set)
    sentence_words_train, sentence_words_test, label_train, label_test = train_test_split(sentence_words, label, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    # to create a bag of words which we are going to use to train the dataset 
    vectorizer.fit(sentence_words_train)
    # then we transform the test and train into a sparse matrix 
    X_train = vectorizer.transform(sentence_words_train)
    X_test = vectorizer.transform(sentence_words_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, label_train)
    score = classifier.score(X_test, label_test)
    print("Accuracy for {}: {:.4f}".format(source, score))

