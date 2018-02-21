import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim import models
import nltk
from sklearn.externals import joblib
from nltk.tag import pos_tag


def lda_train(raw):
    stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    text_array = []
    for i in range(len(raw)):
        text = raw[i].lower()
        text = text.replace('\r\n', ' ')
        text = re.sub("[^a-z0-9]", " ", text)
        # Tokenization segments a document into its atomic elements.
        words = text.split()
        # Stop words
        # Certain parts of English speech, like (for, or) or the word the are meaningless to a topic model.
        # These terms are called stop words and need to be removed from our token list.
        words = [j for j in words if j not in stop]
        tokenized = nltk.word_tokenize(text)
        tagged_sent = pos_tag(words)
        words = [word for word,pos in tagged_sent if pos == 'NN']
        # Stemming words is another common NLP technique to reduce topically similar words to their root.
        # stemming reduces those terms to stem. This is important for topic modeling, which would otherwise view those terms as separate entities and reduce their importance in the model.
        #words = [p_stemmer.stem(s) for s in words]
        text_array.append(words)
    dictionary = corpora.Dictionary(text_array)
    dictionary.save('dictionary.dic')
    corpus = [dictionary.doc2bow(text) for text in text_array]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=15, id2word=dictionary, passes=20)
    filename = 'finalized_model_15.sav'
    joblib.dump(ldamodel, filename)
    print(ldamodel.print_topics(num_topics=15, num_words=6))
    return ldamodel,dictionary




if __name__ == '__main__':
    state = 'NC'
    review = pd.read_csv("./data/" + str(state) + "review.csv")
    raw = review.text.tolist()
    lda,dict = lda_train(raw)
    filename = 'finalized_model_15.sav'
    loaded_model = joblib.load(filename)
    dict = corpora.Dictionary()
    dict = dict.load('dictionary.dic')
    print(loaded_model.print_topics(num_topics=15, num_words=10))








