

###########################################################################################
# # from vocarum

from os import listdir
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import string  as string


##########################################################################################
# remember to change for submission
##########################################################################################

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation
stopwords_path = "stopwords.en.txt"

#train_path = 'C:/barrett/Edx/AI/Assign 5/train/'
#test_path  = 'C:/barrett/Edx/AI/Assign 5/imdb_te.csv'
#stopwords_path = "C:/barrett/Edx/AI/Assign 5/stopwords.en.txt"
###############################################################################################
# stop words
sw = open(stopwords_path, "r", encoding="utf8")
stopwords = sw.read()
#print (stopwords)
sw.close()
stopwords = stopwords.lower().rstrip().split("\n")
#print (stopwords)

################################################################################################
# functions to process text files into usable single file
def process_words(string_set):
    words = string_set
    replace = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = words.translate(replace)
    words = words.replace("<br />", " ").rstrip().lower().split()
    words = [word for word in words if word not in stopwords]
    words = ' '.join(words)
    return words


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    fo = open(name, "w", encoding='utf8')
    fo.write("row_number,text,polarity\n")

    
    ################# more complex than needed, 
    #in future use scandir but likely not supported in vocareum
    #the_path = 'C:/barrett/Edx/AI/Assign 5/train/pos'
    #def get_files(the_path):
    #    f = []
    #    for (dirpath, dirnames, filenames) in walk(the_path):
    #        f.extend(filenames)
    #        break
    # #   return f
    #a = get_files(the_path)
    #######################
  
    count = 0
    for text in listdir(inpath + "pos"):
        file_in = open(inpath + "pos/" + text, "r", encoding='utf8')
        #print(inpath + "pos/" + '    '+text)
        text = process_words(file_in.read())
        fo.write(str(count) + "," + text + ",1" + "\n")
        count += 1
        file_in.close()

    for text in listdir(inpath + "neg"):
        file_in = open(inpath + "neg/" + text, "r", encoding='utf8')
        text = process_words(file_in.read())
        fo.write(str(count) + "," + text + ",0" + "\n")
        count += 1
        file_in.close()

#########################################################################################################
if "__main__" == __name__:

    imdb_data_preprocess(train_path)

    training = pd.read_csv("imdb_tr.csv")
    test_set = pd.read_csv(test_path, encoding="ISO-8859-1")
    test_set['text'] = test_set['text'].apply(process_words)

    # Unigram
    count_vec = CountVectorizer(stop_words=stopwords)
    transform_train = count_vec.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1") #, verbose =1)  # alpha Defaults to 0.0001
    classifier.fit(transform_train, training['polarity'])

    # Output Unigram
    transform_test = count_vec.transform(test_set['text'])
    results = classifier.predict(transform_test)
    with open("unigram.output.txt", "w") as uni:
        for result in results:
            uni.write(str(result) + "\n")


    # Bigram
    count_vec = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    transform_train = count_vec.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transform_train, training['polarity'])

    # Output Bigram
    transform_test = count_vec.transform(test_set['text'])
    results = classifier.predict(transform_test)
    with open("bigram.output.txt", "w") as bi:
        for result in results:
            bi.write(str(result) + "\n")


    # Unigram TdIdf
    count_vec = TfidfVectorizer(stop_words=stopwords)
    transform_train = count_vec.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transform_train, training['polarity'])

    # Output Unigram TdIdf
    transform_test = count_vec.transform(test_set['text'])
    results = classifier.predict(transform_test)
    with open("unigramtfidf.output.txt", "w") as uni_f:
        for result in results:
            uni_f.write(str(result) + "\n")


    # Bigram TdIdf
    count_vec = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    transform_train = count_vec.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transform_train, training['polarity'])

    # Output Bigram TdIdf
    transform_test = count_vec.transform(test_set['text'])
    results = classifier.predict(transform_test)
    with open("bigramtfidf.output.txt", "w") as bi_f:
        for result in results:
            bi_f.write(str(result) + "\n")


###################################
#  Ref
# http://www.pythonforbeginners.com/code-snippets-source-code/python-os-listdir-and-endswith
# https://docs.python.org/3/library/stdtypes.html#str.translate
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# https://stackoverflow.com/questions/39938435/using-strip-method-and-punctuation-in-python
# https://stackoverflow.com/questions/41535571/how-to-explain-the-str-maketrans-function-in-python-3-6
# https://docs.python.org/3.1/library/string.html
# https://stackoverflow.com/questions/16096627/pandas-select-row-of-data-frame-by-integer-index
# https://stackoverflow.com/questions/10406130/check-if-something-is-not-in-a-list-in-python
# https://stackoverflow.com/questions/28534292/read-word-from-a-unicode-line-instead-of-char
# https://stackoverflow.com/questions/8113782/split-string-on-whitespace-in-python
