import nltk

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import blankline_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ne_chunk
from nltk import pos_tag, word_tokenize, RegexpParser
import svgling

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk. __version__

# nltk.download('punkt')

# nltk.download('averaged_perceptron_tagger')

AI = '''Artificial Intelligence refers to the intelligence of machines. This is in contrast to the natural intelligence of humans and animals. With Artificial Intelligence, machines perform functions such as learning, planning, reasoning and problem-solving. Most noteworthy, Artificial Intelligence is the simulation of human intelligence by machines. It is probably the fastest-growing development in the World of technology and innovation. Furthermore, many experts believe AI could solve major challenges and crisis situations.'''

AI

type(AI)

AI_tokens = word_tokenize(AI)
print( AI_tokens )

len(AI_tokens)

AI_sent = sent_tokenize(AI)
AI_sent

len(AI_sent)

AI

AI_blank = blankline_tokenize(AI) 
AI_blank

len(AI_blank)

AI_wt = WhitespaceTokenizer().tokenize(AI)
print( AI_wt )

print( len(AI_tokens) )
print( len(AI_wt) )

s = 'Good apple cost $3.88 in Hyderbad. Please buy me two of them. Thanks.'

s_tokens = word_tokenize(s)
print( s_tokens )

wt_s = WhitespaceTokenizer().tokenize(s)
print( wt_s )

print( len(s_tokens) )
print( len(wt_s) )

s = 'Good apple cost $3.88 in Hyderbad. Please buy me two of them. Thanks.'

s_wp = wordpunct_tokenize(s)
print( s_wp )

print( len(s_wp) )

AI_wp = wordpunct_tokenize(AI)
print( AI_wp )

print( len(AI_wp) )

string = 'hello the best and most beautifull thing in the world cannot be seen or even touched,they must be felt with heart'
quotes_tokens = nltk.word_tokenize(string)
print( quotes_tokens )
print( len(quotes_tokens) )

quotes_bigrams = list(nltk.bigrams(quotes_tokens))
print( quotes_bigrams )
print( len(quotes_bigrams) )

quotes_trigrams = list(nltk.trigrams(quotes_tokens))
print( quotes_trigrams )
print( len(quotes_trigrams) )

quotes_ngrams_4 = list(nltk.ngrams(quotes_tokens, 4)) 
print( quotes_ngrams_4 )
print( len(quotes_ngrams_4) )

quotes_ngrams_5 = list(nltk.ngrams(quotes_tokens, 5)) 
print( quotes_ngrams_5 )
print( len(quotes_ngrams_5) )

quotes_ngrams_9 = list(nltk.ngrams(quotes_tokens, 9)) 
print( quotes_ngrams_9 )
print( len(quotes_ngrams_9) )

quotes_ngrams_22 = list(nltk.ngrams(quotes_tokens, 22)) 
print( quotes_ngrams_22 )
print( len(quotes_ngrams_22) )

quotes_ngrams_23 = list(nltk.ngrams(quotes_tokens, 23)) 
print( quotes_ngrams_23 )
print( len(quotes_ngrams_23) )

quotes_ngrams_24 = list(nltk.ngrams(quotes_tokens, 24)) 
print( quotes_ngrams_24 )
print( len(quotes_ngrams_24) )

quotes_ngrams_25 = list(nltk.ngrams(quotes_tokens, 25)) 
print( quotes_ngrams_25 )
print( len(quotes_ngrams_25) )

pst = PorterStemmer()

print( pst.stem('having') )
print( pst.stem('affection') )
print( pst.stem('playing') )
print( pst.stem('give') )

pst = PorterStemmer()

words_to_stem=['give','giving','given','gave','thinking', 'loving', 'final', 'finalized', 'finally']

for words in words_to_stem:
    print(words+ ' : ' +pst.stem(words))

lst = LancasterStemmer()

for words in words_to_stem:
    print(words + ' : ' + lst.stem(words))

sbst = SnowballStemmer('english')

for words in words_to_stem:
    print(words+ ' : ' +sbst.stem(words))

words_to_stem = ['give','giving','given','gave','thinking', 'loving', 'final', 'finalized', 'finally']

word_lem = WordNetLemmatizer()

for words in words_to_stem:
    print(words+ ' : ' +word_lem.lemmatize(words))

print( stopwords.words('english') )
print( len(stopwords.words('english')) )

print( stopwords.words('spanish') )
print( len(stopwords.words('spanish')) )

print( stopwords.words('french') )
print( len(stopwords.words('french')) )

print( stopwords.words('german') )
print( len(stopwords.words('german')) )

print( stopwords.words('chinese') )
print( len(stopwords.words('chinese')) )

# OSError: No such file or directory: 'C:\\Users\\RAJEEV\\AppData\\Roaming\\nltk_data\\corpora\\stopwords\\hindi'
# print( stopwords.words('hindi') )
# print( len(stopwords.words('hindi')) )

punctuation = re.compile(r'[-.?!,:;()|0-9]')
punctuation

print( AI_tokens )
print( len(AI_tokens) )

sent = 'kathy is a natural when it comes to drawing'
sent_tokens = word_tokenize(sent)
sent_tokens

for token in sent_tokens:
    print(nltk.pos_tag([token]))

sent2 = 'john is eating a delicious cake'
sent2_tokens = word_tokenize(sent2)

for token in sent2_tokens:
    print(nltk.pos_tag([token]))

sent3 = 'the big cat ate the little mouse who was after fresh cheese'
sent3_tokens = nltk.pos_tag(word_tokenize(sent3))
print( sent3_tokens )

NE_sent = 'The US president stays in the WHITEHOUSE '
print( len(NE_sent) )

NE_tokens = word_tokenize(NE_sent)
print( NE_tokens )
print( len(NE_tokens) )

NE_tags = nltk.pos_tag(NE_tokens)
print( NE_tags )
print( len(NE_tags) )

NE_NER = ne_chunk(NE_tags)
print( NE_NER )
print( len(NE_NER) )

text = ("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")

print( text )

wordcloud_1 = WordCloud(
    width=680, 
    height=480, 
    margin=2, 
    background_color='black', 
    colormap='Accent', 
    mode='RGBA').generate(text)


print( wordcloud_1 )
print( type(wordcloud_1) )

plt.imshow(wordcloud_1, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

plt.imshow(wordcloud_1, interpolation='quadric')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

vectorizer_count = CountVectorizer()
vectorizer_count

sentence = " data science and ai genai has great career ahead "
vector = vectorizer_count.fit_transform([sentence])
vector

vector.toarray()

vectorizer_tfid = TfidfVectorizer()
vectorizer_tfid

sentence = " data science and ai genai has great career ahead "
vector_tf = vectorizer_tfid.fit_transform([sentence])
vector_tf

vector_tf.toarray()

text = "The quick brown fox jumps over the lazy dog."

tokens = word_tokenize(text)
print(tokens)

tagged_tokens = pos_tag(tokens)
tagged_tokens

# chunk_grammar = """\n NP:{<DT>?<JJ>*<NN>} \n VP:{<VB.*><NP|PP>*} \n PP:{<IN><NP>} \n"""
# chunk_grammar

chunk_grammar = r"""
NP: {<DT>?<JJ>*<NN>}
VP: {<VB.*><NP|PP>*}
PP: {<IN><NP>}
"""

chunk_grammar

chunk_parser = RegexpParser(chunk_grammar)
chunk_parser

chunked = chunk_parser.parse(tagged_tokens)
chunked

print(chunked)

# It will open in new tkinter window
# chunked.draw()

