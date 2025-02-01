# 0034.NLTK_NLP_Basic_Techniques

# NLTK_Basic_Techniques

This file demonstrates various Natural Language Processing (NLP) techniques using the NLTK and scikit-learn libraries.  It covers tokenization (word, sentence, whitespace, and punctuation-based), n-grams, stemming (Porter, Lancaster, Snowball), lemmatization, stop word removal, part-of-speech tagging, named entity recognition, word cloud generation, and text vectorization (CountVectorizer and TfidfVectorizer).  Additionally, it includes basic chunking/shallow parsing using regular expressions.

# NLTK_CountVectorizer_GaussianNB

This code performs sentiment analysis on restaurant reviews. It preprocesses text data (cleaning, lowercasing, stemming, stop word removal), converts it to a numerical representation using CountVectorizer, trains a Gaussian Naive Bayes classifier on the data, and evaluates the model using a confusion matrix.  It also saves the processed data and the vectorized features to CSV files.

# NLP_Transformers (Completed in Colab)

This file utilizes the transformers library to generate text using the GPT-2 model. It defines functions to chunk long text into smaller segments that the model can handle, and then generates responses for each chunk. The file demonstrates how to load a pre-trained model and tokenizer, handle padding, and decode the generated tokens back into text.
