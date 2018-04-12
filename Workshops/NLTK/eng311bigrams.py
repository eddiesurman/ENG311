
# coding: utf-8

import nltk
from nltk.collocations import *
import csv
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

corpusdir= 'corpus/'
newcorpus= PlaintextCorpusReader(corpusdir, '.*')
files = newcorpus.fileids()

print(files)

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
tokens = newcorpus.words()
word_fd = nltk.FreqDist(tokens)
bigram_fd = nltk.FreqDist(nltk.bigrams(tokens))
finder = BigramCollocationFinder(word_fd, bigram_fd)
scored = finder.score_ngrams(bigram_measures.raw_freq)
sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[::]

def csv_writer():
    with open('PleaseRenameThisNewBigramRaw.csv', 'w',newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        for k,v in sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[::]:
            writer.writerow([str(k)]+[str(v)])

csv_writer()
