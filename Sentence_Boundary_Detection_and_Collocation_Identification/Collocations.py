# CIS 511 Natural Language Processing
# Assignment 1

# Code Written by Zihao Zhao

from math import log
import sys
import string

class Measure:
    def __init__(self, path='Collocations'):
        self.path = path
        self.unigram = {}
        self.bigram = {}
        self.bigram_tables = {}
        self.chi_score = {}
        self.pmi_score = {}

    def raw_count(self):
        with open(self.path) as file:
            for line in file:
                # new_line = line.translate(str.maketrans('','',string.punctuation))
                words = line.split()
                for word in words:
                    if word not in string.punctuation:
                        self.unigram[word] = self.unigram.get(word, 0) + 1

                for i in range(len(words)-1):
                    if words[i] not in string.punctuation and words[i+1] not in string.punctuation:
                        temp = words[i]+" "+words[i+1]
                        self.bigram[temp]=self.bigram.get(temp, 0) + 1

        print("Raw count for unigram: {}".format(len(self.unigram)))
        print("Raw count for bigram: {}".format(len(self.bigram)))

    def calculate(self, method='PMI'):
        # generate tables for all bigrams first
        w1, w2 = {}, {}
        for line, val in self.bigram.items():
            words = line.split()
            w1[words[0]] = w1.get(words[0], 0) + 1
            w2[words[1]] = w2.get(words[1], 0) + 1
        for line, val in self.bigram.items():
            words = line.split()
            self.bigram_tables[line] = [val, w2[words[1]]-val, \
                                        w1[words[0]]-val, len(self.bigram)+val-w1[words[0]]-w1[words[0]]]
        # print(len(self.bigram_tables))
        if method == 'chi-square':
            self.calculate_chi(w1,w2)
        if method == 'PMI':
            self.calculate_pmi(w1,w2)

    def calculate_chi(self,w1,w2):
        i, j = 2, 2
        for line, val in self.bigram_tables.items():
            chi_score = 0
            for _i in range(i):
                for _j in range(j):
                    o = val[0]
                    e = (val[0]+val[2])*(val[0]+val[1])/len(self.bigram)
                    chi_score += (o-e)*(o-e)/e

            self.chi_score[line] = chi_score

        self.chi_score = sorted(self.chi_score.items(), key=lambda item: item[1], reverse=True)
        print(self.chi_score[:20])

    def calculate_pmi(self,w1,w2):
        for line, val in self.bigram.items():
            words = line.split()
            p_w1w2 = val/len(self.bigram)
            p_w1 = w1[words[0]]/len(w1)
            p_w2 = w2[words[1]]/len(w2)
            pmi_score = log(p_w1w2/(p_w1*p_w2))

            self.pmi_score[line] = pmi_score

        self.pmi_score = sorted(self.pmi_score.items(), key=lambda item:item[1],reverse=True)
        print(self.pmi_score[:20])


measure = Measure(sys.argv[1])
measure.raw_count()
measure.calculate(sys.argv[2])