# CIS 511 Natural Language Processing
# Assignment 2

# Code Written by Zihao Zhao

import sys
import random
from tqdm import tqdm
import time


class Measure:
    def __init__(self, train_path='POS.train', test_path='POS.test'):
        self.train_path = train_path
        self.test_path = test_path

        self.unigram_tags = {}
        self.bigram_tags = {}
        # for each word, calculate the unigram for all possible tags
        self.word_unigram_tags = {}
        self.total_tags = []

        # for testing data, seperate word and tags
        self.test_words = []
        self.test_tags = []

        self.pred_tags = []

    def prepare_unitags_bitags_uniwords(self):
        print("Preparing training data....")
        with open(self.train_path) as file:
            for line in file:
                words = line.split()
                for i in range(len(words)):
                    # I found some cases like 50\/50/CD, rpartition can avoid this.
                    word, part, tag = words[i].rpartition('/')

                    # Generate unigram_tags
                    self.unigram_tags[tag] = self.unigram_tags.get(tag, 0) + 1

                    # Generate bigram_tags
                    if i == 0:
                        # For the bi-gram tags, Use BOS as the tag of beginning of a sentence
                        temp = "BOS" + "," + tag
                    else:
                        word1, part1, tag1 = words[i - 1].rpartition('/')
                        temp = tag1+","+tag
                    self.bigram_tags[temp] = self.bigram_tags.get(temp, 0) + 1

                    # Generate word_unigram_tags
                    if word not in self.word_unigram_tags.keys():
                        self.word_unigram_tags[word] = {tag:0}
                    # self.word_unigram_tags[word] = self.word_unigram_tags.get(word, {word:{tag:0}})
                    self.word_unigram_tags[word][tag] = self.word_unigram_tags[word].get(tag, 0) + 1

                    # Add tag to total_tags
                    if tag not in self.total_tags:
                        self.total_tags.append(tag)


        print("Raw count for unigram_tags: {}".format(len(self.unigram_tags)))
        print("Raw count for bigram_tags: {}".format(len(self.bigram_tags)))
        print("Raw count for for words: {}".format(len(self.word_unigram_tags)))
        print("Raw count for for tags: {}".format(len(self.total_tags)))
        print("Done")

    def prepare_testwords_testtags(self):
        print("Preparing testing data....")
        with open(self.test_path) as file:
            for line in file:
                words = line.split()
                sentence = []
                tags = []
                for i in range(len(words)):
                    # I found some cases like 50\/50/CD, rpartition can avoid this.
                    word, part, tag = words[i].rpartition('/')
                    sentence.append(word)
                    tags.append(tag)
                self.test_words.append(sentence)
                self.test_tags.append(tags)
        print("Done")

    def test_by_Viterbi(self):
        # count = 0
        for words, tags in tqdm(zip(self.test_words, self.test_tags)):
            # print("{}:{}".format(count,len(self.test_words)))
            # count += 1
            # Initialization Step
                # for t = 1 to T
                # Score(t, 1) = Pr(W1 | Tt) * Pr(Tt | φ)
                # BackPtr(t, 1) = 0;
            score = {}
            backPtr = {}
            for tag in self.total_tags:
                prob_tag_given_null = self.cal_tag_given_tag('BOS', tag)
                prob_word_given_tag = self.cal_word_given_tag(words[0], tag)
                index = tag + ",0"
                score[index] = prob_tag_given_null*prob_word_given_tag
                backPtr[index] = 0

            # Iteration Step
            # for w = 2 to W
            #   for t = 1 to T
            #     Score(t, w) = Pr(Ww | Tt) * MAXj = 1, T(Score(j, w - 1) * Pr(Tt | Tj))
            #     BackPtr(t, w) = index of j that gave the max above
            for i in range(1,len(words)):
                for tag in self.total_tags:
                    prob_word_given_tag = self.cal_word_given_tag(words[i], tag)

                    max_val = 0
                    max_tag = None
                    for j_tag in self.total_tags:
                        prev_index = j_tag + ",{}".format(i-1)
                        if prev_index not in score.keys():
                            score[prev_index] = 1e-5
                        cur_val = score[prev_index] * self.cal_tag_given_tag(j_tag, tag)
                        if cur_val > max_val:
                            max_val = cur_val
                            max_tag = j_tag
                    index = tag+",{}".format(i)
                    score[index] = prob_word_given_tag*max_val
                    backPtr[index] = max_tag

            # Sequence Identification
            # Seq(W) = t that maximizes Score(t, W)
            # for w = W -1 to 1
            # Seq(w) = BackPtr(Seq(w + 1), w + 1)
            max_val = 0
            max_tag = None
            pred_tags = []
            for j_tag in self.total_tags:
                prev_index = j_tag + ",{}".format(len(words)-1)
                if prev_index not in score.keys():
                    score[prev_index] = 1e-5
                cur_val = score[prev_index]
                if cur_val > max_val:
                    max_val = cur_val
                    max_tag = j_tag
            pred_tags.append(max_tag)

            for i in range(len(words)-2,-1,-1):
                if i == len(words)-2:
                    max_tag = max_tag
                elif words[i] in self.word_unigram_tags.keys():
                    max_val = 0
                    pred_tags = []
                    for j_tag in self.total_tags:
                        cur_val = score[j_tag + ",{}".format(i+1)]
                        if cur_val > max_val:
                            max_val = cur_val
                            max_tag = j_tag

                pred_tags.append(backPtr[max_tag+",{}".format(i+1)])

            pred_tags.reverse()
            self.pred_tags.append(pred_tags)

    def test_by_baseline(self):
        for i in range(len(self.test_words)):
            pred_tags = []
            for j in range((len(self.test_words[i]))):
                if self.test_words[i][j] in self.word_unigram_tags.keys():
                    pred_tags.append(max(self.word_unigram_tags[self.test_words[i][j]],\
                                         key=self.word_unigram_tags[self.test_words[i][j]].get))
            self.pred_tags.append(pred_tags)


    def eval(self):
        total = 0
        correct = 0
        for i in range(len(self.test_tags)):
            for j in range(len(self.test_tags[i])):
                if len(self.pred_tags[i]) == j:
                    break
                if self.test_tags[i][j] == self.pred_tags[i][j]:
                    correct += 1
                # else:
                #     print("incorrect:labeled:{},predicted:{}".format(self.test_tags[i][j],self.pred_tags[i][j]))
                total += 1
        acc = correct/total

        print("Accuracy: {}".format(acc))

    def save_file(self):
        # print("Saving POS.test.out.....")
        file = open("POS.test.out", "w")
        for i in range(len(self.test_words)):
            line = ''
            for j in range((len(self.test_words[i]))):
                line += self.test_words[i][j]+"/"+self.test_tags[i][j]+' '
            file.write(line+'\n')
        file.close()

    def cal_word_given_tag(self, word, tag):
        # assign unseen bi tags to 1e-5 instead of 0
        if word in self.word_unigram_tags:
            if tag in self.word_unigram_tags[word]:
                return self.word_unigram_tags[word][tag]/sum(list(self.unigram_tags.values()))
        return 1e-5

    def cal_tag_given_tag(self, prev_tag, cur_tag):
        # assign unseen bi tags to 1e-5 instead of 0
        bigram_tag = prev_tag+","+cur_tag
        if bigram_tag in self.bigram_tags.keys():
            return self.bigram_tags[bigram_tag]/sum(list(self.bigram_tags.values()))
        else:
            return 1e-5


print(time.asctime(time.localtime(time.time())))
measure = Measure(sys.argv[1], sys.argv[2])
print("Data preprocessing...")
measure.prepare_unitags_bitags_uniwords()
measure.prepare_testwords_testtags()
print("Testing...")
measure.test_by_Viterbi()
print("Evaluating...")
measure.eval()
measure.save_file()