# CIS 511 Natural Language Processing
# Assignment 3

# Code Written by Zihao Zhao

import sys
import math

class NBWSD:
    def __init__(self, data_path = '​plant.wsd', fold = 5):
        self.data_path = data_path
        self.fold = fold
        self.fold_list = []
        self.avg_acc_list = []
        # self.target = 'plant'
        self.target = data_path[:data_path.rfind('.')]


        # need to be cleared per cross_validation ends
        self.cur_fold = 0
        self.sense_dict = {}
        self.sense_words_dict = {}
        self.total_instances = 0
        self.correct = 0
        self.dict = {}
        self.output_lines = []

    def clear_list(self):
        # print("Clearing lists......")
        self.cur_fold = 0
        self.sense_dict = {}
        self.sense_words_dict = {}
        self.total_instances = 0
        self.correct = 0
        self.dict = {}

    def save_file(self):
        # print("Saving "+self.data_path+".out......")
        file = open(self.data_path+".out", "w")
        for line in self.output_lines:
            file.write(line+'\n')
        file.close()

    def read_data(self):
        # print("Reading data......")
        # calculate total # of instances
        with open(self.data_path) as file:
            total_instances = 0
            for line in file:
                if line.find('<instance') == 0:
                    total_instances += 1

        # split into 5 folds
        count = round(total_instances/self.fold)
        for i in range(self.fold):
            if i == self.fold-1:
                self.fold_list.append([i*count, total_instances])
            else:
                self.fold_list.append([i*count, i*count+count])

    def cal_prob(self, fold_id):
        # print("Counting......")
        test_data = self.fold_list[fold_id]
        with open(self.data_path) as file:
            total = 0
            for line in file:
                if line.find('<answer') == 0:
                    total += 1
                    # avoid test data
                    if (total >= test_data[0]) and (total < test_data[1]):
                        continue
                    self.total_instances += 1
                    sense = line[line.rfind('%')+1:line.rfind('"')]
                    self.sense_dict[sense] = self.sense_dict.get(sense, 0) + 1

                    if sense not in self.sense_words_dict:
                        self.sense_words_dict[sense] = {}
                    next(file)
                    next_line = next(file)
                    words = next_line.split()
                    for i in range(len(words)):
                        if words[i].rfind('<head') != -1:
                            continue
                        self.sense_words_dict[sense][words[i]] = self.sense_words_dict[sense].get(words[i], 0) + 1
                        self.sense_words_dict[sense]["ALLWORDS"] = self.sense_words_dict[sense].get("ALLWORDS",0)+1
                        self.dict[words[i]] = self.dict.get(words[i], 0)+1
    # must call this function after cal_prob()
    def cal_acc(self, fold_id):
        # print(self.sense_words_dict['music']["ALLWORDS"])
        # print(self.sense_words_dict['fish']["ALLWORDS"])
        self.output_lines.append('Fold '+str(fold_id+1))
        print("Testing Fold{}......".format(fold_id+1))
        test_data = self.fold_list[fold_id]
        with open(self.data_path) as file:
            total = 0
            for line in file:
                if line.find('<answer') == 0:
                    output1 = line[line.find('"')+1:line.find('" sen')]
                    output2 = line[line.find('senseid="')+9:line.rfind('%')]
                    gt = line[line.rfind('%')+1:line.rfind('"')]
                    total += 1
                    # Test data
                    if (total >= test_data[0]) and (total < test_data[1]):
                        next(file)
                        next_line = next(file)
                        words = next_line.split()
                        max_p = float('-inf')
                        max_sense = ''
                        for key, val in self.sense_dict.items():
                            # print(val)
                            # print(self.total_instances)
                            # calculate p(yi)
                            _p = math.log(val / self.total_instances)
                            for i in range(len(words)):
                                # add one smoothing and log space
                                print(len(self.sense_dict))
                                _p += math.log((self.sense_words_dict[key].get(words[i], 0)+1)/(val+len(self.sense_dict)))
                            # _p = math.log(_p)

                            if _p > max_p:
                                max_p = _p
                                max_sense = key
                        self.output_lines.append(output1+" "+output2+"%"+max_sense)
                        if gt == max_sense:
                            self.correct += 1
        print("Correct:{}, Total:{}, Accuracy:{:.4f}".format(self.correct,test_data[1]-test_data[0],self.correct/(test_data[1]-test_data[0])))
        self.avg_acc_list.append(self.correct/(test_data[1]-test_data[0]))

    def cal_avg_acc(self):
        total = 0
        for acc in self.avg_acc_list:
            total += acc
        print("Avg accuracy:{:.4f}".format(total/self.fold))


print(sys.argv[1])
nbwsd = NBWSD(sys.argv[1], 5)
nbwsd.read_data()
for i in range(5):
    # accumulating words
    nbwsd.cal_prob(fold_id=i)
    # add one smoothing and log
    nbwsd.cal_acc(fold_id=i)
    nbwsd.clear_list()
# save file
nbwsd.save_file()
nbwsd.cal_avg_acc()
print("Done")