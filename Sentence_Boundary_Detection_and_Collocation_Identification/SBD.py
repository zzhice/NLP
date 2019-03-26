# CIS 511 Natural Language Processing
# Assignment 1

# Code Written by Zihao Zhao

import pandas as pd
import sys
import csv
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class SentenceBD:
    def __init__(self):
        self.train_dataset = []
        self.test_dataset = []
        self.train_features_list = []
        self.train_labels_list = []
        self.test_features_list = []
        self.test_labels_list = []
        self.y_pred = []

    def prepare_data(self, train_path='SBD.train', test_path='SBD.test'):
        # Feature description:
        # feature_dict['RWord']: Word to the left of “.” (L) (values: English vocab)
        # feature_dict['LWord']: Word to the right of “.” (R) (values: English vocab)
        # feature_dict['LLength']: Length of L < 3 (values: binary)
        # feature_dict['LCap']: Is L capitalized (values: binary)
        # feature_dict['RCap']: Is R capitalized (values: binary)
        # feature_dict['RLength']: Length of L < 3 (values: binary)
        # feature_dict['LCount']:  The number of times this word appears in the Left. (values: numeric)
        # feature_dict['RCount']: The number of times this word appears in the Right. (values: numeric)

        print("Preparing data......")
        all_dataset = []
        all_path = [train_path, test_path]

        for i in range(len(all_path)):
            _dataset = pd.read_csv(all_path[i], header=None, delim_whitespace=True, quoting=csv.QUOTE_NONE)
            new_names = {0: 'index', 1: 'words', 2: 'label'}
            _dataset.rename(columns=new_names, inplace=True)
            _dataset.drop('index', axis=1, inplace=True)
            all_dataset.append(_dataset)

        self.train_dataset = all_dataset[0]
        self.test_dataset = all_dataset[1]

    # extract all features from dataframe
        all_features_list = []
        all_labels_list = []
        for i in tqdm(range(len(all_dataset))):
            features_list = []
            label_list = []
            left, right = {}, {}
            for index, val in enumerate(all_dataset[i]['label'].values):
                if val == 'EOS' or val == 'NEOS':
                    if index == len(all_dataset[i]['label'])-1:
                        rword = 'null'
                    else:
                        rword = all_dataset[i]['words'][index+1]
                    lword = all_dataset[i]['words'][index][:-1]
                    left[lword] = left.get(lword, 0) + 1
                    right[rword] = right.get(rword, 0) + 1

            for index, val in enumerate(all_dataset[i]['label'].values):
                feature_dict = {}
                if val == 'EOS' or val == 'NEOS':
                    '''Important!:
                        if you want to train with certain features,
                        just comment the rest features below,
                        that's all you need to do.
                    '''
                    if index == len(all_dataset[i]['label'])-1:
                        feature_dict['RWord'] = 'null'
                        feature_dict['RCap'] =  False
                        feature_dict['RLength'] = True
                        feature_dict['RCount'] = right['null']
                    else:
                        feature_dict['RWord'] = all_dataset[i]['words'][index+1]
                        feature_dict['RCap'] = True if all_dataset[i]['words'][index + 1].isupper() else False
                        feature_dict['RLength'] = True if len(all_dataset[i]['words'][index + 1]) < 3 else False
                        feature_dict['RCount'] = right[all_dataset[i]['words'][index+1]]
                    feature_dict['LWord'] = all_dataset[i]['words'][index][:-1]
                    feature_dict['LLength'] = True if len(all_dataset[i]['words'][index][:-1]) < 3 else False
                    feature_dict['LCap'] = True if all_dataset[i]['words'][index][:-1].isupper() else False
                    feature_dict['LCount'] = left[all_dataset[i]['words'][index][:-1]]
                    '''Important!:
                        Comment/uncomment the features above to 
                        make your life easier to debug.
                    '''

                    features_list.append(feature_dict)
                    label_list.append(val)

            all_features_list.append(features_list)
            all_labels_list.append(label_list)

        self.train_features_list = all_features_list[0]
        self.test_features_list = all_features_list[1]
        self.train_labels_list = all_labels_list[0]
        self.test_labels_list = all_labels_list[1]
        print("Finished")

    def train(self):
        print("Start training and testing......")

        dv = DictVectorizer()
        le = LabelEncoder()

        x_train = dv.fit_transform(self.train_features_list)
        x_test = dv.transform(self.test_features_list)
        # print("After extracting features from dict:\n {}".format(dv.get_feature_names()))
        print("Labels: ['EOS', 'NEOS']")
        y_train = le.fit_transform(self.train_labels_list)
        y_test = le.transform(self.test_labels_list)
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        target_names = ['EOS', 'NEOS']
        print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print(classification_report(y_test, y_pred, target_names=target_names))

        self.y_pred = y_pred

    def save_file(self):
        print("Creating SBD.test.out.....")
        _dataset = self.test_dataset
        pred = []
        count = 0
        for index, val in enumerate(_dataset['label'].values):
            if val == 'TOK':
                pred.append('TOK')
            else:
                pred.append('EOS' if self.y_pred[count] == 0 else 'NEOS')
                count += 1
        _dataset['predict'] = pred
        _dataset.to_csv("SBD.test.out", sep=" ", header=0)



sentenceBD = SentenceBD()
sentenceBD.prepare_data(sys.argv[1], sys.argv[2])
sentenceBD.train()
sentenceBD.save_file()