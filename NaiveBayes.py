# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 23:08:19 2014

@author: heshenghuan
"""

from __future__ import division
import codecs
import math

class NBClassifier(object):
    '''
    Naive Bayes classifier implement by Python
    '''
    def __init__(self):
        self.samp_feat_vec = []
        self.samp_class_vec = []
        self.feat_set_size = 0
        self.class_set_size = 0
        #self.count_Dic = {}
        #self.class_set = []
        self.class_prb = []
        self.feat_class_prb = []

    def train(self, event_model=0):
        """
        training Naive Bayes classifier
        """
        print "Learning..."
        self.calc_class_prb()
        if event_model == 0:
            self.calc_feat_class_prb_bernoulli()
        else:
            self.calc_feat_class_prb_multinomial()
        self.samp_class_vec = []
        self.samp_feat_vec = []

    def classify(self, test_file, output_file, event_model=0, output_format=1):
        """
        classify data
        """
        print "Classifying test file..."
        self.load_data(test_file)
        pred_class_vec = []
        output = codecs.open(output_file, "w")
        for i in range(len(self.samp_class_vec)):
            #samp_class = self.samp_class_vec[i]
            samp_feat = self.samp_feat_vec[i]
            pred_score = []
            if event_model == 0:
                pred_score = self.predict_logp_bernoulli(samp_feat)
            else:
                pred_score = self.predict_logp_multinomial(samp_feat)
            pred_class = self.score_to_class(pred_score)
            pred_class_vec.append(pred_class)
            output.write(str(pred_class)+'\t')
            if output_format == 1:
                for j in range(self.class_set_size):
                    output.write(str(j+1)+':'+str(pred_class[j])+' ')
            else:
                pred_prb = self.score_to_prb(pred_score)
                for j in range(self.class_set_size):
                    output.write(str(j+1)+':'+str(pred_prb[j])+' ')
            output.write('\n')
        output.close()
        acc = self.calc_acc(self.samp_class_vec, pred_class_vec)
        return acc

    def load_data(self, filepath):
        """
        make traing set from file
        return sample_set, label_set
        """
        data = codecs.open(filepath, 'r')
        for line in data.readlines():
            val = line.strip().split('\t')
            label = int(val[0])
            self.samp_class_vec.append(label)
            if label > self.class_set_size:
                self.class_set_size = label
            sample_vec = {}
            val = val[-1].split(" ")
            for i in range(0, len(val)):
                [index, value] = val[i].split(':')
                sample_vec[int(index)] = float(value)
                if index > self.feat_set_size:
                    self.feat_set_size = index
            self.samp_feat_vec.append(sample_vec)

    def calc_class_prb(self):
        """
        calc_class_prb
        """
        self.class_prb = [0]*self.class_set_size
        for label in self.samp_class_vec:
            self.class_prb[label-1] += 1

        samp_num = len(self.samp_class_vec)
        for i in range(self.class_set_size):
            self.class_prb[i] = self.class_prb[i]/samp_num

    def calc_feat_class_prb_bernoulli(self):
        """
        The model is bernoulli model.
        Calculate feat prb given class with Laplace Smoothing.
        """
        feat_class_df = []
        #calculate feat_class_df
        for i in range(self.feat_set_size):
            feat_class_df.append([0.0]*self.class_set_size)
        for k in range(len(self.samp_class_vec)):
            samp_feat = self.samp_feat_vec[k]
            samp_class = self.samp_class_vec[k]
            for feat_id in samp_feat.keys():
                #feat_value = samp_feat[feat_id]
                feat_class_df[feat_id-1][samp_class-1] += 1
        #calculate class_df
        class_df = [0]*self.class_set_size
        for samp_class in self.samp_class_vec:
            class_df[samp_class-1] += 1
        #calculate feat_class_prb
        for j in range(self.feat_set_size):
            prb = [0]*self.class_set_size
            for k in range(self.class_set_size):
                prb[j][k] = float((1+feat_class_df[j][k])/(2+class_df[k]))
            self.feat_class_prb.append(prb)

    def calc_feat_class_prb_multinomial(self):
        """
        The model is multinomial model.
        Calculate feat prb given class with Laplace Smoothing.
        """
        feat_class_tf = []
        #calculate feat_class_tf
        for i in range(self.feat_set_size):
            feat_class_tf.append([0.0]*self.class_set_size)
        for k in range(len(self.samp_class_vec)):
            samp_feat = self.samp_feat_vec[k]
            samp_class = self.samp_class_vec[k]
            for feat_id in samp_feat.keys():
                feat_value = samp_feat[feat_id]
                feat_class_tf[feat_id-1][samp_class-1] += feat_value
        #calculate class_tf
        class_tf = [0]*self.class_set_size
        for j in range(self.class_set_size):
            for k in range(len(self.feat_class_prb)):
                class_tf[j] += feat_class_tf[k][j]
        #calculate feat_class_prb
        for j in range(self.feat_set_size):
            prb = [0]*self.class_set_size
            for k in range(self.class_set_size):
                prb[j][k] = float((1+feat_class_tf[j][k])/\
                            (self.feat_set_size+class_tf[j]))
            self.feat_class_prb.append(prb)

    def predict_logp_bernoulli(self, samp_feat):
        """
        predict_logp_bernoulli
        """
        feat_vec_out = range(self.feat_set_size)
        for feat_id in samp_feat.keys():
            feat_vec_out.remove(feat_id)
        logp = [0]*self.class_set_size
        for j in range(self.class_set_size):
            logp_samp_given_class = 0.0
            for feat_id in samp_feat.keys():
                logp_samp_given_class += math.log(self.feat_class_prb[feat_id-1][j])
            for feat_id in feat_vec_out:
                logp_samp_given_class += math.log(1-self.feat_class_prb[feat_id-1][j])
            logp_samp_and_class = logp_samp_given_class + math.log(self.class_prb[j])
            logp[j] = logp_samp_and_class
        return logp

    def predict_logp_multinomial(self, samp_feat):
        """
        predict_logp_multinomial
        """
        logp = [0]*self.class_set_size
        for j in range(self.class_set_size):
            logp_samp_given_class = 0.0
            for feat_id in samp_feat.keys():
                logp_samp_given_class += math.log(samp_feat[feat_id] * \
                                            self.feat_class_prb[feat_id-1][j])
            logp_samp_and_class = logp_samp_given_class + math.log(self.class_prb[j])
            logp[j] = logp_samp_and_class
        return logp

    def score_to_prb(self, score):
        """
        score_to_prb
        """
        prb = [0] * self.class_set_size
        for i in range(self.class_set_size):
            delta_prb_sum = 0.0
            for j in range(self.class_set_size):
                delta_prb_sum += math.exp(score[j]-score[i])
            prb[i] = 1/delta_prb_sum
        return prb

    def score_to_class(self, score):
        """
        score_to_class
        """
        pred_class = 0
        max_score = score[0]
        for j in range(1, len(score)):
            if score[j] > max_score:
                max_score = score[j]
                pred_class = j
        return pred_class+1

    def calc_acc(self, test_class_vec, pred_class_vec):
        """
        calc_acc
        """
        if len(test_class_vec) != len(pred_class_vec):
            print "Error: two vectors should have same length."
            return 0
        else:
            error_num = 0
            for i in range(len(test_class_vec)):
                if test_class_vec[i] != pred_class_vec[i]:
                    error_num += 1
            return 1-float(error_num/len(test_class_vec))
