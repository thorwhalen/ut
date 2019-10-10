__author__ = 'thor'

# -*- coding: utf-8 -*-

from . import math
import operator
import random
import gzip
import sys
import marshal
from functools import reduce

def cos_sim(p, q):
    sum0 = sum([x*x for x in p])
    sum1 = sum([x*x for x in q])
    sum2 = sum([x[0]*x[1] for x in zip(p, q)])
    return sum2/(sum0**0.5)/(sum1**0.5)

def _rand_mat(sizex, sizey):
    ret = []
    for i in range(sizex):
        ret.append([])
        for _ in range(sizey):
            ret[-1].append(random.random())
        norm = sum(ret[-1])
        for j in range(sizey):
            ret[-1][j] /= norm
    return ret

class Plsa:

    def __init__(self, corpus, topics=2):
        self.topics = topics
        self.corpus = corpus
        self.docs = len(corpus)
        self.each = list(map(sum, [list(x.values()) for x in corpus]))
        self.words = max(reduce(operator.add, [list(x.keys()) for x in corpus]))+1
        self.likelihood = 0
        self.zw = _rand_mat(self.topics, self.words)
        self.dz = _rand_mat(self.docs, self.topics)
        self.dw_z = None
        self.p_dw = []
        self.beta = 0.8

    def save(self, fname, iszip=True):
        d = {}
        for k, v in list(self.__dict__.items()):
            if hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            f = gzip.open(fname, 'wb')
            f.write(marshal.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        for k, v in list(d.items()):
            if hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def _cal_p_dw(self):
        self.p_dw = []
        for d in range(self.docs):
            self.p_dw.append({})
            for w in self.corpus[d]:
                tmp = 0
                for _ in range(self.corpus[d][w]):
                    for z in range(self.topics):
                        tmp += (self.zw[z][w]*self.dz[d][z])**self.beta
                self.p_dw[-1][w] = tmp

    def _e_step(self):
        self._cal_p_dw()
        self.dw_z = []
        for d in range(self.docs):
            self.dw_z.append({})
            for w in self.corpus[d]:
                self.dw_z[-1][w] = []
                for z in range(self.topics):
                    self.dw_z[-1][w].append(((self.zw[z][w]*self.dz[d][z])**self.beta)/self.p_dw[d][w])

    def _m_step(self):
        for z in range(self.topics):
            self.zw[z] = [0]*self.words
            for d in range(self.docs):
                for w in self.corpus[d]:
                    self.zw[z][w] += self.corpus[d][w]*self.dw_z[d][w][z]
            norm = sum(self.zw[z])
            for w in range(self.words):
                self.zw[z][w] /= norm
        for d in range(self.docs):
            self.dz[d] = [0]*self.topics
            for z in range(self.topics):
                for w in self.corpus[d]:
                    self.dz[d][z] += self.corpus[d][w]*self.dw_z[d][w][z]
            for z in range(self.topics):
                self.dz[d][z] /= self.each[d]

    def _cal_likelihood(self):
        self.likelihood = 0
        for d in range(self.docs):
            for w in self.corpus[d]:
                self.likelihood += self.corpus[d][w]*math.log(self.p_dw[d][w])

    def train(self, max_iter=100):
        cur = 0
        for i in range(max_iter):
            print('%d iter' % i)
            self._e_step()
            self._m_step()
            self._cal_likelihood()
            print('likelihood %f ' % self.likelihood)
            if cur != 0 and abs((self.likelihood-cur)/cur) < 1e-8:
                break
            cur = self.likelihood

    def inference(self, doc, max_iter=100):
        doc = dict([x for x in list(doc.items()) if x[0]<self.words])
        words = sum(doc.values())
        ret = []
        for i in range(self.topics):
            ret.append(random.random())
        norm = sum(ret)
        for i in range(self.topics):
            ret[i] /= norm
        tmp = 0
        for _ in range(max_iter):
            p_dw = {}
            for w in doc:
                p_dw[w] = 0
                for _ in range(doc[w]):
                    for z in range(self.topics):
                        p_dw[w] += (ret[z]*self.zw[z][w])**self.beta
            # e setp
            dw_z = {}
            for w in doc:
                dw_z[w] = []
                for z in range(self.topics):
                    dw_z[w].append(((self.zw[z][w]*ret[z])**self.beta)/p_dw[w])
            # m step
            ret = [0]*self.topics
            for z in range(self.topics):
                for w in doc:
                    ret[z] += doc[w]*dw_z[w][z]
            for z in range(self.topics):
                ret[z] /= words
            # cal likelihood
            likelihood = 0
            for w in doc:
                likelihood += doc[w]*math.log(p_dw[w])
            if tmp != 0 and abs((likelihood-tmp)/tmp) < 1e-8:
                break
            tmp = likelihood
        return ret

    def post_prob_sim(self, docd, q):
        sim = 0
        for w in docd:
            tmp = 0
            for z in range(self.topics):
                tmp += self.zw[z][w]*q[z]
            sim += docd[w]*math.log(tmp)
        return sim

######### unittest #################################
import unittest

class TestPlsa(unittest.TestCase):

    def test_train(self):
        corpus = [{0:2,3:5},{0:5,2:1},{1:2,4:5}]
        p = Plsa(corpus)
        p.train()
        self.assertTrue(cos_sim(p.dz[0], p.dz[1])>cos_sim(p.dz[0], p.dz[2]))
        self.assertTrue(p.post_prob_sim(p.corpus[0], p.dz[1])>p.post_prob_sim(p.corpus[0], p.dz[2]))

    def test_inference(self):
        corpus = [{0:2,3:5},{0:5,2:1},{1:2,4:5}]
        p = Plsa(corpus)
        p.train()
        z = p.inference({0:4, 6:7})
        self.assertTrue(abs(cos_sim(p.dz[0], p.dz[1])-cos_sim(p.dz[0], z))<1e-8)

if __name__ == '__main__':
    unittest.main()
