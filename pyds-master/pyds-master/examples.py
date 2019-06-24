"""
Shows different use cases of the library.
"""

from __future__ import print_function
from pyds import *
from itertools import product
import numpy as np
import math
import matplotlib.pyplot as plt

# 创建mass function

FOD = {1, 2, 3}
n1 = {1}
n2 = {2}
n3 = {3}
n4 = {1, 3}

pa = [0.41, 0, 0.58, 0.55, 0.6]
pb = [0.29, 0.9, 0.07, 0.1, 0.1]
pc = [0.3, 0.1, 0.0, 0.0, 0.0]
pac = [0.0, 0.0, 0.35, 0.35, 0.3]

ds = DS()  # 若干证据组成一个DS证据组
ds.setFOD(FOD)

for i in range(5):
    temp = []
    evi = Evidence()
    evi.setEvidence(pa[i], n1)
    evi.setEvidence(pb[i], n2)
    evi.setEvidence(pc[i], n3)
    evi.setEvidence(pac[i], n4)
    temp.append(evi)
    ds.setDS(temp)


e3 = weighted_Belief_Ent(ds)
sum = 0
weight = [0, 0, 0, 0, 0]
for i in list(e3):
    sum += i
for i in range(len(e3)):
    weight[i] = list(e3)[i] / sum
print(weight)

mw = [0, 0, 0, 0]
for i in range(4):
    for j in range(5):
        mw[i] += weight[j] * ds.getDS()[j][0].getMass(i)
print(mw)

'''e1 = deng_Ent(ds)
e2 = improved_Belief_Ent(ds)
e3 = weighted_Belief_Ent(ds)
e4 = proposed_Belief_Ent(ds, math.e)
e5 = pan_Proposed_Belief_Ent(ds)
e6 = improved_Deng_Ent(ds)

x = range(5)
plt.xlabel("Element number in T")
plt.ylabel("Uncertainty degree")
plt.plot(x, e1, 'o-', label="Deng_Ent")
plt.plot(x, e2, '*-', label="Improved_Belief_Ent")
plt.plot(x, e3, '^-', label="Weighted_Belief_Ent")
plt.plot(x, e4, 'D-', label="Proposed_Belief_Ent")
plt.plot(x, e5, '>-', label="pan_Proposed_Belief_Ent")
plt.plot(x, e6, 'x-', label="Improved_Deng_Ent")
plt.legend(loc="upper left")
plt.show()
'''

'''
print('=== creating mass functions ===')
m1 = MassFunction({'ab':0.6, 'bc':0.3, 'a':0.1, 'ad':0.0}) # using a dictionary
print('m_1 =', m1)
m2 = MassFunction([({'a', 'b', 'c'}, 0.2), ({'a', 'c'}, 0.5), ({'c'}, 0.3)]) # using a list of tuples
print('m_2 =', m2)
m3 = MassFunction()
m3['bc'] = 0.8
m3[{}] = 0.2
print('m_3 =', m3, ('(unnormalized mass function)'))'''
m4 = MassFunction({'a':mw[0], 'b':mw[1], 'c':mw[2], 'ac':mw[3]}) # using a dictionary
print('m_4 =', m4)
'''
# 四个基本函数的使用
print('\n=== belief, plausibility, and commonality ===')
print('bel_1({a, b}) =', m1.bel({'a', 'b'}))
print('pl_1({a, b}) =', m1.pl({'a', 'b'}))
print('q_1({a, b}) =', m1.q({'a', 'b'}))
print('bel_1 =', m1.bel()) # entire belief function，如果不指定焦元，默认输出全部
print('bel_3 =', m3.bel())
print('m_3 from bel_3 =', MassFunction.from_bel(m3.bel())) # construct a mass function from a belief function

# 输出识别框架、焦元、核、核的交
print('\n=== frame of discernment, focal sets, and core  ===')
print('frame of discernment of m_1 =', m1.frame())
print('focal sets of m_1 =', m1.focal())
print('core of m_1 =', m1.core())
print('combined core of m_1 and m_3 =', m1.core(m3))
'''

# DS证据组合规则
print('\n=== Dempster\'s combination rule, unnormalized conjunctive combination (exact and approximate) ===')
print('Dempster\'s combination rule for m_1 and m_2 =', m4 & m4 & m4 & m4 & m4)
#print('Dempster\'s combination rule for m_1 and m_2 (Monte-Carlo, importance sampling) =', m1.combine_conjunctive(m2, sample_count=1000, importance_sampling=True))
#print('Dempster\'s combination rule for m_1, m_2, and m_3 =', m1.combine_conjunctive(m2, m3))
#print('unnormalized conjunctive combination of m_1 and m_2 =', m1.combine_conjunctive(m2, normalization=False))
#print('unnormalized conjunctive combination of m_1 and m_2 (Monte-Carlo) =', m1.combine_conjunctive(m2, normalization=False, sample_count=1000))
#print('unnormalized conjunctive combination of m_1, m_2, and m_3 =', m1.combine_conjunctive([m2, m3], normalization=False))
'''
print('\n=== normalized and unnormalized conditioning ===')
print('normalized conditioning of m_1 with {a, b} =', m1.condition({'a', 'b'}))
print('unnormalized conditioning of m_1 with {b, c} =', m1.condition({'b', 'c'}, normalization=False))

print('\n=== disjunctive combination rule (exact and approximate) ===')
print('disjunctive combination of m_1 and m_2 =', m1 | m2)
print('disjunctive combination of m_1 and m_2 (Monte-Carlo) =', m1.combine_disjunctive(m2, sample_count=1000))
print('disjunctive combination of m_1, m_2, and m_3 =', m1.combine_disjunctive([m2, m3]))

print('\n=== weight of conflict ===')
print('weight of conflict between m_1 and m_2 =', m1.conflict(m2))
print('weight of conflict between m_1 and m_2 (Monte-Carlo) =', m1.conflict(m2, sample_count=1000))
print('weight of conflict between m_1, m_2, and m_3 =', m1.conflict([m2, m3]))

print('\n=== pignistic transformation ===')
print('pignistic transformation of m_1 =', m1.pignistic())
print('pignistic transformation of m_2 =', m2.pignistic())
print('pignistic transformation of m_3 =', m3.pignistic())

print('\n=== local conflict uncertainty measure ===')
print('local conflict of m_1 =', m1.local_conflict())
print('entropy of the pignistic transformation of m_3 =', m3.pignistic().local_conflict())

print('\n=== sampling ===')
print('random samples drawn from m_1 =', m1.sample(5, quantization=False))
print('sample frequencies of m_1 =', m1.sample(1000, quantization=False, as_dict=True))
print('quantization of m_1 =', m1.sample(1000, as_dict=True))

print('\n=== map: vacuous extension and projection ===')
extended = m1.map(lambda h: product(h, {1, 2}))
print('vacuous extension of m_1 to {1, 2} =', extended)
projected = extended.map(lambda h: (t[0] for t in h))
print('project m_1 back to its original frame =', projected)

print('\n=== construct belief from data ===')
hist = {'a':2, 'b':0, 'c':1}
print('histogram:', hist)
print('maximum likelihood:', MassFunction.from_samples(hist, 'bayesian', s=0))
print('Laplace smoothing:', MassFunction.from_samples(hist, 'bayesian', s=1))
print('IDM:', MassFunction.from_samples(hist, 'idm'))
print('MaxBel:', MassFunction.from_samples(hist, 'maxbel'))
print('MCD:', MassFunction.from_samples(hist, 'mcd'))
'''