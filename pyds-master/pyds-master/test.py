
from __future__ import print_function
from pyds import *
from itertools import product
import numpy as np
import math
import matplotlib.pyplot as plt
import csv    #加载csv包便于读取csv文件

# 读取csv文件
csv_file = open('C:/Users/ZChang/Desktop/elman.csv')    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件
elman=[]    #创建列表准备接收csv各行数据
for one_line in csv_reader_lines:
    elman.append(one_line)    #将读取的csv分行数据按行存入列表‘data’中
'''
i = 0
while i < 200:
    print(elman[i])    #访问列表中的数据
    i = i+1
'''
csv_file = open('C:/Users/ZChang/Desktop/bp1.csv')    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件
bp=[]    #创建列表准备接收csv各行数据
for one_line in csv_reader_lines:
    bp.append(one_line)    #将读取的csv分行数据按行存入列表‘data’中
'''
i = 0
while i < 200:
    print(bp[i])    #访问列表中的数据
    i = i+1
'''
csv_file = open('C:/Users/ZChang/Desktop/pnn.csv')    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件
pnn=[]    #创建列表准备接收csv各行数据
for one_line in csv_reader_lines:
    pnn.append(one_line)    #将读取的csv分行数据按行存入列表‘data’中
'''
i = 0
while i < 200:
    print(pnn[i])    #访问列表中的数据
    i = i+1
'''

# 创建mass function
FOD = {0, 1, 2, 3}

# Jousselme距离中的矩阵D
D = np.mat([[1,0.5,0.5,0.5], [0.5,1,0.5,0.5], [0.5,0.5,1,0.5], [0.5,0.5,0.5,1]])

fussion_result = []
for i in range(200):

    # 计算证据距离
    d = [[0,0,0],[0,0,0],[0,0,0]]
    m = np.mat([[float(elman[i][0]), float(elman[i][1]), float(elman[i][2]), float(elman[i][3])],
         [float(bp[i][0]), float(bp[i][1]), float(bp[i][2]), float(bp[i][3])],
         [float(pnn[i][0]), float(pnn[i][1]), float(pnn[i][2]), float(pnn[i][3])]])

    for k in range(3):
        for j in range(3):
            d[k][j] = math.sqrt(0.5*np.dot(np.dot((m[k]-m[j]), D), np.transpose(m[k]-m[j])))
    #print(d)

    sim = [[0,0,0],[0,0,0],[0,0,0]]
    for k in range(3):
        for j in range(3):
            sim[k][j] = 1 - d[k][j]
    #print(sim)

    sup = [0, 0, 0]
    sum = 0
    for k in range(3):
        for j in range(3):
            if k != j :
                sup[k] += sim[k][j]
        sum += sup[k]
    #print(sup)

    crd = [0, 0, 0]
    for k in range(3):
        crd[k] = sup[k] / sum
    #print(crd)

    ds = DS()  # 若干证据组成一个DS证据组
    ds.setFOD(FOD)
    temp = []
    evi1 = Evidence()
    evi1.setEvidence(float(elman[i][0]), {0})
    evi1.setEvidence(float(elman[i][1]), {1})
    evi1.setEvidence(float(elman[i][2]), {2})
    evi1.setEvidence(float(elman[i][3]), {3})
    temp.append(evi1)
    ds.setDS(temp)

    temp = []
    evi2 = Evidence()
    evi2.setEvidence(float(bp[i][0]),  {0})
    evi2.setEvidence(float(bp[i][1]),  {1})
    evi2.setEvidence(float(bp[i][2]),  {2})
    evi2.setEvidence(float(bp[i][3]),  {3})
    temp.append(evi2)
    ds.setDS(temp)

    temp = []
    evi3 = Evidence()
    evi3.setEvidence(float(pnn[i][0]),  {0})
    evi3.setEvidence(float(pnn[i][1]),  {1})
    evi3.setEvidence(float(pnn[i][2]),  {2})
    evi3.setEvidence(float(pnn[i][3]),  {3})
    temp.append(evi3)
    ds.setDS(temp)
    #u = deng_Ent(ds)
    #u = improved_Belief_Ent(ds)
    #u = weighted_Belief_Ent(ds)
    #u = proposed_Belief_Ent(ds, math.e)
    #u = pan_Proposed_Belief_Ent(ds)
    u = improved_Deng_Ent(ds)


    un = [0,0,0]
    u_sum = 0
    for k in range(3):
        u_sum += u[k]
    for k in range(3):
        un[k] = u[k]/u_sum
    #print(un)

    crdm = [0, 0, 0]
    for k in range(3):
        crdm[k] = crd[k] * math.exp(1+un[k])
    #print(crdm)

    crdmn = [0, 0, 0]
    crdm_sum = 0
    for k in range(3):
        crdm_sum += crdm[k]
    for k in range(3):
        crdmn[k] = crdm[k]/crdm_sum
    #print(crdmn)

    wam = [0,0,0,0]
    wam[0] = crdmn[0]*float(elman[i][0]) + crdmn[1]*float(bp[i][0]) + crdmn[2]*float(pnn[i][0])
    wam[1] = crdmn[0] * float(elman[i][1]) + crdmn[1] * float(bp[i][1]) + crdmn[2] * float(pnn[i][1])
    wam[2] = crdmn[0] * float(elman[i][2]) + crdmn[1] * float(bp[i][2]) + crdmn[2] * float(pnn[i][2])
    wam[3] = crdmn[0] * float(elman[i][3]) + crdmn[1] * float(bp[i][3]) + crdmn[2] * float(pnn[i][3])

    '''

    m1 = MassFunction({'0': float(elman[i][0]), '1': float(elman[i][1]), '2': float(elman[i][2]), '3': float(elman[i][3])})
    #print('m_1 =', m1)
    m2 = MassFunction({'0': float(bp[i][0]), '1': float(bp[i][1]), '2': float(bp[i][2]), '3': float(bp[i][3])})
    #print('m_1 =', m2)
    m3 = MassFunction({'0': float(pnn[i][0]), '1': float(pnn[i][1]), '2': float(pnn[i][2]), '3': float(pnn[i][3])})
    #print('m_1 =', m3)
    fussion_result.append(max((m1 & m2 & m3), key=(m1 & m2 & m3).get))
    #print(m1 & m2 & m3)
    '''
    m = MassFunction({'0': wam[0], '1': wam[1], '2': wam[2], '3': wam[3]})
    fussion_result.append(max((m & m), key=(m & m).get))
    print(m & m)


a = []
csv_file = open('C:/Users/ZChang/Desktop/111.csv')    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件
b=[]    #创建列表准备接收csv各行数据
for one_line in csv_reader_lines:
    b.append(one_line)
#print(b)

for i in fussion_result:
    if i == frozenset({'0'}):
        a.append(0)
    if i == frozenset({'1'}):
        a.append(1)
    if i == frozenset({'2'}):
        a.append(2)
    if i == frozenset({'3'}):
        a.append(3)



sum = 0
for i in range(200):
    if a[i] == int(b[i][0]):
        sum = sum + 1
print(sum/200)


