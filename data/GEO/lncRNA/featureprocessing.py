# @Date:   2024/5/28 18:49
# description: RNA一维、二维特征提取
# Feature processing

import numpy as np
import re
import math
from sklearn import ensemble
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from _07_GCcounts import GCconder
from CTDcode import CTDcoder
from _06_proparcoder import ProtPar
from sklearn.feature_extraction.text import TfidfVectorizer


# np.random.seed(1337) # random seed
#
separator, Sequencekmertotal, SequenceGgaptotal, Structurekmertotal, StructureGgaptotal = ' ', 3, 3, 3, 3

def SequencekmerExtract(sequence, totalkmer):
    #可以不用替换了
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequencekmer = ''
    for k in range(totalkmer):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalkmer - kk))
        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * sequence.count(s1) / sk
                string1 = str(f1) + separator
                sequencekmer = sequencekmer + string1
        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0
                    for lkmer2 in range(len(sequence) - kk + 1):
                        if sequence[lkmer2] == s2[0] and sequence[lkmer2 + 1] == s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    string2 = str(f2) + separator
                    sequencekmer = sequencekmer + string2
        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0
                        for lkmer3 in range(len(sequence) - kk + 1):
                            if sequence[lkmer3] == s3[0] and sequence[lkmer3 + 1] == s3[1] and sequence[lkmer3 + 2] == s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        string3 = str(f3) + separator
                        sequencekmer = sequencekmer + string3
        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0
                            for lkmer4 in range(len(sequence) - kk + 1):
                                if sequence[lkmer4] == s4[0] and sequence[lkmer4 + 1] == s4[1] and sequence[lkmer4 + 2] == s4[2] and sequence[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            string4 = str(f4) + separator
                            sequencekmer = sequencekmer + string4
        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0
                                for lkmer5 in range(len(sequence) - kk + 1):
                                    if sequence[lkmer5] == s5[0] and sequence[lkmer5 + 1] == s5[1] and sequence[lkmer5 + 2] == s5[2] and sequence[lkmer5 + 3] == s5[3] and sequence[lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                string5 = str(f5) + separator
                                sequencekmer = sequencekmer + string5
        # 6-mer
        if kk == 6:
            for char61 in character:
                for char62 in character:
                    for char63 in character:
                        for char64 in character:
                            for char65 in character:
                                for char66 in character:
                                    s6 = char61 + char62 + char63 + char64 + char65 + char66
                                    numkmer6 = 0
                                    for lkmer6 in range(len(sequence) - kk + 1):
                                        if sequence[lkmer6] == s6[0] and sequence[lkmer6 + 1] == s6[1] and sequence[lkmer6 + 2] == s6[2] and sequence[lkmer6 + 3] == s6[3] and sequence[lkmer6 + 4] == s6[4] and sequence[lkmer6 + 5] == s6[5]:
                                            numkmer6 = numkmer6 + 1
                                    f6 = wk * numkmer6 / sk
                                    string6 = str(f6) + separator
                                    sequencekmer = sequencekmer + string6
    return sequencekmer

def SequenceGgapExtract(sequence, totalGgap):
    sequence = sequence.replace('U', 'T')
    character = 'ATCG'
    sequenceGgap = ''
    for k in range(totalGgap):
        kk = k + 1
        sk = len(sequence) - kk + 1
        wk = 1 / (4 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    num1 = 0
                    for l1 in range(len(sequence) - kk - 1):
                        if sequence[l1] == char11 and sequence[l1 + kk + 1] == char12:
                            num1 = num1 + 1
                    f1 = wk * num1 / sk
                    string1 = str(f1) + separator
                    sequenceGgap = sequenceGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    num2 = 0
                    for l2 in range(len(sequence) - kk - 3):
                        if sequence[l2] == char21 and sequence[l2 + kk + 1] == char22:
                            num2 = num2 + 1
                    f2 = wk * num2 / sk
                    string2 = str(f2) + separator
                    sequenceGgap = sequenceGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    num3 = 0
                    for l3 in range(len(sequence) - kk - 3):
                        if sequence[l3] == char31 and sequence[l3 + kk + 1] == char32:
                            num3 = num3 + 1
                    f3 = wk * num3 / sk
                    string3 = str(f3) + separator
                    sequenceGgap = sequenceGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    num4 = 0
                    for l4 in range(len(sequence) - kk - 3):
                        if sequence[l4] == char41 and sequence[l4 + kk + 1] == char42:
                            num4 = num4 + 1
                    f4 = wk * num4 / sk
                    string4 = str(f4) + separator
                    sequenceGgap = sequenceGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    num5 = 0
                    for l5 in range(len(sequence) - kk - 3):
                        if sequence[l5] == char51 and sequence[l5 + kk + 1] == char52:
                            num5 = num5 + 1
                    f5 = wk * num5 / sk
                    string5 = str(f5) + separator
                    sequenceGgap = sequenceGgap + string5
    return sequenceGgap

def StructurekmerExtract(structure, totalkmer):
    character = ').'
    structurekmer = '' # 特征

    sp = structure.split()
    ssf = sp[0]
    ssf = ssf.replace('(', ')')
    for k in range(totalkmer):
        kk = k + 1
        sk = len(ssf) - kk + 1
        wk = 1 / (2 ** (totalkmer - kk))
        # 1-mer
        if kk == 1:
            for char11 in character:
                s1 = char11
                f1 = wk * ssf.count(s1) / sk
                string1 = str(f1) + separator
                structurekmer = structurekmer + string1
        # 2-mer
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    s2 = char21 + char22
                    numkmer2 = 0
                    for lkmer2 in range(len(ssf) - kk + 1):
                        if ssf[lkmer2] == s2[0] and ssf[lkmer2 + 1] == s2[1]:
                            numkmer2 = numkmer2 + 1
                    f2 = wk * numkmer2 / sk
                    string2 = str(f2) + separator
                    structurekmer = structurekmer + string2
        # 3-mer
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        s3 = char31 + char32 + char33
                        numkmer3 = 0
                        for lkmer3 in range(len(ssf) - kk + 1):
                            if ssf[lkmer3] == s3[0] and ssf[lkmer3 + 1] == s3[1] and ssf[lkmer3 + 2] == s3[2]:
                                numkmer3 = numkmer3 + 1
                        f3 = wk * numkmer3 / sk
                        string3 = str(f3) + separator
                        structurekmer = structurekmer + string3
        # 4-mer
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            s4 = char41 + char42 + char43 + char44
                            numkmer4 = 0
                            for lkmer4 in range(len(ssf) - kk + 1):
                                if ssf[lkmer4] == s4[0] and ssf[lkmer4 + 1] == s4[1] and ssf[lkmer4 + 2] == s4[2] and ssf[lkmer4 + 3] == s4[3]:
                                    numkmer4 = numkmer4 + 1
                            f4 = wk * numkmer4 / sk
                            string4 = str(f4) + separator
                            structurekmer = structurekmer + string4
        # 5-mer
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            for char55 in character:
                                s5 = char51 + char52 + char53 + char54 + char55
                                numkmer5 = 0
                                for lkmer5 in range(len(ssf) - kk + 1):
                                    if ssf[lkmer5] == s5[0] and ssf[lkmer5 + 1] == s5[1] and ssf[lkmer5 + 2] == s5[2] and ssf[lkmer5 + 3] == s5[3] and ssf[lkmer5 + 4] == s5[4]:
                                        numkmer5 = numkmer5 + 1
                                f5 = wk * numkmer5 / sk
                                string5 = str(f5) + separator
                                structurekmer = structurekmer + string5
    return structurekmer

def StructureGgapExtract(structure, totalGgap):
    character = ').'
    structureGgap = ''
    sp = structure.split()
    ssf = sp[0]
    ssf = ssf.replace('(', ')')
    for k in range(totalGgap):
        kk = k + 1
        sk = len(ssf) - kk + 1
        wk = 1 / (2 ** (totalGgap - kk))
        if kk == 1:
            for char11 in character:
                for char12 in character:
                    for char13 in character:
                        for char14 in character:
                            num1 = 0
                            for l1 in range(len(ssf) - kk - 3):
                                if ssf[l1] == char11 and ssf[l1 + 1] == char12 and ssf[l1 + kk + 2] == char13 and ssf[l1 + kk + 3] == char14:
                                    num1 = num1 + 1
                            f1 = wk * num1 / sk
                            string1 = str(f1) + separator
                            structureGgap = structureGgap + string1
        if kk == 2:
            for char21 in character:
                for char22 in character:
                    for char23 in character:
                        for char24 in character:
                            num2 = 0
                            for l2 in range(len(ssf) - kk - 3):
                                if ssf[l2] == char21 and ssf[l2 + 1] == char22 and ssf[l2 + kk + 2] == char23 and ssf[l2 + kk + 3] == char24:
                                    num2 = num2 + 1
                            f2 = wk * num2 / sk
                            string2 = str(f2) + separator
                            structureGgap = structureGgap + string2
        if kk == 3:
            for char31 in character:
                for char32 in character:
                    for char33 in character:
                        for char34 in character:
                            num3 = 0
                            for l3 in range(len(ssf) - kk - 3):
                                if ssf[l3] == char31 and ssf[l3 + 1] == char32 and ssf[l3 + kk + 2] == char33 and ssf[l3 + kk + 3] == char34:
                                    num3 = num3 + 1   
                            f3 = wk * num3 / sk
                            string3 = str(f3) + separator
                            structureGgap = structureGgap + string3
        if kk == 4:
            for char41 in character:
                for char42 in character:
                    for char43 in character:
                        for char44 in character:
                            num4 = 0
                            for l4 in range(len(ssf) - kk - 3):
                                if ssf[l4] == char41 and ssf[l4 + 1] == char42 and ssf[l4 + kk + 2] == char43 and ssf[l4 + kk + 3] == char44:
                                    num4 = num4 + 1
                            f4 = wk * num4 / sk
                            string4 = str(f4) + separator
                            structureGgap = structureGgap + string4
        if kk == 5:
            for char51 in character:
                for char52 in character:
                    for char53 in character:
                        for char54 in character:
                            num5 = 0
                            for l5 in range(len(ssf) - kk - 3):
                                if ssf[l5] == char51 and ssf[l5 + 1] == char52 and ssf[l5 + kk + 2] == char53 and ssf[l5 + kk + 3] == char54:
                                    num5 = num5 + 1
                            f5 = wk * num5 / sk
                            string5 = str(f5) + separator
                            structureGgap = structureGgap + string5
    return structureGgap

def ArithmeticLevel(feature1, feature2):
    fpair = ''
    if feature1 != '' and feature2 != '':
        f1, f2 = feature1.strip().split(' '), feature2.strip().split(' ')
        for i in range(len(f1)):
            a = float(f1[i])
            b = float(f2[i])
            c = 50 * (a + b) / 2
            fpair += str(c) + separator
    return fpair

def get_kmers(sequence, k):
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return kmers

def get_sequence_vector(sequence, model, k):
    kmers = get_kmers(sequence, k)
    vector_size = model.vector_size
    # 初始化一个零向量
    sequence_vector = np.zeros(vector_size)
    count = 0
    for kmer in kmers:
        if kmer in model.wv:
            sequence_vector += model.wv[kmer]
            count += 1
    if count > 0:
        sequence_vector /= count
    return sequence_vector

def kmer_word2vec(seq, k):
    seq = seq[:-(len(seq) % 3)] if len(seq) % 3 != 0 else seq
    kmers = get_kmers(seq, k)
    sentences = [kmers]
    model = Word2Vec(vector_size=128, window=5, min_count=1, sg=1, negative=5)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=50)
    # 获取RNA序列的特征表示
    sequence_vector = get_sequence_vector(seq, model, k)
    # 转换numpy数组为一个字符串，并用空格分隔
    sequence_vector_str = ' '.join(map(str, sequence_vector))
    return sequence_vector_str


# def get_sequence_vector(sequence, model, k, tfidf_weights):
#     kmers = get_kmers(sequence, k)
#     vector_size = model.vector_size
#     sequence_vector = np.zeros(vector_size)
#     count = 0
#     for kmer in kmers:
#         if kmer in model.wv:
#             # 取对应k-mer的TF-IDF权重
#             tfidf_weight = tfidf_weights.get(kmer, 1.0)
#             sequence_vector += model.wv[kmer] * tfidf_weight
#             count += 1
#     if count > 0:
#         sequence_vector /= count
#     return sequence_vector
#
#
# def kmer_word2vec(seq, k):
#     seq = seq[:-(len(seq) % 3)] if len(seq) % 3 != 0 else seq
#     kmers = get_kmers(seq, k)
#     sentences = [" ".join(kmers)]
#
#     # 计算 TF-IDF 权重
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_vectorizer.fit(sentences)
#     tfidf_features = tfidf_vectorizer.transform(sentences)
#
#     # 创建一个词到TF-IDF权重的映射
#     feature_names = tfidf_vectorizer.get_feature_names_out()
#     tfidf_weights = {feature_names[i]: tfidf_features[0, i] for i in range(len(feature_names))}
#
#     # 输出TF-IDF权重进行调试
#     # print("TF-IDF weights:", tfidf_weights)
#
#     model = Word2Vec(vector_size=32, window=5, min_count=1, sg=1, negative=5)
#     model.build_vocab([kmers])
#     model.train([kmers], total_examples=model.corpus_count, epochs=50)
#
#     # 获取RNA序列的特征表示，加入TF-IDF权重
#     sequence_vector = get_sequence_vector(seq, model, k, tfidf_weights)
#
#     # 转换numpy数组为一个字符串，并用空格分隔
#     sequence_vector_str = ' '.join(map(str, sequence_vector))
#     return sequence_vector_str


# def FeatureConstruction(ListPair):
#     Seq_kmer=[]
#     Seq_Ggap=[]
#     Str_kmer =[]
#     Str_Ggap = []
#     Feature=[]
#     DACC=[]
#     CTD=[]
#     for LinePair in ListPair:
#         RNAiname, RNAisequence, RNAistructure = LinePair.strip().split(',')
#         # RNAi sequence k-mer
#         SequenceRNAikmer = SequencekmerExtract(RNAisequence, Sequencekmertotal)
#         # RNAi sequence g-gap
#         SequenceRNAiGgap = SequenceGgapExtract(RNAisequence, SequenceGgaptotal)
#         # RNAi structure k-mer
#         StructureRNAikmer = StructurekmerExtract(RNAistructure, Structurekmertotal)
#         # RNAi structure g-gap
#         StructureRNAiGgap = StructureGgapExtract(RNAistructure, StructureGgaptotal)
#         # RNAi k-mer&word2vec features
#         k_word2vec = kmer_word2vec(RNAisequence, 3)
#         # GC特征：7维    初始化GCconder类，提供序列信息
#         gc_calculator = GCconder(sequence=RNAisequence)
#         gc = gc_calculator.get_gc()
#         gc = ' '.join(map(str, gc))
#         # 转录序列描述CTD：30维
#         ctd_calculator = CTDcoder(sequence=RNAisequence)
#         ctd = ctd_calculator.CTD()
#         ctd = ' '.join(map(str, ctd))
#         # 伪蛋白特征： 5维 (需要后续进行归一化处理，保持量纲统一)
#         protpar = ProtPar(RNAisequence)
#         pro_fea = protpar.get_features()
#         # print(k_word2vec)
#         feature=SequenceRNAikmer + SequenceRNAiGgap + StructureRNAikmer + StructureRNAiGgap
#         DACC.append(dacc_features)
#         Feature.append(f"{feature}")
#         Seq_kmer.append(f"{SequenceRNAikmer}")
#         Seq_Ggap.append(f"{SequenceRNAiGgap}")
#         Str_kmer.append(f"{StructureRNAikmer}")
#         Str_Ggap.append(f"{StructureRNAiGgap}")
#
#     return Seq_kmer, Seq_Ggap, Str_kmer, Str_Ggap, Feature, DACC


pathi='二级结构.csv'   #lncRNA存放格式  名称  人类的序列  二级结构
list_tv_set = open(pathi, 'r').readlines()
CTD = []
Protpar = []
GC = []
Dacc = []
w2v = []
Seq=[]
Strc = []
Feature = []
for line in list_tv_set:
    RNAiname,  RNAisequence, RNAistructure = line.strip().split(',')
    # # RNAi sequence k-mer
    SequenceRNAikmer = SequencekmerExtract(RNAisequence, Sequencekmertotal)
    # # RNAi sequence g-gap
    SequenceRNAiGgap = SequenceGgapExtract(RNAisequence, SequenceGgaptotal)
    # # RNAi structure k-mer
    StructureRNAikmer = StructurekmerExtract(RNAistructure, Structurekmertotal)
    # # RNAi structure g-gap
    StructureRNAiGgap = StructureGgapExtract(RNAistructure, StructureGgaptotal)
    featurei = SequenceRNAikmer + SequenceRNAiGgap + StructureRNAikmer + StructureRNAiGgap
    Feature.append(f"{featurei}")
    # 3_mer + Word2Vec：32维
    k_word2vec = kmer_word2vec(RNAisequence, 3)
    w2v.append(f"{k_word2vec}")
#     # GC特征：7维    初始化GCconder类，提供序列信息
    gc_calculator = GCconder(sequence=RNAisequence)
    gc = gc_calculator.get_gc()
    gc = ' '.join(map(str, gc))
    GC.append(gc)
#     # 转录序列描述CTD：30维
    ctd_calculator = CTDcoder(sequence=RNAisequence)
    ctd = ctd_calculator.CTD()
    ctd = ' '.join(map(str, ctd))
    CTD.append(f"{ctd}")
#     # 伪蛋白特征： 5维 (需要后续进行归一化处理，保持量纲统一)
    protpar = ProtPar(RNAisequence)
    pro_fea = protpar.get_features()
    Protpar.append(pro_fea)

    print(RNAiname)
#
scaler = MinMaxScaler()
Protpar= scaler.fit_transform(Protpar)
pro_str = [' '.join(map(str, features)) for features in Protpar]


# with open ('../data/lnc_old/Seq.csv', 'w') as file:
#     for fea in Seq:
#         file.write(fea+'\n')
# with open ('../data/lnc_old/Str.csv', 'w') as file:
#     for fea in Strc:
#         file.write(fea+'\n')

with open ('Word2Vec.csv','a') as file:
    for fea in w2v:
        file.write(fea+'\n')

with open ('GC_related.csv','a') as file1:
    for gc in GC:
        file1.write(gc+'\n')

with open ('CTD_sequence.csv','a') as file2:
    for gc in CTD:
        file2.write(gc+'\n')
with open ('Protpar.csv','a') as file3:
    for pro in pro_str:
        file3.write(pro+'\n')
with open ('Feature_i.csv','a') as file4:
    for pro in Feature:
        file4.write(pro+'\n')




# pathj='data/miRNA_data.csv'
# tv_set = open(pathj, 'r').readlines()
# CTD_j = []
# Protpar_j = []
# GC_j = []
# Dacc_j = []
# w2v_j = []
# Seq_j=[]
# Str_j = []
# for line in tv_set:
#     RNAiname, RNAisequence, RNAistructure = line.strip().split(',')
#     # RNAi sequence k-mer
#     SequenceRNAikmer = SequencekmerExtract(RNAisequence, Sequencekmertotal)
#     # RNAi sequence g-gap
#     SequenceRNAiGgap = SequenceGgapExtract(RNAisequence, SequenceGgaptotal)
#     # RNAi structure k-mer
#     StructureRNAikmer = StructurekmerExtract(RNAistructure, Structurekmertotal)
#     # RNAi structure g-gap
#     StructureRNAiGgap = StructureGgapExtract(RNAistructure, StructureGgaptotal)
#     seq = SequenceRNAikmer + SequenceRNAiGgap
#     strc = StructureRNAikmer + StructureRNAiGgap
#     Seq_j.append(f"{seq}")
#     Str_j.append(f"{strc}")
#     # 3_mer + Word2Vec：32维
#     k_word2vec = kmer_word2vec(RNAisequence, 3)
#     w2v_j.append(f"{k_word2vec}")
    # # GC特征：7维    初始化GCconder类，提供序列信息
    # gc_calculator = GCconder(sequence=RNAisequence)
    # gc = gc_calculator.get_gc()
    # gc = ' '.join(map(str, gc))
    # GC_j.append(gc)
    # # 转录序列描述CTD：30维
    # ctd_calculator = CTDcoder(sequence=RNAisequence)
    # ctd = ctd_calculator.CTD()
    # ctd = ' '.join(map(str, ctd))
    # CTD_j.append(f"{ctd}")
    # # 伪蛋白特征： 5维 (需要后续进行归一化处理，保持量纲统一)
    # protpar = ProtPar(RNAisequence)
    # pro_fea = protpar.get_features()
    # Protpar_j.append(pro_fea)

# scaler = MinMaxScaler()
# Protpar_j= scaler.fit_transform(Protpar_j)
# pro_str_j = [' '.join(map(str, features)) for features in Protpar_j]

# with open ('../data/miRNA/Seq.csv', 'w') as file:
#     for fea in Seq_j:
#         file.write(fea+'\n')
# with open ('../data/miRNA/Str.csv', 'w') as file:
#     for fea in Str_j:
#         file.write(fea+'\n')

# with open ('data/miRNA/Word2Vec_TF.csv','w') as file:
#     for fea in w2v_j:
#         file.write(fea+'\n')
# with open ('data/miRNA/GC_related.csv','w') as file1:
#     for gc in GC_j:
#         file1.write(gc+'\n')
# with open ('data/miRNA/CTD_sequence.csv','w') as file2:
#     for gc in CTD_j:
#         file2.write(gc+'\n')
# with open ('data/miRNA/Protpar.csv','w') as file3:
#     for pro in pro_str_j:
#         file3.write(pro+'\n')
# with open ('data/miRNA/Feature_j.csv','w') as file4:
#     for pro in Feature_j:
#         file4.write(pro+'\n')
