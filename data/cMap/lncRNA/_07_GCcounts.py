import numpy as np
from pandas import DataFrame
import pandas as pd
from Bio import Seq

class GCconder:
    def __init__(self, sequence=None):
        self.sequence = sequence

    # def read_sequences_from_csv(self):
    #     # Read RNA sequences from the CSV file
    #     df = pd.read_csv(self.csv_file)
    #     if 'sequence' not in df.columns:
    #         raise ValueError("CSV file must contain a 'sequence' column")
    #     return df['sequence'].tolist()

    def GetGC_Content(self, seq):
        '''Calculate GC content of sequence'''
        A, C, G, T = 1e-9, 1e-9, 1e-9, 1e-9
        for i in range(0, len(seq)):
            if seq[i] == 'A':
                A += 1
            elif seq[i] == 'C':
                C += 1
            elif seq[i] == 'G':
                G += 1
            elif seq[i] == 'T':
                T += 1

        GC = (G + C) / (A + C + G + T)
        return GC

    # Modify other methods to work with sequences directly
    def GC1(self, mRNA):
        if len(mRNA) < 3:
            return 0
        return self.GetGC_Content(mRNA[0::3])

    def GC2(self, mRNA):
        if len(mRNA) < 3:
            return 0
        return self.GetGC_Content(mRNA[1::3])

    def GC3(self, mRNA):
        if len(mRNA) < 3:
            return 0
        return self.GetGC_Content(mRNA[2::3])

    def gc1_frame_score(self, seq):
        return np.var([self.GetGC_Content(seq[i::3]) for i in range(3)])

    def gc2_frame_score(self, seq):
        return np.var([self.GetGC_Content(seq[i+1::3]) for i in range(3)])

    def gc3_frame_score(self, seq):
        return np.var([self.GetGC_Content(seq[i+2::3]) for i in range(3)])

    def get_gc(self):
        seq = self.sequence
        # features = []

        gc = self.GetGC_Content(seq)
        gc1 = self.GC1(seq)
        gc2 = self.GC2(seq)
        gc3 = self.GC3(seq)
        gc1_f = self.gc1_frame_score(seq)
        gc2_f = self.gc2_frame_score(seq)
        gc3_f = self.gc3_frame_score(seq)

        feature=[gc, gc1, gc2, gc3, gc1_f, gc2_f, gc3_f]

        return feature

# seq='CTTTCTCCCCGCCGCATTCCCGGTGTCGACTTACTAGCTGCAAGCCTCTGCCTGCCTTCCTGCGCGCCGTTCCCCGCTAGTCGCTGCTGCTGGCGCGCACTCGCC'
# # 初始化GCconder类，提供序列信息
# gc_calculator = GCconder(sequence=seq)
#
# # 计算GC特征
# result = gc_calculator.get_gc()
# result=' '.join(map(str, result))
#
# print(result)

