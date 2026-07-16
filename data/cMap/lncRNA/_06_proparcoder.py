import re
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ProtPar:

    def __init__(self, sequence):
        self.sequence = sequence

    def mRNA_translate(self, mRNA):
        return Seq(mRNA).translate()

    def protein_param(self, putative_seqprot):
        return (putative_seqprot.instability_index(), putative_seqprot.isoelectric_point(), putative_seqprot.gravy(), putative_seqprot.molecular_weight())

    def param(self, seq):
        strinfoAmbiguous = re.compile("X|B|Z|J|U", re.I)
        ptU = re.compile("U", re.I)
        seqRNA = ptU.sub("T", str(seq).strip())
        seqRNA = seqRNA.upper()
        CDS_size1, CDS_integrity, seqCDS = ExtractORF(seqRNA).longest_ORF(start=['ATG'], stop=['TAA', 'TAG', 'TGA'])
        seqprot = self.mRNA_translate(seqCDS)
        if '*' in seqprot:
            nPos = seqprot.index('*')
            if nPos != len(seqprot) - 1:
                seqprot = seqprot.split('*')[0]
                seqprot = seqprot + '*'
                print('seqprot')
                print(seqprot)
        pep_len = len(seqprot.strip("*"))
        newseqprot = strinfoAmbiguous.sub("", str(seqprot))
        protparam_obj = ProtParam.ProteinAnalysis(str(newseqprot.strip("*")))
        if pep_len > 0:
            Instability_index, PI, Gravy, Mw = self.protein_param(protparam_obj)
            pI_Mw = np.log10((float(Mw) / PI) + 1)
        else:
            Instability_index = 0.0
            PI = 0.0
            Gravy = 0.0
            Mw = 0.0
            pI_Mw = 0.0
        return (Instability_index, PI, Gravy, Mw, pI_Mw)

    def get_features(self):
        insta_fe, PI_fe, gra_fe, Mw, pI_Mw = self.param(self.sequence)
        # features = {
        #     "ProII: Pseudo protein instability index": insta_fe,  # 伪蛋白质的不稳定性指数
        #     "ProPI: Pseudo protein isoelectric point": PI_fe,   # 伪蛋白质的等电点
        #     "ProAH: Pseudo protein average hydropathy": gra_fe,   # 伪蛋白质的平均亲水性
        #     "ProMW: Pseudo protein molecular weight": Mw,   #  伪蛋白质的分子量
        #     "PPMFS: Pseudo protein PI-MW frame score": pI_Mw  # 伪蛋白质的 PI-MW 框架得分
        # }
        # features=f"{insta_fe} {PI_fe} {gra_fe} {Mw} {pI_Mw}"
        features = [insta_fe, PI_fe, gra_fe, Mw, pI_Mw]
        return features

class ExtractORF:
    def __init__(self, seq):
        self.seq = seq

    def longest_ORF(self, start, stop):
        longest_orf = ""
        start_codon = start[0]
        stop_codons = set(stop)
        for frame in range(3):
            seq = self.seq[frame:]
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if codon in start_codon:
                    for j in range(i, len(seq) - 2, 3):
                        codon = seq[j:j+3]
                        if codon in stop_codons:
                            orf = seq[i:j+3]
                            if len(orf) > len(longest_orf):
                                longest_orf = orf
                            break
        return len(longest_orf), "complete", longest_orf

# 调用
# pathi = 'lncRNA_data_test.csv'
# list_tv_set = open(pathi, 'r').readlines()
# Fea=[]
# for line in list_tv_set:
#     RNAiname, RNAisequence, RNAistructure = line.strip().split(',')
#     protpar = ProtPar(RNAisequence)
#     features = protpar.get_features()
#     Fea.append(features)
# scaler = MinMaxScaler()
# Fea= scaler.fit_transform(Fea)
# fea_str = [' '.join(map(str, features)) for features in Fea]
