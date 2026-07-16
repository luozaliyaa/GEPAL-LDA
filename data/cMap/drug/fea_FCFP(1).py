import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def get_fcfp_fingerprint(DBid, smiles, radius=2, nbits=1024):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useFeatures=True)
    fingerprint_str = fingerprint.ToBitString()
    fingerprint_arr = [int(x) for x in fingerprint_str]
    fingerprint_arr.insert(0, DBid)
    return fingerprint_arr


data = pd.read_csv("smiles.csv")
DBid = data["Drug"]
smiles = data["SMILES"]

Flen = len(get_fcfp_fingerprint(DBid[0], smiles[0]))
header = ['Col-{}'.format(i) for i in range(1, Flen)]
header.insert(0,'DrugBank ID')

file_name = 'fea_FCFP.csv'
with open(file_name,'w' ,newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

for i in range(len(smiles)):
    fingerprint = get_fcfp_fingerprint(DBid[i], smiles[i])
    print(i+1)
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fingerprint)
file.close()

print("FCFP特征维数：",Flen-1)