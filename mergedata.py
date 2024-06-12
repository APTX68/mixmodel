from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
import torch


# 批处理大小
batch_size = 200.

# 生成器函数从文件中逐行读取化合物SMILES数据
def read_smiles_generator(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# 合并并去重两个数据集的化合物SMILES
def merge_and_deduplicate_smiles(file1, file2):
    smiles_set = set()
    for file_path in [file1, file2]:
        for smiles in read_smiles_generator(file_path):
            smiles_set.add(smiles)
    return smiles_set

# 过滤和处理化合物数据
filtered_smiles = []
combined_smiles = merge_and_deduplicate_smiles("/home/bks/tmp/pycharm_project_994/dataset/zinc.txt", "/home/bks/tmp/pycharm_project_994/dataset/chem.txt")

for idx, smiles in enumerate(combined_smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    mw = Descriptors.MolWt(mol)
    num_atoms = mol.GetNumHeavyAtoms()
    clogp = Crippen.MolLogP(mol)
    atoms = set([a.GetSymbol() for a in mol.GetAtoms()])

    if 12 <= mw <= 600 and 3 <= num_atoms <= 50 and 5 <= clogp <= 7 and \
            atoms.issubset({'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'}):
        filtered_smiles.append(Chem.MolToSmiles(mol))

    # 每个 batch 处理完毕后打印信息
    if (idx + 1) % batch_size == 0:
        print(f"Processed {idx + 1} compounds. Valid compounds: {len(filtered_smiles)}")


# 保存处理后的化合物SMILES到新文件
output_file_path = "/home/bks/tmp/pycharm_project_994/dataset/pretrain_data.txt"
with open(output_file_path, 'w') as output_file:
    for smiles in filtered_smiles:
        output_file.write(smiles + "\n")