from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import os


# 定义将SMILES转换为分子图像的函数
def convert_smiles_to_image(smiles, image_size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol, size=image_size)
        return img
    else:
        return None


# 加载包含SMILES序列的CSV文件
data = pd.read_csv('/home/a16/zyx/TransUNet-main/prompt/datasets/clintox.csv')

image_folder = '/home/a16/zyx/TransUNet-main/prompt/image/clintox'
label_file = '/home/a16/zyx/TransUNet-main/prompt/labels/clintoxlabel.txt'

# 打开标签文件以写入成功转换的SMILES序列及其对应列
with open(label_file, 'w') as label_file:
    for index, row in data.iterrows():
        smiles = row['smiles']

        try:
            # 将SMILES转换为分子图像并保存
            molecule_image = convert_smiles_to_image(smiles)
            image_path = os.path.join(image_folder, f'image_{index}.png')
            molecule_image.save(image_path)

            # 将所有列信息写入标签文件
            label_info = ', '.join([f'{col}: {row[col]}' for col in data.columns])
            label_file.write(f'SMILES: {smiles}, {label_info}\n')
        except Exception as e:
            print(f"无法处理SMILES：{smiles}，错误信息：{str(e)}")

# 确保图像和标签文件一一对应
num_images = len(os.listdir(image_folder))
num_labels = sum(1 for line in open(label_file))

if num_images == num_labels:
    print("图像和标签文件生成成功，并且一一对应。")
else:
    print("图像和标签文件数量不一致，请检查生成过程。")