import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from dgllife.utils import mol_to_bigraph
from sklearn.metrics import roc_auc_score
from transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from load_traning_v2 import BERTDataset
from models_bert.bert_model import *

# 读取CSV文件
df = pd.read_csv('/home/a16/zyx/mix/datasets/clintox.csv')

# 确保列名正确
print(df.columns)

# 提取SMILES字符串和标签
smiles_list = df['smiles'].tolist()
fda_approved_labels = df['FDA_APPROVED'].tolist()
ct_tox_labels = df['CT_TOX'].tolist()

# 定义多分类标签生成函数
def combine_labels(fda_approved, ct_tox):
    if fda_approved == 0 and ct_tox == 0:
        return 0
    elif fda_approved == 1 and ct_tox == 0:
        return 1
    elif fda_approved == 0 and ct_tox == 1:
        return 2
    elif fda_approved == 1 and ct_tox == 1:
        return 3

# 生成多分类标签
combined_labels = [combine_labels(fda, ct) for fda, ct in zip(fda_approved_labels, ct_tox_labels)]

# 构建分子图
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_bigraph(mol, add_self_loop=True)

graphs = [smiles_to_graph(smiles) for smiles in smiles_list]

# 定义TransUNet模型结构
class MoleculeImageModel(nn.Module):
    def __init__(self, config_vit):
        super(MoleculeImageModel, self).__init__()
        self.transformer = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)

    def forward(self, x):
        logits, _ = self.transformer(x)
        return logits

# 配置TransUNet
config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 4  # 根据你的任务设置类别数
config_vit.patches.grid = (14, 14)  # 确保网格大小设置正确
pretrained_molecule_image_model_path = '/home/a16/zyx/mix/transunetepoch_9.pth'

# 加载TransUNet的特征提取器
molecule_image_model = MoleculeImageModel(config_vit)
molecule_image_model.load_state_dict(torch.load(pretrained_molecule_image_model_path), strict=False)

# 定义MolBERT模型结构
class MoleculeSequenceModel(nn.Module):
    def __init__(self):
        super(MoleculeSequenceModel, self).__init__()
        config = BertConfig.BertPreTrainedModel('bert-base-uncased')
        self.bert = BertModel.BertPreTrainedModel('bert-base-uncased', config=config)
        self.fc = nn.Linear(config.hidden_size, 256)  # 示例层

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 使用BERT的池化输出
        return self.fc(pooled_output)

# 加载MolBERT的预训练模型
pretrained_molecule_sequence_model_path = '/home/a16/zyx/mix/bert.model.epoch.14'
molecule_sequence_model = MoleculeSequenceModel()
molecule_sequence_model.load_state_dict(torch.load(pretrained_molecule_sequence_model_path), strict=False)

# 定义Prompt模型结构
class PromptModel(nn.Module):
    def __init__(self):
        super(PromptModel, self).__init__()
        # 定义模型结构
        self.layer = nn.Linear(128, 256)  # 示例层

    def forward(self, g):
        # 定义前向传播
        return self.layer(g)

pretrained_prompt_model_path = '/home/a16/zyx/mix/clintox.pth'
prompt_model = PromptModel()
prompt_model.load_state_dict(torch.load(pretrained_prompt_model_path), strict=False)

# 定义融合模型
class FusionModel(nn.Module):
    def __init__(self, molecule_image_model, molecule_sequence_model, prompt_model):
        super(FusionModel, self).__init__()
        self.molecule_image_model = molecule_image_model
        self.molecule_sequence_model = molecule_sequence_model
        self.prompt_model = prompt_model
        self.fc = nn.Linear(768, 4)  # 假设融合后的特征维度为768，输出维度为4（多分类）

    def forward(self, g, input_ids, attention_mask):
        image_features = self.molecule_image_model(g)
        sequence_features = self.molecule_sequence_model(input_ids, attention_mask)
        prompt_features = self.prompt_model(g)

        # 融合特征
        fused_features = torch.cat((image_features, sequence_features, prompt_features), dim=1)
        output = self.fc(fused_features)
        return output

fusion_model = FusionModel(molecule_image_model, molecule_sequence_model, prompt_model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类任务使用CrossEntropyLoss
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100  # 训练轮数
targets = torch.tensor(combined_labels, dtype=torch.long)  # 多分类标签

for epoch in range(num_epochs):
    fusion_model.train()
    optimizer.zero_grad()
    outputs = []
    for g in graphs:
        if g is not None:
            # 这里需要提供input_ids和attention_mask
            input_ids = torch.randint(0, 30522, (1, 128))  # 示例input_ids
            attention_mask = torch.ones((1, 128))  # 示例attention_mask
            output = fusion_model(g, input_ids, attention_mask)
            outputs.append(output)
    outputs = torch.stack(outputs)
    loss = criterion(outputs, targets)  # 假设targets是目标属性
    loss.backward()
    optimizer.step()

# 保存模型到指定路径
model_save_path = '/home/a16/zyx/mix/fusion_model.pth'
torch.save(fusion_model.state_dict(), model_save_path)

# 属性预测
fusion_model.eval()
predictions = []
with torch.no_grad():
    for g in graphs:
        if g is not None:
            # 这里需要提供input_ids和attention_mask
            input_ids = torch.randint(0, 30522, (1, 128))  # 示例input_ids
            attention_mask = torch.ones((1, 128))  # 示例attention_mask
            prediction = fusion_model(g, input_ids, attention_mask)
            predictions.append(prediction)

# 计算ROC-AUC
y_true = np.array(combined_labels)
y_score = torch.stack(predictions).numpy()
roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
print("ROC-AUC:", roc_auc)