import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from dgllife.utils import mol_to_bigraph
from sklearn.metrics import roc_auc_score
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from transformers import BertModel, BertConfig
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from dataset.load_traning_v2 import BERTDataset
from models.bert_model import BertForPreTraining

# 读取CSV文件
df = pd.read_csv('/home/a16/zyx/TransUNet-main/prompt/datasets/clintox.csv')

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
pretrained_molecule_image_model_path = '/home/a16/zyx/TransUNet-main/model/TU_own224/epoch_9.pth'

# 加载TransUNet的特征提取器
molecule_image_model = MoleculeImageModel(config_vit)
molecule_image_model.load_state_dict(torch.load(pretrained_molecule_image_model_path), strict=False)

# 定义MolBERT模型结构
class MoleculeSequenceModel(nn.Module):
    def __init__(self):
        super(MoleculeSequenceModel, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=config)
        self.fc = nn.Linear(config.hidden_size, 256)  # 示例层

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 使用BERT的池化输出
        return self.fc(pooled_output)

# 加载MolBERT的预训练模型
pretrained_molecule_sequence_model_path = '/home/a16/zyx/TransUNet-main/model/bertmodel.pth'
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

pretrained_prompt_model_path = '/home/a16/zyx/TransUNet-main/model/prompt/clintox.pth'
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
model_save_path = '/home/a16/zyx/TransUNet-main/model/fusion_model.pth'
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

# 分子序列预训练模型相关的参数和训练方法
class Pretrainer:
    def __init__(self, bert_model,
                 vocab_size,
                 max_seq_len,
                 batch_size,
                 lr,
                 with_cuda=True,
                 ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.max_seq_len = max_seq_len
        bertconfig = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["vocab_size"],
            hidden_dropout_prob=0.5,
            attention_probs_dropout_prob=0.5
        )
        self.bert_model = bert_model(config=bertconfig)
        self.bert_model.to(self.device)
        train_dataset = BERTDataset(corpus_path=config["train_corpus_path"],
                                    word2idx_path=config["word2idx_path"],
                                    seq_len=self.max_seq_len,
                                    hidden_dim=bertconfig.hidden_size,
                                    on_memory=True,
                                    )
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=config["num_workers"],
                                           collate_fn=lambda x: x)
        test_dataset = BERTDataset(corpus_path=config["test_corpus_path"],
                                   word2idx_path=config["word2idx_path"],
                                   seq_len=self.max_seq_len,
                                   hidden_dim=bertconfig.hidden_size,
                                   on_memory=True,
                                   )
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          num_workers=config["num_workers"],
                                          collate_fn=lambda x: x)
        self.positional_enc = self.init_positional_encoding(hidden_dim=bertconfig.hidden_size,
                                                            max_seq_len=self.max_seq_len)
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)
        optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(optim_parameters, lr=self.lr)
        print("Total Parameters:", sum([p.nelement() for p in self.bert_model.parameters()]))

    def init_positional_encoding(self, hidden_dim, max_seq_len):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
            if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def test(self, epoch, df_path="/home/bks/tmp/pycharm_project_994/output_model/df_log.pickle"):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False, df_path=df_path)

    def load_model(self, model, dir_path="/home/bks/tmp/pycharm_project_994/output_model"):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        print("load name : " , dir_path)
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for training!".format(checkpoint_dir))

    def train(self, epoch, df_path="/home/bks/tmp/pycharm_project_994/output_model/df_log.pickle"):
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True, df_path=df_path)

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=None):
        if ignore_index is None:
            loss_func = CrossEntropyLoss()
        else:
            loss_func = CrossEntropyLoss(ignore_index = ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def get_mlm_accuracy(self, predictions, labels):
        predictions = torch.argmax(predictions, dim=-1, keepdim=False)
        mask = (labels > 0).to(self.device)
        mlm_accuracy = torch.sum((predictions == labels) * mask).float()
        mlm_accuracy /= (torch.sum(mask).float() + 1e-8)
        return mlm_accuracy.item()

    def padding(self, output_dic_lis):
        bert_input = [i["bert_input"] for i in output_dic_lis]
        bert_label = [i["bert_label"] for i in output_dic_lis]
        bert_input = torch.nn.utils.rnn.pad_sequence(bert_input, batch_first=True)
        bert_label = torch.nn.utils.rnn.pad_sequence(bert_label, batch_first=True)
        return {"bert_input": bert_input,
                "bert_label": bert_label,}

    def iteration(self, epoch, data_loader, train=True, df_path="/home/bks/tmp/pycharm_project_994/output_model/df_log.pickle"):
        global log_dic
        if not os.path.isfile(df_path) and epoch != 0:
            raise RuntimeError("log DataFrame path not found and can't create a new one because we're not training from scratch!")
        if not os.path.isfile(df_path) and epoch == 0:
            df = pd.DataFrame(columns=["epoch","train_mlm_loss", "train_mlm_acc",
                                       "test_mlm_loss","test_mlm_acc"
                                       ])
            df.to_pickle(df_path)
            print("log DataFrame created!")

        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        total_mlm_loss = 0
        total_mlm_acc = 0
        total_element = 0
        for i, data in data_iter:
            data = self.padding(data)
            data = {key: value.to(self.device) for key, value in data.items()}
            positional_enc = self.positional_enc[:, :data["bert_input"].size()[-1], :].to(self.device)
            mlm_preds, next_sen_preds = self.bert_model.forward(input_ids=data["bert_input"],
                                                                positional_enc=positional_enc,)
            mlm_acc = self.get_mlm_accuracy(mlm_preds, data["bert_label"])
            mlm_loss = self.compute_loss(mlm_preds, data["bert_label"], self.vocab_size, ignore_index=0)
            loss = mlm_loss

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step