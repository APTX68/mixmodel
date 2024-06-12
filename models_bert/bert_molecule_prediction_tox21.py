from torch import nn
from models.bert_model_v2 import *

"""使用mean max pool"""

class Bert_Smiles_Analysis(nn.Module):
    def __init__(self, config, deviceId=2, isRegression=False):
        super(Bert_Smiles_Analysis, self).__init__()
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dense1 = nn.Linear(config.hidden_size, 100)
        self.final_dense1 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense2 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense3 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense4 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense5 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense6 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense7 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense8 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense9 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense10 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense11 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense12 = nn.Linear(100, 1)  # work_nums 分类的数目

        self.activation = nn.Sigmoid()
        self.work_nums = config.work_nums
        self.deviceId = deviceId
        self.isRegression = isRegression
        self.class_nums = config.work_nums

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        if predictions.shape[1] == 1:
            torch.cuda.set_device(device=self.deviceId)
            predictions = predictions.view(-1)
            y = labels.float().view(-1)
            epsilon = 1e-8

            # 交叉熵
            loss = -(y+1)/2 * torch.log(predictions + epsilon).cuda() - (torch.tensor(1.0) - ((y+1)/2)).cuda() * \
                   torch.log(torch.tensor(1.0) - predictions + epsilon).cuda()

            # loss matrix after removing null target
            is_valid = y ** 2 > 0
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
            loss = torch.mean(loss) # scalar值才可以反向传播

            # y = labels.view(predictions.shape).to(torch.float64)
            # Whether y is non-null or not.
            # is_valid = y ** 2 > 0
            # Loss matrix
            # loss_mat = criterion(predictions.double(), (y + 1) / 2)
            # loss matrix after removing null target
            # loss_mat = torch.where(is_valid, loss_mat,
            #                        torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            # loss = torch.sum(loss_mat) / torch.sum(is_valid)
            # print(loss)
            # 求均值, 并返回可以反传的loss
            # loss为一个实数
        return loss

    def forward(self, text_input, positional_enc, labels=None,text_input1=None,fp=None):
        # print(text_input.shape,labels.shape)
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                      output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]  # 为了避免过拟合
        # # sequence_output的维度是[batch_size, seq_len, embed_dim]
        avg_pooled = sequence_output.mean(1)  # [batch_size, embed_dim]
        max_pooled = torch.max(sequence_output, dim=1)  # [batch_size, embed_dim]
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)  # [batch_size, embed_dim * 2]
        pooled = torch.cat((pooled, text_input1), dim=1)
        pooled = torch.cat((pooled, fp), dim=1)
        pooled = self.dense(pooled)  # [batch_size, embed]
        pooled = self.dense1(pooled)
        # 我们在这里要解决的是多分类问题
        # predictions = self.final_dense(pooled)
        # print(pooled,".......")
        pred1 = self.activation(self.final_dense1(pooled))
        pred2 = self.activation(self.final_dense2(pooled))
        pred3 = self.activation(self.final_dense3(pooled))
        pred4 = self.activation(self.final_dense4(pooled))
        pred5 = self.activation(self.final_dense5(pooled))
        pred6 = self.activation(self.final_dense6(pooled))
        pred7 = self.activation(self.final_dense7(pooled))
        pred8 = self.activation(self.final_dense8(pooled))
        pred9 = self.activation(self.final_dense9(pooled))
        pred10 = self.activation(self.final_dense10(pooled))
        pred11 = self.activation(self.final_dense11(pooled))
        pred12 = self.activation(self.final_dense12(pooled))



        loss1 = self.compute_loss(pred1, labels[:, 0])
        loss2 = self.compute_loss(pred2, labels[:, 1])
        loss3 = self.compute_loss(pred3, labels[:, 2])
        loss4 = self.compute_loss(pred4, labels[:, 3])
        loss5 = self.compute_loss(pred5, labels[:, 4])
        loss6 = self.compute_loss(pred6, labels[:, 5])
        loss7 = self.compute_loss(pred7, labels[:, 6])
        loss8 = self.compute_loss(pred8, labels[:, 7])
        loss9 = self.compute_loss(pred9, labels[:, 8])
        loss10 = self.compute_loss(pred10, labels[:, 9])
        loss11 = self.compute_loss(pred11, labels[:, 10])
        loss12 = self.compute_loss(pred12, labels[:, 11])



        pred = [pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12,]
        loss = [loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11, loss12,]


        if labels is not None:
            # 计算loss
            # loss = self.compute_loss(predictions, labels)
            return pred, loss
        else:
            return pred