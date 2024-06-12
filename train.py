import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms  # 添加这一行来导入transforms模块
from utils import DiceLoss
from datasets.dataset_synapse import MoleculeDataset, RandomGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/a16/zyx/TransUNet-main/prompt/image/clintox', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--label_file', type=str,
                    default='/home/a16/zyx/TransUNet-main/prompt/labels/clintoxlabel.txt', help='label file path')
parser.add_argument('--image_list_file', type=str,
                    default='/home/a16/zyx/TransUNet-main/lists/prompt/clintoxlist.txt', help='image list file path')
parser.add_argument('--num_classes', type=int,
                    default=10, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=50, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_size', type=int,  default=512,
                    help='input patch size')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    args.dataset = 'Synapse'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/home/a16/zyx/TransUNet-main/prompt/image/clintox',
            'label_file': '/home/a16/zyx/TransUNet-main/prompt/labels/clintoxlabel.txt',
            'image_list_file': '/home/a16/zyx/TransUNet-main/lists/prompt/clintoxlist.txt',
            'num_classes': 4,  # 根据combine_labels函数的输出类别数
        },
        'own': {
            'root_path': '/home/a16/zyx/TransUNet-main/datasets/glass',
            'list_dir': '/home/a16/zyx/TransUNet-main/lists/lists_Synapse',
            'num_classes': 2,
        }
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.label_file = dataset_config[dataset_name]['label_file']
    args.image_list_file = dataset_config[dataset_name]['image_list_file']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "/home/a16/zyx/TransUNet-main/model/prompt/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    net.load_from(weights=np.load(config_vit.pretrained_path))

    transform = RandomGenerator(output_size=(args.img_size, args.img_size), device='cuda')
    train_dataset = MoleculeDataset(args.root_path, args.image_list_file, args.label_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    trainer = {'Synapse': trainer_synapse, }
    trainer['Synapse'](args, net, snapshot_path, train_loader)