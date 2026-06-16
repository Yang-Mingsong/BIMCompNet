import os

import torch
import torchvision.transforms as transforms
import yaml
from dgl.dataloading import GraphDataLoader
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from data.custom_dataset import IfcFileGraphsDataset, MultiViewDataset, ObjDataset, VoxDataset, DgcnnDataset
from model.dgcnn import DGCNN
from model.mesh_net import MeshNet
from model.mvcnn import SVCNN, MVCNN
from model.rgcn import RGCN
from model.voxnet import VoxNet
from scripts.test import RGCNTester, MVCNNTester, MESHNETTester, DGCNNTester, VOXNETTester
from scripts.train import RGCNTrainer, MVCNNTrainer, MESHNETTrainer, DGCNNTrainer, VOXNETTrainer
from utils.early_stopping import EarlyStopping

# configs
with open(r'Q:\pychem_project\BIMCompNet\configs\config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
ifg_Config = config.get('RGCN')
img_Config = config.get('MVCNN')
obj_Config = config.get('MESHNET')
vox_Config = config.get('VOXNET')
ply_Config = config.get('DGCNN')


def rgcn_validation_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ifg_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    print('Creating graph...')
    test_dataset = IfcFileGraphsDataset(name='test_dataset', root=ifg_Config['data_dir'], data_type='test', sample=ifg_Config['sample'],
                                        c_dir=ifg_Config['ifc_file_graph_cache'],
                                        isg=ifg_Config['isg_name'])
    test_loader = GraphDataLoader(test_dataset, batch_size=ifg_Config['test_batch_size'], drop_last=False, shuffle=False)

    classes = test_dataset.classes
    metagraph = test_dataset.re_metagraph
    print('Creating model...')
    model = RGCN(ifg_Config['in_feats'], ifg_Config['hidden_feats'], len(classes), metagraph).to(device)
    pretrained_weights = torch.load(ifg_Config['model_weights_path'], weights_only=True)
    model.load_state_dict(pretrained_weights)

    loss = nn.CrossEntropyLoss()
    tester = RGCNTester(model, test_loader, loss)
    tester.test(1, device, None, ifg_Config['test_batch_size'], '')
    torch.cuda.empty_cache()


def mvcnn_validation_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = img_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    tset_dataset = MultiViewDataset(img_Config['data_dir'], data_type='test', sample=img_Config['sample'], view_name=img_Config['view_name'],
                                    data_transform=transform)
    test_loader = DataLoader(tset_dataset, num_workers=0, batch_size=img_Config['test_batch_size'], shuffle=False, drop_last=False)

    print('Creating model...')
    classes = tset_dataset.classes
    svcnn = SVCNN(num_classes=len(classes), pretraining=img_Config['pretraining'], cnn_name=img_Config['svcnn']).to(device)
    mvcnn = MVCNN(svcnn_model=svcnn, num_classes=len(classes), num_views=img_Config['num_views']).to(device)
    del svcnn
    pretrained_weights = torch.load(img_Config['model_weights_path'], weights_only=True)
    mvcnn.load_state_dict(pretrained_weights)

    tester = MVCNNTester(mvcnn, test_loader, nn.CrossEntropyLoss())
    tester.test(1, device, None, img_Config['test_batch_size'], '')


def meshnet_validation_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = obj_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    tset_dataset = ObjDataset(obj_Config['data_dir'], data_type='test', sample=obj_Config['sample'], cache_dir=obj_Config['cache_dir'],
                              max_faces=obj_Config['max_faces'], seed=obj_Config['seed'])
    test_loader = DataLoader(tset_dataset, num_workers=0, batch_size=obj_Config['test_batch_size'], shuffle=False, drop_last=False)

    print('Creating model...')
    classes = tset_dataset.classes
    meshnet = MeshNet(num_classes=len(classes), mesh_convolution=obj_Config['mesh_convolution'], mask_ratio=obj_Config['mask_ratio'],
                      dropout=obj_Config['dropout'], require_fea=False).to(device)
    pretrained_weights = torch.load(obj_Config['model_weights_path'], weights_only=True)
    meshnet.load_state_dict(pretrained_weights)

    loss = nn.CrossEntropyLoss()
    tester = MESHNETTester(meshnet, test_loader, loss)
    tester.test(1, device, None, obj_Config['test_batch_size'], '')


def voxnet_validation_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = vox_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    tset_dataset = VoxDataset(root=vox_Config['data_dir'], data_type='test', sample=vox_Config['sample'], cache_dir=vox_Config['cache_dir'])
    test_loader = DataLoader(tset_dataset, num_workers=0, batch_size=vox_Config['test_batch_size'], shuffle=False, drop_last=False)

    print('Creating model...')
    classes = tset_dataset.classes
    voxnet = VoxNet(num_classes=len(classes)).to(device)
    pretrained_weights = torch.load(vox_Config['model_weights_path'], weights_only=True)
    voxnet.load_state_dict(pretrained_weights)

    loss = nn.CrossEntropyLoss()
    tester = VOXNETTester(voxnet, test_loader, loss)
    tester.test(1, device, None, vox_Config['test_batch_size'], '')


def dgcnn_validation_main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ply_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    test_dataset = DgcnnDataset(ply_Config['data_dir'], data_type='test', sample=ply_Config['sample'], cache_dir=ply_Config['cache_dir'],
                                point_num=ply_Config['point_num'], seed=ply_Config['seed'])
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=ply_Config['test_batch_size'], shuffle=False, drop_last=False)

    classes = test_dataset.classes
    dgcnn = DGCNN(num_classes=len(classes), dropout=ply_Config['dropout'], k=ply_Config['k'], emb_dims=ply_Config['emb_dims']).to(device)
    pretrained_weights = torch.load(ply_Config['model_weights_path'], weights_only=True)
    dgcnn.load_state_dict(pretrained_weights)

    loss = nn.CrossEntropyLoss()
    tester = DGCNNTester(dgcnn, test_loader, loss)
    tester.test(1, device, None, ply_Config['test_batch_size'], '')


if __name__ == '__main__':
    # rgcn_validation_main()
    # meshnet_validation_main()
    # mvcnn_validation_main()
    voxnet_validation_main()
    # dgcnn_validation_main()
