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


def rgcn_main():
    # Initialize wandb
    wandb.init(project=ifg_Config['project_name'], name=ifg_Config['run_name'], config=ifg_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ifg_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    early_stopping = EarlyStopping(save_path=ifg_Config['save_path'] + '\\early_stopping', patience=ifg_Config['early_stopping'])

    # data
    print('Creating graph...')
    train_dataset = IfcFileGraphsDataset(name='train_dataset', root=ifg_Config['data_dir'], data_type='train', sample=ifg_Config['sample'],
                                         c_dir=ifg_Config['ifc_file_graph_cache'],
                                         isg=ifg_Config['isg_name'])
    train_loader = GraphDataLoader(train_dataset, batch_size=ifg_Config['batch_size'], drop_last=False, shuffle=True)
    test_dataset = IfcFileGraphsDataset(name='test_dataset', root=ifg_Config['data_dir'], data_type='test', sample=ifg_Config['sample'],
                                        c_dir=ifg_Config['ifc_file_graph_cache'],
                                        isg=ifg_Config['isg_name'])
    test_loader = GraphDataLoader(test_dataset, batch_size=ifg_Config['test_batch_size'], drop_last=False, shuffle=False)

    # model
    classes = train_dataset.classes
    metagraph = train_dataset.re_metagraph
    print('Creating model...')
    model = RGCN(ifg_Config['in_feats'], ifg_Config['hidden_feats'], len(classes), metagraph).to(device)
    pretrained_weights = torch.load(r"Q:\pychem_project\BIMCompNet\results\rgcn_output\test_500\f1_rgcn_model.pth", weights_only=True)  # 预训练权重文件
    model.load_state_dict(pretrained_weights)


    # Loss and Optimizer
    optimizer = Adam(model.parameters(), lr=ifg_Config['learning_rate'], weight_decay=ifg_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, ifg_Config['num_epochs'], eta_min=ifg_Config['eta_min'])

    # train and tes
    trainer = RGCNTrainer(model, train_loader, optimizer, loss)
    tester = RGCNTester(model, test_loader, loss)

    print('start training.')

    for epoch in range(1, ifg_Config['num_epochs'] + 1):
        trainer.train(epoch, device, ifg_Config['batch_size'])
        scheduler.step()
        if epoch % ifg_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, ifg_Config['test_batch_size'], ifg_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练

            torch.cuda.empty_cache()


def mvcnn_main():
    # Initialize wandb
    wandb.init(project=img_Config['project_name'], name=img_Config['run_name'], config=img_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = img_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    early_stopping = EarlyStopping(save_path=img_Config['save_path'] + '\\early_stopping', patience=img_Config['early_stopping'])

    # data
    train_dataset = MultiViewDataset(img_Config['data_dir'], data_type='train', sample=img_Config['sample'], view_name=img_Config['view_name'], data_transform=transform)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=img_Config['batch_size'], shuffle=True, drop_last=False)

    tset_test = MultiViewDataset(img_Config['data_dir'], data_type='test', sample=img_Config['sample'], view_name=img_Config['view_name'], data_transform=transform)
    test_loader = DataLoader(tset_test, num_workers=0, batch_size=img_Config['test_batch_size'], shuffle=False, drop_last=False)

    # model
    print('Creating model...')
    classes = train_dataset.classes
    svcnn = SVCNN(num_classes=len(classes), pretraining=img_Config['pretraining'], cnn_name=img_Config['svcnn']).to(device)
    mvcnn = MVCNN(svcnn_model=svcnn, num_classes=len(classes), num_views=img_Config['num_views']).to(device)
    del svcnn

    # Loss and Optimizer
    optimizer = Adam(mvcnn.parameters(), lr=img_Config['learning_rate'], weight_decay=img_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, img_Config['num_epochs'], img_Config['eta_min'])

    # train and test
    trainer = MVCNNTrainer(mvcnn, train_loader, optimizer, loss)
    tester = MVCNNTester(mvcnn, test_loader, loss)

    print('start training.')
    for epoch in range(1, img_Config['num_epochs'] + 1):
        trainer.train(epoch, device, img_Config['batch_size'])
        scheduler.step()
        if epoch % img_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, img_Config['test_batch_size'], img_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练


def mesh_net_main():
    # Initialize wandb
    wandb.init(project=obj_Config['project_name'], name=obj_Config['run_name'], config=obj_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = obj_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    early_stopping = EarlyStopping(save_path=obj_Config['save_path'] + '\\early_stopping', patience=obj_Config['early_stopping'])

    # data
    train_dataset = ObjDataset(obj_Config['data_dir'], data_type='train', sample=obj_Config['sample'], cache_dir=obj_Config['cache_dir'],
                               max_faces=obj_Config['max_faces'], seed=obj_Config['seed'])
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=obj_Config['batch_size'], shuffle=True, drop_last=False)

    tset_test = ObjDataset(obj_Config['data_dir'], data_type='test', sample=obj_Config['sample'], cache_dir=obj_Config['cache_dir'],
                           max_faces=obj_Config['max_faces'], seed=obj_Config['seed'])
    test_loader = DataLoader(tset_test, num_workers=0, batch_size=obj_Config['test_batch_size'], shuffle=False, drop_last=False)

    # model
    print('Creating model...')
    classes = train_dataset.classes
    meshnet = MeshNet(num_classes=len(classes), mesh_convolution=obj_Config['mesh_convolution'], mask_ratio=obj_Config['mask_ratio'],
                      dropout=obj_Config['dropout'], require_fea=False).to(device)

    # Loss and Optimizer
    optimizer = Adam(meshnet.parameters(), lr=obj_Config['learning_rate'], weight_decay=obj_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, obj_Config['num_epochs'], obj_Config['eta_min'])

    # train and test
    trainer = MESHNETTrainer(meshnet, train_loader, optimizer, loss)
    tester = MESHNETTester(meshnet, test_loader, loss)

    print('start training.')
    for epoch in range(1, obj_Config['num_epochs'] + 1):
        trainer.train(epoch, device, obj_Config['batch_size'])
        scheduler.step()
        if epoch % obj_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, obj_Config['test_batch_size'], obj_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练


def vox_net_main():
    # Initialize wandb
    wandb.init(project=vox_Config['project_name'], name=vox_Config['run_name'], config=vox_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = vox_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    early_stopping = EarlyStopping(save_path=vox_Config['save_path'] + '\\early_stopping', patience=vox_Config['early_stopping'])

    # data
    train_dataset = VoxDataset(root=vox_Config['data_dir'], data_type='train', sample=vox_Config['sample'], cache_dir=vox_Config['cache_dir'])
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=vox_Config['batch_size'], shuffle=True, drop_last=False)
    tset_test = VoxDataset(root=vox_Config['data_dir'], data_type='test', sample=vox_Config['sample'], cache_dir=vox_Config['cache_dir'])
    test_loader = DataLoader(tset_test, num_workers=0, batch_size=vox_Config['test_batch_size'], shuffle=False, drop_last=False)

    # model
    print('Creating model...')
    classes = train_dataset.classes
    voxnet = VoxNet(num_classes=len(classes)).to(device)

    # Loss and Optimizer
    optimizer = Adam(voxnet.parameters(), lr=vox_Config['learning_rate'], weight_decay=vox_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, vox_Config['num_epochs'], vox_Config['eta_min'])

    # train and test
    trainer = VOXNETTrainer(voxnet, train_loader, optimizer, loss)
    tester = VOXNETTester(voxnet, test_loader, loss)

    print('start training...')
    for epoch in range(1, vox_Config['num_epochs'] + 1):
        trainer.train(epoch, device, vox_Config['batch_size'])
        scheduler.step()
        if epoch % vox_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, vox_Config['test_batch_size'], vox_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练


def dgcnn_main():
    # Initialize wandb
    wandb.init(project=ply_Config['project_name'], name=ply_Config['run_name'], config=ply_Config)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ply_Config['gpu_device']
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    early_stopping = EarlyStopping(save_path=ply_Config['save_path'] + '\\early_stopping', patience=ply_Config['early_stopping'])

    # data
    train_dataset = DgcnnDataset(ply_Config['data_dir'], data_type='train', sample=ply_Config['sample'], cache_dir=ply_Config['cache_dir'],
                                 point_num=ply_Config['point_num'], seed=ply_Config['seed'])
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=ply_Config['batch_size'], shuffle=True, drop_last=False)

    tset_test = DgcnnDataset(ply_Config['data_dir'], data_type='test', sample=ply_Config['sample'], cache_dir=ply_Config['cache_dir'],
                             point_num=ply_Config['point_num'], seed=ply_Config['seed'])
    test_loader = DataLoader(tset_test, num_workers=0, batch_size=ply_Config['test_batch_size'], shuffle=False, drop_last=False)

    # model
    print('Creating model...')
    classes = train_dataset.classes
    dgcnn = DGCNN(num_classes=len(classes), dropout=ply_Config['dropout'], k=ply_Config['k'], emb_dims=ply_Config['emb_dims']).to(device)

    # Loss and Optimizer
    optimizer = Adam(dgcnn.parameters(), lr=ply_Config['learning_rate'], weight_decay=ply_Config['weight_decay'], betas=(0.9, 0.999))
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, ply_Config['num_epochs'], ply_Config['eta_min'])

    # train and test
    trainer = DGCNNTrainer(dgcnn, train_loader, optimizer, loss)
    tester = DGCNNTester(dgcnn, test_loader, loss)

    print('start training...')
    for epoch in range(1, ply_Config['num_epochs'] + 1):
        trainer.train(epoch, device, ply_Config['batch_size'])
        scheduler.step()
        if epoch % ply_Config['valid_freq'] == 0:
            tester.test(epoch, device, early_stopping, ply_Config['test_batch_size'], ply_Config['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练


if __name__ == '__main__':
    rgcn_main()
    # mesh_net_main()
    # mvcnn_main()
    # vox_net_main()
    #dgcnn_main()
