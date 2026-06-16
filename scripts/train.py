import os
import dgl
import numpy as np
import torch
import wandb
import torch.nn.functional as F
import sklearn.metrics as metrics
from tqdm import tqdm


class RGCNTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (batched_graph, labels) in enumerate(tqdm_batch):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outs = self.model(batched_graph)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc, train_balanced_acc))
        torch.cuda.empty_cache()


class MVCNNTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (inputs, targets) in enumerate(tqdm_batch):
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs = torch.from_numpy(inputs)
            N, V, C, H, W = inputs.size()
            inputs = inputs.view(-1, C, H, W)
            inputs, targets = inputs.cuda(device), targets.cuda(device)

            # compute output
            out = self.model(inputs)
            loss = self.loss_fn(out, targets)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = out.max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(targets.cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc, train_balanced_acc))


class MESHNETTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (labels, batched_obj) in enumerate(tqdm_batch):
            if isinstance(batched_obj, list):
                batched_obj = [item.to(device) for item in batched_obj]
            else:
                batched_obj = batched_obj.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outs = self.model(batched_obj)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc,
                                                                                                      train_balanced_acc))
        torch.cuda.empty_cache()


class POINTNETTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (labels, batched_ply) in enumerate(tqdm_batch):
            batched_ply = batched_ply.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outs = self.model(batched_ply)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc,
                                                                                                      train_balanced_acc))
        torch.cuda.empty_cache()


class DGCNNTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (labels, batched_pg) in enumerate(tqdm_batch):
            batched_pg = batched_pg.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outs = self.model(batched_pg)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc,
                                                                                                      train_balanced_acc))
        torch.cuda.empty_cache()


class VOXNETTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (labels, batched_vox) in enumerate(tqdm_batch):
            batched_vox = batched_vox.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outs = self.model(batched_vox)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc,
                                                                                                      train_balanced_acc))
        torch.cuda.empty_cache()
