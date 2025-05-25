import os
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import torch
import wandb


class RGCNTester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.best_test_pr = 0.0
        self.best_test_f1 = 0.0
        self.best_test_ba = 0.0
        self.best_test_loss = 1.0

    def test(self, epoch, device, early_stopping, test_batch_size, save_path):
        tqdm_batch = tqdm(self.test_loader, desc='Epoch-{} testing'.format(epoch))
        self.model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []

        for i, (batched_graph, labels) in enumerate(tqdm_batch):
            with torch.no_grad():
                batched_graph, labels = batched_graph.to(device), labels.to(device)

                outs = self.model(batched_graph)
                loss = self.loss_fn(outs, labels)
                pred = outs.detach().max(dim=1)[1]

                test_loss += loss.item() * test_batch_size
                count += test_batch_size
                test_pred.append(pred.detach().cpu().numpy())
                test_true.append(labels.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Calculate metrics
        avg_test_loss = test_loss * 1.0 / count
        test_overall_acc = metrics.accuracy_score(test_true, test_pred)
        test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=1)
        test_f1_score = metrics.f1_score(test_true, test_pred, average='macro', zero_division=1)
        test_recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=1)
        # 计算每一类的 precision, recall, f1-score
        test_precision_per_class = metrics.precision_score(test_true, test_pred, average=None, zero_division=1)
        test_recall_per_class = metrics.recall_score(test_true, test_pred, average=None, zero_division=1)
        test_f1_score_per_class = metrics.f1_score(test_true, test_pred, average=None, zero_division=1)
        if early_stopping is None:
            print("Overall Accuracy:{}, Balanced Accuracy:{}, Precision{}, F1 Score:{}, "
                  "Test Recall:{}".format(test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
            print("Pre_class metrics")
            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                print("{}_precision:{}".format(class_name, test_precision_per_class[i]))
                print("{}_recall:{}".format(class_name, test_recall_per_class[i]))
                print("{}_f1-score:{}".format(class_name, test_f1_score_per_class[i]))
        else:
            early_stopping(test_loss, self.model)  # 达到早停止条件时，early_stop会被置为True

            if test_balanced_acc >= self.best_test_ba:
                self.best_test_ba = test_balanced_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'ba_rgcn_model.pth'))
            print('best_test_ba: {:.6f}'.format(self.best_test_ba))

            if test_precision >= self.best_test_pr:
                self.best_test_pr = test_precision
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'pr_rgcn_model.pth'))
            print('best_test_pr: {:.6f}'.format(self.best_test_pr))

            if test_f1_score >= self.best_test_f1:
                self.best_test_f1 = test_f1_score
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'f1_rgcn_model.pth'))
            print('best_test_f1: {:.6f}'.format(self.best_test_f1))

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "test/loss": avg_test_loss,
                "test/overall_acc": test_overall_acc,
                "test/balanced_acc": test_balanced_acc,
                "test/f1": test_f1_score,
                "test/precision": test_precision,
                "test/recall": test_recall,
            })

            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                wandb.log({
                    "{}/precision".format(class_name): test_precision_per_class[i],
                    "{}/recall".format(class_name): test_recall_per_class[i],
                    "{}/f1-score".format(class_name): test_f1_score_per_class[i]
                })

            print("Epoch{} - Test Loss:{}, Test Overall Accuracy:{}, Test Balanced Accuracy:{}, Test Precision{}, Test F1 Score:{}, "
                  "Test Recall:{}".format(epoch, avg_test_loss, test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))


class MVCNNTester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.best_test_pr = 0.0
        self.best_test_f1 = 0.0
        self.best_test_ba = 0.0
        self.best_test_loss = 1.0

    def test(self, epoch, device, early_stopping, test_batch_size, save_path):
        tqdm_batch = tqdm(self.test_loader, desc='Epoch-{} testing'.format(epoch))
        self.model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []
        for i, (inputs, targets) in enumerate(tqdm_batch):
            with torch.no_grad():
                inputs = np.stack(inputs, axis=1)  # 12,12,3,224,224
                inputs = torch.from_numpy(inputs)
                N, V, C, H, W = inputs.size()
                inputs = inputs.view(-1, C, H, W)
                inputs, targets = inputs.cuda(device), targets.cuda(device)

                # compute output
                out = self.model(inputs)
                loss = self.loss_fn(out, targets)
                pred = out.max(dim=1)[1]
                test_loss += loss.item() * test_batch_size
                test_pred.append(pred.detach().cpu().numpy())
                count += test_batch_size
                test_true.append(targets.cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Calculate metrics
        avg_test_loss = test_loss * 1.0 / count
        test_overall_acc = metrics.accuracy_score(test_true, test_pred)
        test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=1)
        test_f1_score = metrics.f1_score(test_true, test_pred, average='macro', zero_division=1)
        test_recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=1)
        # 计算每一类的 precision, recall, f1-score
        test_precision_per_class = metrics.precision_score(test_true, test_pred, average=None, zero_division=1)
        test_recall_per_class = metrics.recall_score(test_true, test_pred, average=None, zero_division=1)
        test_f1_score_per_class = metrics.f1_score(test_true, test_pred, average=None, zero_division=1)

        if early_stopping is None:
            print("Overall Accuracy:{}, Balanced Accuracy:{}, Precision:{}, F1 Score:{}, Test Recall:{}".format(test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
            print("Pre_class metrics")
            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            for i, class_name in enumerate(class_names):
                print("{}_precision:{}".format(class_name, test_precision_per_class[i]))
                print("{}_recall:{}".format(class_name, test_recall_per_class[i]))
                print("{}_f1-score:{}".format(class_name, test_f1_score_per_class[i]))
        else:
            early_stopping(test_loss, self.model)
            if test_balanced_acc >= self.best_test_ba:
                self.best_test_ba = test_balanced_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'ba_mvcnn_model.pth'))
            print('best_test_ba: {:.6f}'.format(self.best_test_ba))

            if test_precision >= self.best_test_pr:
                self.best_test_pr = test_precision
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'pr_mvcnn_model.pth'))
            print('best_test_pr: {:.6f}'.format(self.best_test_pr))

            if test_f1_score >= self.best_test_f1:
                self.best_test_f1 = test_f1_score
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'f1_mvcnn_model.pth'))
            print('best_test_f1: {:.6f}'.format(self.best_test_f1))

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "test/loss": avg_test_loss,
                "test/overall_acc": test_overall_acc,
                "test/balanced_acc": test_balanced_acc,
                "test/f1": test_f1_score,
                "test/precision": test_precision,
                "test/recall": test_recall,
            })

            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                wandb.log({
                    "{}/precision".format(class_name): test_precision_per_class[i],
                    "{}/recall".format(class_name): test_recall_per_class[i],
                    "{}/f1-score".format(class_name): test_f1_score_per_class[i]
                })

            print("Epoch{} - Test Loss:{}, Test Overall Accuracy:{}, Test Balanced Accuracy:{}, Test Precision{}, Test F1 Score:{}, "
                  "Test Recall:{}".format(epoch, avg_test_loss, test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))


class MESHNETTester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.best_test_pr = 0.0
        self.best_test_f1 = 0.0
        self.best_test_ba = 0.0
        self.best_test_loss = 1.0

    def test(self, epoch, device, early_stopping, test_batch_size, save_path):
        tqdm_batch = tqdm(self.test_loader, desc='Epoch-{} testing'.format(epoch))
        self.model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []

        for i, (labels, batched_obj) in enumerate(tqdm_batch):
            with torch.no_grad():
                if isinstance(batched_obj, list):
                    batched_obj = [item.to(device) for item in batched_obj]
                else:
                    batched_obj = batched_obj.to(device)
                labels = labels.to(device)

                outs = self.model(batched_obj)
                loss = self.loss_fn(outs, labels)
                pred = outs.detach().max(dim=1)[1]

                test_loss += loss.item() * test_batch_size
                count += test_batch_size
                test_pred.append(pred.detach().cpu().numpy())
                test_true.append(labels.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Calculate metrics
        avg_test_loss = test_loss * 1.0 / count
        test_overall_acc = metrics.accuracy_score(test_true, test_pred)
        test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=1)
        test_f1_score = metrics.f1_score(test_true, test_pred, average='macro', zero_division=1)
        test_recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=1)
        # 计算每一类的 precision, recall, f1-score
        test_precision_per_class = metrics.precision_score(test_true, test_pred, average=None, zero_division=1)
        test_recall_per_class = metrics.recall_score(test_true, test_pred, average=None, zero_division=1)
        test_f1_score_per_class = metrics.f1_score(test_true, test_pred, average=None, zero_division=1)

        if early_stopping is None:
            print("Overall Accuracy:{}, Balanced Accuracy:{}, Precision:{}, F1 Score:{}, Test Recall:{}".format(test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
            print("Pre_class metrics")
            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            for i, class_name in enumerate(class_names):
                #print("{}_precision:{}".format(class_name, test_precision_per_class[i]))
                #print("{}_recall:{}".format(class_name, test_recall_per_class[i]))
                print("{}_f1-score:{}".format(class_name, test_f1_score_per_class[i]))
        else:
            early_stopping(test_loss, self.model)  # 达到早停止条件时，early_stop会被置为True

            if test_balanced_acc >= self.best_test_ba:
                self.best_test_ba = test_balanced_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'ba_meshnet_model.pth'))
            print('best_test_ba: {:.6f}'.format(self.best_test_ba))

            if test_precision >= self.best_test_pr:
                self.best_test_pr = test_precision
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'pr_meshnet_model.pth'))
            print('best_test_pr: {:.6f}'.format(self.best_test_pr))

            if test_f1_score >= self.best_test_f1:
                self.best_test_f1 = test_f1_score
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'f1_meshnet_model.pth'))
            print('best_test_f1: {:.6f}'.format(self.best_test_f1))

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "test/loss": avg_test_loss,
                "test/overall_acc": test_overall_acc,
                "test/balanced_acc": test_balanced_acc,
                "test/f1": test_f1_score,
                "test/precision": test_precision,
                "test/recall": test_recall,
            })

            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                wandb.log({
                    "{}/precision".format(class_name): test_precision_per_class[i],
                    "{}/recall".format(class_name): test_recall_per_class[i],
                    "{}/f1-score".format(class_name): test_f1_score_per_class[i]
                })

            print("Epoch{} - Test Loss:{}, Test Overall Accuracy:{}, Test Balanced Accuracy:{}, Test Precision{}, Test F1 Score:{}, "
                  "Test Recall:{}".format(epoch, avg_test_loss, test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))


class POINTNETTester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.best_test_pr = 0.0
        self.best_test_f1 = 0.0
        self.best_test_ba = 0.0
        self.best_test_loss = 1.0

    def test(self, epoch, device, early_stopping, test_batch_size, save_path):
        tqdm_batch = tqdm(self.test_loader, desc='Epoch-{} testing'.format(epoch))
        self.model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []

        for i, (labels, batched_ply) in enumerate(tqdm_batch):
            with torch.no_grad():
                batched_ply = batched_ply.to(device)
                labels = labels.to(device)

                outs = self.model(batched_ply)
                loss = self.loss_fn(outs, labels)
                pred = outs.detach().max(dim=1)[1]

                test_loss += loss.item() * test_batch_size
                count += test_batch_size
                test_pred.append(pred.detach().cpu().numpy())
                test_true.append(labels.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Calculate metrics
        avg_test_loss = test_loss * 1.0 / count
        test_overall_acc = metrics.accuracy_score(test_true, test_pred)
        test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=1)
        test_f1_score = metrics.f1_score(test_true, test_pred, average='macro', zero_division=1)
        test_recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=1)
        # 计算每一类的 precision, recall, f1-score
        test_precision_per_class = metrics.precision_score(test_true, test_pred, average=None, zero_division=0)
        test_recall_per_class = metrics.recall_score(test_true, test_pred, average=None, zero_division=0)
        test_f1_score_per_class = metrics.f1_score(test_true, test_pred, average=None, zero_division=0)

        if early_stopping is None:
            print("Overall Accuracy:{}, Balanced Accuracy:{}, Precision:{}, F1 Score:{}, Test Recall:{}".format(test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
            print("Pre_class metrics")
            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            for i, class_name in enumerate(class_names):
                print("{}_precision:{}".format(class_name, test_precision_per_class[i]))
                print("{}_recall:{}".format(class_name, test_recall_per_class[i]))
                print("{}_f1-score:{}".format(class_name, test_f1_score_per_class[i]))
        else:
            early_stopping(test_loss, self.model)  # 达到早停止条件时，early_stop会被置为True

            if test_balanced_acc >= self.best_test_ba:
                self.best_test_ba = test_balanced_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'ba_pointnet_model.pth'))
            print('best_test_ba: {:.6f}'.format(self.best_test_ba))

            if test_precision >= self.best_test_pr:
                self.best_test_pr = test_precision
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'pr_pointnet_model.pth'))
            print('best_test_pr: {:.6f}'.format(self.best_test_pr))

            if test_f1_score >= self.best_test_f1:
                self.best_test_f1 = test_f1_score
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'f1_pointnet_model.pth'))
            print('best_test_f1: {:.6f}'.format(self.best_test_f1))

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "test/loss": avg_test_loss,
                "test/overall_acc": test_overall_acc,
                "test/balanced_acc": test_balanced_acc,
                "test/f1": test_f1_score,
                "test/precision": test_precision,
                "test/recall": test_recall,
            })

            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                wandb.log({
                    "{}/precision".format(class_name): test_precision_per_class[i],
                    "{}/recall".format(class_name): test_recall_per_class[i],
                    "{}/f1-score".format(class_name): test_f1_score_per_class[i]
                })

            print("Epoch{} - Test Loss:{}, Test Overall Accuracy:{}, Test Balanced Accuracy:{}, Test Precision{}, Test F1 Score:{}, "
                  "Test Recall:{}".format(epoch, avg_test_loss, test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))


class DGCNNTester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.best_test_pr = 0.0
        self.best_test_f1 = 0.0
        self.best_test_ba = 0.0
        self.best_test_loss = 1.0

    def test(self, epoch, device, early_stopping, test_batch_size, save_path):
        tqdm_batch = tqdm(self.test_loader, desc='Epoch-{} testing'.format(epoch))
        self.model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []

        for i, (labels, batched_pg) in enumerate(tqdm_batch):
            with torch.no_grad():
                batched_pg = batched_pg.to(device)
                labels = labels.to(device)

                outs = self.model(batched_pg)
                loss = self.loss_fn(outs, labels)
                pred = outs.detach().max(dim=1)[1]

                test_loss += loss.item() * test_batch_size
                count += test_batch_size
                test_pred.append(pred.detach().cpu().numpy())
                test_true.append(labels.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Calculate metrics
        avg_test_loss = test_loss * 1.0 / count
        test_overall_acc = metrics.accuracy_score(test_true, test_pred)
        test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=1)
        test_f1_score = metrics.f1_score(test_true, test_pred, average='macro', zero_division=1)
        test_recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=1)
        # 计算每一类的 precision, recall, f1-score
        test_precision_per_class = metrics.precision_score(test_true, test_pred, average=None, zero_division=0)
        test_recall_per_class = metrics.recall_score(test_true, test_pred, average=None, zero_division=0)
        test_f1_score_per_class = metrics.f1_score(test_true, test_pred, average=None, zero_division=0)

        if early_stopping is None:
            print("Overall Accuracy:{}, Balanced Accuracy:{}, Precision:{}, F1 Score:{}, Test Recall:{}".format(test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
            print("Pre_class metrics")
            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            for i, class_name in enumerate(class_names):
                print("{}_precision:{}".format(class_name, test_precision_per_class[i]))
                print("{}_recall:{}".format(class_name, test_recall_per_class[i]))
                print("{}_f1-score:{}".format(class_name, test_f1_score_per_class[i]))
        else:
            early_stopping(test_loss, self.model)  # 达到早停止条件时，early_stop会被置为True

            if test_balanced_acc >= self.best_test_ba:
                self.best_test_ba = test_balanced_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'ba_dgcnn_model.pth'))
            print('best_test_ba: {:.6f}'.format(self.best_test_ba))

            if test_precision >= self.best_test_pr:
                self.best_test_pr = test_precision
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'pr_dgcnn_model.pth'))
            print('best_test_pr: {:.6f}'.format(self.best_test_pr))

            if test_f1_score >= self.best_test_f1:
                self.best_test_f1 = test_f1_score
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'f1_dgcnn_model.pth'))
            print('best_test_f1: {:.6f}'.format(self.best_test_f1))

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "test/loss": avg_test_loss,
                "test/overall_acc": test_overall_acc,
                "test/balanced_acc": test_balanced_acc,
                "test/f1": test_f1_score,
                "test/precision": test_precision,
                "test/recall": test_recall,
            })

            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                wandb.log({
                    "{}/precision".format(class_name): test_precision_per_class[i],
                    "{}/recall".format(class_name): test_recall_per_class[i],
                    "{}/f1-score".format(class_name): test_f1_score_per_class[i]
                })

            print("Epoch{} - Test Loss:{}, Test Overall Accuracy:{}, Test Balanced Accuracy:{}, Test Precision{}, Test F1 Score:{}, "
                  "Test Recall:{}".format(epoch, avg_test_loss, test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))


class VOXNETTester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.best_test_pr = 0.0
        self.best_test_f1 = 0.0
        self.best_test_ba = 0.0
        self.best_test_loss = 1.0

    def test(self, epoch, device, early_stopping, test_batch_size, save_path):
        tqdm_batch = tqdm(self.test_loader, desc='Epoch-{} testing'.format(epoch))
        self.model.eval()
        test_loss = 0.0
        test_pred = []
        count = 0.0
        test_true = []

        for i, (labels, batched_vox) in enumerate(tqdm_batch):
            with torch.no_grad():
                batched_vox = batched_vox.to(device)
                labels = labels.to(device)

                outs = self.model(batched_vox)
                loss = self.loss_fn(outs, labels)
                pred = outs.detach().max(dim=1)[1]

                test_loss += loss.item() * test_batch_size
                count += test_batch_size
                test_pred.append(pred.detach().cpu().numpy())
                test_true.append(labels.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Calculate metrics
        avg_test_loss = test_loss * 1.0 / count
        test_overall_acc = metrics.accuracy_score(test_true, test_pred)
        test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=1)
        test_f1_score = metrics.f1_score(test_true, test_pred, average='macro', zero_division=1)
        test_recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=1)
        # 计算每一类的 precision, recall, f1-score
        test_precision_per_class = metrics.precision_score(test_true, test_pred, average=None, zero_division=0)
        test_recall_per_class = metrics.recall_score(test_true, test_pred, average=None, zero_division=0)
        test_f1_score_per_class = metrics.f1_score(test_true, test_pred, average=None, zero_division=0)

        if early_stopping is None:
            print("Overall Accuracy:{}, Balanced Accuracy:{}, Precision:{}, F1 Score:{}, Test Recall:{}".format(test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
            print("Pre_class metrics")
            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            for i, class_name in enumerate(class_names):
                print("{}_precision:{}".format(class_name, test_precision_per_class[i]))
                print("{}_recall:{}".format(class_name, test_recall_per_class[i]))
                print("{}_f1-score:{}".format(class_name, test_f1_score_per_class[i]))
        else:
            early_stopping(test_loss, self.model)  # 达到早停止条件时，early_stop会被置为True

            if test_balanced_acc >= self.best_test_ba:
                self.best_test_ba = test_balanced_acc
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'ba_voxnet_model.pth'))
            print('best_test_ba: {:.6f}'.format(self.best_test_ba))

            if test_precision >= self.best_test_pr:
                self.best_test_pr = test_precision
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'pr_voxnet_model.pth'))
            print('best_test_pr: {:.6f}'.format(self.best_test_pr))

            if test_f1_score >= self.best_test_f1:
                self.best_test_f1 = test_f1_score
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, 'f1_voxnet_model.pth'))
            print('best_test_f1: {:.6f}'.format(self.best_test_f1))

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "test/loss": avg_test_loss,
                "test/overall_acc": test_overall_acc,
                "test/balanced_acc": test_balanced_acc,
                "test/f1": test_f1_score,
                "test/precision": test_precision,
                "test/recall": test_recall,
            })

            class_names = ["Class_{}".format(i) for i in range(len(test_precision_per_class))]
            # 记录到 WandB
            for i, class_name in enumerate(class_names):
                wandb.log({
                    "{}/precision".format(class_name): test_precision_per_class[i],
                    "{}/recall".format(class_name): test_recall_per_class[i],
                    "{}/f1-score".format(class_name): test_f1_score_per_class[i]
                })

            print("Epoch{} - Test Loss:{}, Test Overall Accuracy:{}, Test Balanced Accuracy:{}, Test Precision{}, Test F1 Score:{}, "
                  "Test Recall:{}".format(epoch, avg_test_loss, test_overall_acc, test_balanced_acc, test_precision, test_f1_score, test_recall))
