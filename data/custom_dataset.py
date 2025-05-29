import pandas as pd
import torch
import os
from dgl.data import DGLDataset
from torch.utils.data.dataset import Dataset
import dgl
from collections import defaultdict
from dgl import save_graphs, load_graphs
import re
import binvox_rw
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from PIL import Image
import diskcache as dc
import numpy as np
from torchvision import transforms
from plyfile import PlyData
from utils.objutils import process_mesh
import math
import pickle


def find_classes(classes_df):
    classes = classes_df.drop_duplicates().sort_values().tolist()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class IfcFileGraphsDataset(DGLDataset):
    def __init__(self, name, root, data_type, sample, c_dir, isg):
        self.root = root
        self.data_type = data_type
        self.sample = sample
        self._cache_path = c_dir
        self.model_flag = []
        self.graphs = []
        self.labels = []
        self.isg = isg
        self.metagraph = [('class_node', 'hasAttribute', 'attribute_node'),
                          ('attribute_node', 'hasValue', 'class_node'),
                          ('attribute_node', 'hasValue', 'type_node'),
                          ('attribute_node', 'hasValue', 'value_node'),
                          ('type_node', 'hasValue', 'class_node'),
                          ('type_node', 'hasValue', 'type_node'),
                          ('type_node', 'hasValue', 'value_node'),
                          ('class_node', 'selfLoop', 'class_node'),
                          ('attribute_node', 'selfLoop', 'attribute_node'),
                          ('type_node', 'selfLoop', 'type_node'),
                          ('value_node', 'selfLoop', 'value_node')]
        self.re_metagraph = [('attribute_node', 'hasAttribute', 'class_node'),
                             ('class_node', 'hasValue', 'attribute_node'),
                             ('type_node', 'hasValue', 'attribute_node'),
                             ('value_node', 'hasValue', 'attribute_node'),
                             ('class_node', 'hasValue', 'type_node'),
                             ('type_node', 'hasValue', 'type_node'),
                             ('value_node', 'hasValue', 'type_node'),
                             ('class_node', 'selfLoop', 'class_node'),
                             ('attribute_node', 'selfLoop', 'attribute_node'),
                             ('type_node', 'selfLoop', 'type_node'),
                             ('value_node', 'selfLoop', 'value_node')]
        self.sample_file = './data/{}_{}.csv'.format(data_type, sample)
        _df = pd.read_csv(self.sample_file)
        self.classes, self.class_to_idx = find_classes(_df['category'])

        self.cache_dir = r"Q:\pychem_project\BIMCompNet\data\rgcn_data_cache\temp"

        super(IfcFileGraphsDataset, self).__init__(name)

    @staticmethod
    def _is_number(obj: str) -> bool:
        pattern = r'^[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?$'
        return bool(re.match(pattern, obj))

    @staticmethod
    def _is_integer(obj: str) -> bool:
        # 匹配整数的正则表达式
        pattern = r'^[-+]?[0-9]+$'
        return bool(re.match(pattern, obj))

    @staticmethod
    def _is_bool(obj: str) -> bool:
        return obj in {'True', 'False'}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + torch.exp(-torch.tensor(x)))

    def _process_tasks(self, tasks, timeout=60, num_workers=4):
        """
        核心并发处理一组 (label, graph_path) 任务，
        返回一个 dict: { label: [ (model_flag, graph, label), ... ], ... }
        """
        results = defaultdict(list)
        pending = list(tasks)
        attempt = 0

        while pending:
            attempt += 1
            print(f"  [batch] 第 {attempt} 轮重试 {len(pending)} 个任务…")
            failed = []

            # 每轮新建池子
            executor = ProcessPoolExecutor(max_workers=num_workers)
            future_to_task = {
                executor.submit(self._process_label_path, t): t
                for t in pending
            }

            for fut, task in future_to_task.items():
                label, path = task
                try:
                    res = fut.result(timeout=timeout)
                except TimeoutError:
                    print(f"    [WARN] 第{attempt}轮超时：{task}")
                    fut.cancel()
                    failed.append(task)
                except Exception as e:
                    print(f"    [WARN] 第{attempt}轮失败：{task} -> {e}")
                    failed.append(task)
                else:
                    # 确保子进程返回的是三元组
                    if isinstance(res, (list, tuple)) and len(res) == 1:
                        res = res[0]
                    results[label].append(res)

            # 立即关掉，这样不会被卡死
            executor.shutdown(cancel_futures=True, wait=False)
            pending = failed

        return results

    def _process_label_path(self, label_path):
        label, _path = label_path
        graphs = []
        a = ''
        for root, dirs, files in os.walk(_path):
            node_file_path = ''
            edge_file_path = ''
            bin_file_path = ''
            for file in files:
                if file.endswith('geo_node.csv'):
                    node_file_path = os.path.join(root, file)
                elif file.endswith('geo_edge.csv'):
                    edge_file_path = os.path.join(root, file)
                elif file.endswith(self.isg):
                    bin_file_path = os.path.join(root, file)

            if node_file_path != '' and edge_file_path != '':
                parts = root.split('\\')
                _model_flag = parts[4]
                _shared_embedding_information = torch.load(str(bin_file_path), weights_only=False)
                graph = self._load_hetero_graph_from_csv(edge_file_path, node_file_path, label, _shared_embedding_information, _model_flag)
                graphs.append(graph)
                a = node_file_path
        print(' {}已完成'.format(a))
        return graphs

    def process(self):
        _df = pd.read_csv(self.sample_file)
        label_path_list = []
        if "00" in self.root:
            for category in os.listdir(self.root):
                category_path = os.path.join(self.root, category)
                for instance in os.listdir(category_path):
                    instance_path = os.path.join(self.root, category, instance)
                    graph_path = os.path.join(instance_path, 'GRAPH')
                    label_path_list.append((category, graph_path))
        else:
            for category, instance_path in _df.itertuples(index=False, name=None):
                graph_path = os.path.join(instance_path, 'GRAPH')
                label_path_list.append((category, graph_path))

        N = 8
        total = len(label_path_list)
        chunk_size = math.ceil(total / N)
        cache_dir = os.path.join(self.cache_dir, 'graph_batches')
        os.makedirs(cache_dir, exist_ok=True)

        for i in range(N):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total)
            chunk = label_path_list[start:end]
            if not chunk:
                break
            print(f"[Main] 开始处理第 {i + 1}/{N} 批，共 {len(chunk)} 个任务")
            sub_results = self._process_tasks(chunk,
                                              timeout=60,
                                              num_workers=4)

            cache_path = os.path.join(cache_dir, f"results_batch_{i + 1}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(sub_results, f)
            print(f"[Main] 已缓存第 {i + 1} 批结果到 {cache_path}")
            del sub_results

        final_results = defaultdict(list)
        for i in range(1, N + 1):
            cache_path = os.path.join(cache_dir, f"results_batch_{i}.pkl")
            if not os.path.isfile(cache_path):
                continue
            with open(cache_path, 'rb') as f:
                sub = pickle.load(f)
            for label, lst in sub.items():
                final_results[label].extend(lst)

        for res_list in final_results.values():
            for model_flag, g, label in res_list:
                self.model_flag.append(model_flag)
                self.graphs.append(g)
                self.labels.append(label)
        print("[Main] 全部 batch 合并完毕，共加载图:", len(self.graphs))

    def _load_hetero_graph_from_csv(self, edge_file_path, node_file_path, label, _shared_embedding_information, model_flag):
        print('{}开始'.format(node_file_path))
        # 定义所有可能的边类型
        all_edge_types = self.metagraph

        # 初始化包含所有可能边类型的字典，所有边类型对应的值初始为空列表
        edge_dict = {etype: ([], []) for etype in all_edge_types}

        edges = pd.read_csv(edge_file_path, header=None)
        # 根据CSV中的数据填充边字典
        for _, row in edges.iterrows():
            src_type, src_id, dst_type, dst_id, edge_type = row
            edge_key = (src_type, edge_type, dst_type)
            edge_dict[edge_key][0].append(src_id)
            edge_dict[edge_key][1].append(dst_id)

        # 创建异构图
        g = dgl.heterograph(edge_dict)

        # 按照IFCSchema嵌入表示结果初始化特征
        nodes_infor = pd.read_csv(node_file_path, header=None)
        node_feat_dict = defaultdict(dict)
        for seed_flag, row in nodes_infor.iterrows():
            node_type, node_id, item = row

            if node_type == 'class_node':
                features = _shared_embedding_information['class_node'][item]
            elif node_type == 'attribute_node':
                features = _shared_embedding_information['attribute_node'][item]
            elif node_type == 'type_node':
                if item == 'TUPLE':
                    features = _shared_embedding_information['type_node']['LIST']
                elif item == 'TUPLETUPLE':
                    features = _shared_embedding_information['type_node']['LISTLIST']
                else:
                    features = _shared_embedding_information['type_node'][item]
            elif node_type == 'value_node':
                if item.upper() in _shared_embedding_information['value_node'].keys():
                    features = _shared_embedding_information['value_node'][item.upper()]
                elif self._is_integer(str(item)):
                    scaled_value = self._sigmoid(float(item))
                    features = scaled_value * _shared_embedding_information['type_node']['INTEGER']
                elif self._is_number(str(item)):
                    scaled_value = self._sigmoid(float(item))
                    features = scaled_value * _shared_embedding_information['type_node']['REAL']
                elif self._is_bool(str(item)):
                    scaled_value = 1.0 if item == 'True' else 0.0
                    features = scaled_value * _shared_embedding_information['type_node']['BOOLEAN']
                else:
                    features = _shared_embedding_information['type_node']['STRING']
            else:
                print(node_type, node_id, item)
                features = _shared_embedding_information[node_type][item]

            node_feat_dict[node_type][node_id] = features
        for ntype in node_feat_dict:
            node_ids = list(node_feat_dict[ntype].keys())
            features_list = [node_feat_dict[ntype][nid] for nid in node_ids]
            feats = torch.stack(features_list)
            if g.num_nodes(ntype) != len(feats):
                print("不相等：Node type: {}, Number of nodes: {}, Number of features: {}".format(ntype, g.num_nodes(ntype), len(feats)))
            g.nodes[ntype].data['features'] = feats
        label = self.class_to_idx[label]
        _graph = (model_flag, g, label)
        return _graph

    def save(self):
        os.makedirs(self._cache_path, exist_ok=True)
        graph_path = os.path.join(self._cache_path, self.data_type, 'graphs.bin')
        label_path = os.path.join(self._cache_path, self.data_type, 'labels.pt')
        flag_path = os.path.join(self._cache_path, self.data_type, 'flags.pt')
        sorted_data = sorted(zip(self.model_flag, self.graphs, self.labels), key=lambda x: x[0])

        self.model_flag, self.graphs, self.labels = map(list, zip(*sorted_data))

        save_graphs(str(graph_path), self.graphs)
        torch.save(self.labels, str(label_path))
        torch.save(self.model_flag, str(flag_path))

    def load(self):
        graph_path = os.path.join(self._cache_path, self.data_type, 'graphs.bin')
        label_path = os.path.join(self._cache_path, self.data_type, 'labels.pt')
        flag_path = os.path.join(self._cache_path, self.data_type, 'flags.pt')
        self.graphs, _ = load_graphs(str(graph_path))
        self.labels = torch.load(str(label_path), weights_only=True)
        self.model_flag = torch.load(str(flag_path), weights_only=True)

    def has_cache(self):
        graph_path = os.path.join(self._cache_path, self.data_type, 'graphs.bin')
        label_path = os.path.join(self._cache_path, self.data_type, 'labels.pt')
        flag_path = os.path.join(self._cache_path, self.data_type, 'flags.pt')
        return os.path.exists(graph_path) and os.path.exists(label_path) and os.path.exists(flag_path)

    def get_model_flag(self, index):
        model_flag = self.model_flag[index]
        return model_flag

    def __getitem__(self, index):
        g = dgl.reverse(self.graphs[index], copy_ndata=True, copy_edata=True)
        return g, self.labels[index]

    def __len__(self):
        return len(self.graphs)


class MultiViewDataset(Dataset):
    def __init__(self, root, data_type, sample, view_name, data_transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root
        self.sample = sample
        self.view_name = view_name
        self.data_transform = data_transform
        self.target_transform = target_transform

        _sample_file = './data/{}_{}.csv'.format(data_type, sample)
        _df = pd.read_csv(_sample_file)
        self.classes, self.class_to_idx = find_classes(_df['category'])

        if "00" in self.root:
            for category in os.listdir(self.root):
                category_path = os.path.join(self.root, category)
                for instance in os.listdir(category_path):
                    instance_path = os.path.join(self.root, category, instance)
                    views = []
                    view_path = os.path.join(instance_path, 'PNG', self.view_name)
                    for view in os.listdir(str(view_path)):
                        img_path = os.path.join(str(view_path), view)
                        views.append(img_path)
                    if view_name == 'ArchShapesNet':
                        top_img_path = os.path.join(str(instance_path), r'PNG\Faces\0.png')
                        down_img_path = os.path.join(str(instance_path), r'PNG\Faces\1.png')
                        views.append(top_img_path)
                        views.append(down_img_path)
                    self.x.append(views)
                    self.y.append(self.class_to_idx[category])
        else:
            for category, instance_path in _df.itertuples(index=False, name=None):
                views = []
                view_path = os.path.join(instance_path, 'PNG', self.view_name)
                for view in os.listdir(str(view_path)):
                    img_path = os.path.join(str(view_path), view)
                    views.append(img_path)
                if view_name == 'ArchShapesNet':
                    top_img_path = os.path.join(str(instance_path), r'PNG\Faces\0.png')
                    down_img_path = os.path.join(str(instance_path), r'PNG\Faces\1.png')
                    views.append(top_img_path)
                    views.append(down_img_path)
                self.x.append(views)
                self.y.append(self.class_to_idx[category])

    def __getitem__(self, index):
        org_views = self.x[index]
        views = []
        for view in org_views:
            im = Image.open(str(view))
            im = im.convert('RGB')
            if self.data_transform is not None:
                im = self.data_transform(im)
            views.append(im)
        return views, self.y[index]

    def __len__(self):
        return len(self.x)


class ObjDataset(Dataset):
    def __init__(self, root: str, data_type: str, sample: str, cache_dir: str, max_faces, seed):
        super(ObjDataset, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.x = []
        self.y = []
        self.root = root
        self.sample = sample
        self.tensor_cache = dc.Cache(os.path.join(cache_dir, data_type))
        self.data_type = data_type
        self.max_faces = max_faces

        _sample_file = './data/{}_{}.csv'.format(data_type, sample)
        _df = pd.read_csv(_sample_file)
        self.classes, self.class_to_idx = find_classes(_df['category'])

        if "00" in self.root:
            for category in os.listdir(self.root):
                category_path = os.path.join(self.root, category)
                for instance in os.listdir(category_path):
                    instance_path = os.path.join(self.root, category, instance)
                    obj_path = ""
                    for obj_name in os.listdir(os.path.join(instance_path, 'OBJ')):
                        obj_path = os.path.join(instance_path, 'OBJ', obj_name)
                    self.x.append(obj_path)
                    self.y.append(self.class_to_idx[category])
        else:
            for category, instance_path in _df.itertuples(index=False, name=None):
                obj_path = ""
                for obj_name in os.listdir(os.path.join(instance_path, 'OBJ')):
                    obj_path = os.path.join(instance_path, 'OBJ', obj_name)
                self.x.append(obj_path)
                self.y.append(self.class_to_idx[category])

    def __getitem__(self, index):
        if index in self.tensor_cache:
            return self.tensor_cache.get(index)
        obj_dir = self.x[index]
        label = self.y[index]
        face, neighbor_index = process_mesh(obj_dir, self.max_faces)
        if face is None:
            y, x = self.__getitem__(0)
            self.tensor_cache.set(index, (y, x))
            return y, x
        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        y = torch.tensor(label, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)
        x = (centers, corners, normals, neighbor_index)
        self.tensor_cache.set(index, (y, x))
        return y, x

    def __len__(self):
        return len(self.x)


class PointDataset(Dataset):
    def __init__(self, root: str, data_type: str, sample: str, cache_dir: str, point_num, seed):
        super(PointDataset, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.x = []
        self.y = []
        self.root = root
        self.sample = sample
        self.tensor_cache = dc.Cache(os.path.join(cache_dir, data_type))
        self.data_type = data_type
        self.point_num = point_num
        self.transform = transforms.Compose([lambda x: self._down_sample_points(x), lambda x: self._normalize_points(x), transforms.ToTensor()])

        _sample_file = './data/{}_{}.csv'.format(data_type, sample)
        _df = pd.read_csv(_sample_file)
        self.classes, self.class_to_idx = find_classes(_df['category'])

        if "00" in self.root:
            for category in os.listdir(self.root):
                category_path = os.path.join(self.root, category)
                for instance in os.listdir(category_path):
                    instance_path = os.path.join(self.root, category, instance)
                    ply_path = ""
                    for ply_name in os.listdir(os.path.join(instance_path, 'PLY')):
                        ply_path = os.path.join(instance_path, 'PLY', ply_name)
                    self.x.append(ply_path)
                    self.y.append(self.class_to_idx[category])
        else:
            for category, instance_path in _df.itertuples(index=False, name=None):
                ply_path = ""
                for ply_name in os.listdir(os.path.join(instance_path, 'PLY')):
                    ply_path = os.path.join(instance_path, 'PLY', ply_name)
                self.x.append(ply_path)
                self.y.append(self.class_to_idx[category])

    def _down_sample_points(self, point_cloud):
        if point_cloud.shape[0] > self.point_num:
            # 使用随机选择点
            indices = np.random.choice(point_cloud.shape[0], self.point_num, replace=False)
            point_cloud = point_cloud[indices, :]
        return point_cloud

    @staticmethod
    def _normalize_points(point_cloud):
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        return point_cloud

    def __getitem__(self, index):
        if index in self.tensor_cache:
            return self.tensor_cache.get(index)
        ply_path, label = self.x[index], self.y[index]
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex'].data
        points = np.array([list(p) for p in vertices[['x', 'y', 'z']]])
        if self.transform:
            points = self.transform(points)
        x = points.transpose(2, 1).squeeze(0)
        y = torch.tensor(label)
        self.tensor_cache.set(index, (y, x))
        return y, x

    def __len__(self):
        return len(self.x)


class DgcnnDataset(Dataset):
    def __init__(self, root: str, data_type: str, sample: str, cache_dir: str, point_num, seed):
        super(DgcnnDataset, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.x = []
        self.y = []
        self.root = root
        self.sample = sample
        self.tensor_cache = dc.Cache(os.path.join(cache_dir, data_type))
        self.data_type = data_type
        self.point_num = point_num
        self.transform = transforms.Compose([lambda x: self._down_sample_points(x), lambda x: self._normalize_points(x), transforms.ToTensor()])

        _sample_file = './data/{}_{}.csv'.format(data_type, sample)
        _df = pd.read_csv(_sample_file)
        self.classes, self.class_to_idx = find_classes(_df['category'])

        if "00" in self.root:
            for category in os.listdir(self.root):
                category_path = os.path.join(self.root, category)
                for instance in os.listdir(category_path):
                    instance_path = os.path.join(self.root, category, instance)
                    ply_path = ""
                    for ply_name in os.listdir(os.path.join(instance_path, 'PLY')):
                        ply_path = os.path.join(instance_path, 'PLY', ply_name)
                    self.x.append(ply_path)
                    self.y.append(self.class_to_idx[category])
        else:
            for category, instance_path in _df.itertuples(index=False, name=None):
                ply_path = ""
                for ply_name in os.listdir(os.path.join(instance_path, 'PLY')):
                    ply_path = os.path.join(instance_path, 'PLY', ply_name)
                self.x.append(ply_path)
                self.y.append(self.class_to_idx[category])

    def _down_sample_points(self, point_cloud):
        if point_cloud.shape[0] > self.point_num:
            # 使用随机选择点
            indices = np.random.choice(point_cloud.shape[0], self.point_num, replace=False)
            point_cloud = point_cloud[indices, :]
        return point_cloud

    @staticmethod
    def _normalize_points(point_cloud):
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        return point_cloud


    def __getitem__(self, index):
        if index in self.tensor_cache:
            return self.tensor_cache.get(index)
        ply_path, label = self.x[index], self.y[index]
        try:
            ply_data = PlyData.read(ply_path)
        except Exception as e:
            print(f"Failed to load PLY file: {ply_path}, error: {e}")
            raise
        vertices = ply_data['vertex'].data
        points = np.array([list(p) for p in vertices[['x', 'y', 'z']]])
        if self.transform:
            points = self.transform(points)
        x = points.transpose(2, 1).squeeze(0)
        y = torch.tensor(label)
        self.tensor_cache.set(index, (y, x))
        return y, x

    def __len__(self):
        return len(self.x)


class VoxDataset(Dataset):
    def __init__(self, root: str, data_type: str, sample: str, cache_dir: str):
        super(VoxDataset, self).__init__()
        self.x = []
        self.y = []
        self.root = root
        self.sample = sample
        self.tensor_cache = dc.Cache(os.path.join(cache_dir, data_type))
        self.data_type = data_type

        _sample_file = './data/{}_{}.csv'.format(data_type, sample)
        _df = pd.read_csv(_sample_file)
        self.classes, self.class_to_idx = find_classes(_df['category'])

        if "00" in self.root:
            for category in os.listdir(self.root):
                category_path = os.path.join(self.root, category)
                for instance in os.listdir(category_path):
                    instance_path = os.path.join(self.root, category, instance)
                    vox_path = ""
                    for vox_name in os.listdir(os.path.join(instance_path, 'VOX')):
                        vox_path = os.path.join(instance_path, 'VOX', vox_name)
                    self.x.append(vox_path)
                    self.y.append(self.class_to_idx[category])
        else:
            for category, instance_path in _df.itertuples(index=False, name=None):
                vox_path = ""
                for vox_name in os.listdir(os.path.join(instance_path, 'VOX')):
                    vox_path = os.path.join(instance_path, 'VOX', vox_name)
                self.x.append(vox_path)
                self.y.append(self.class_to_idx[category])

    def __getitem__(self, index):
        if index in self.tensor_cache:
            return self.tensor_cache.get(index)
        vox_dir = self.x[index]
        label = self.y[index]
        with open(vox_dir, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
            x = np.array(model.data, dtype=np.float32)
        x = x[np.newaxis, :]
        x = torch.from_numpy(x).float().cuda()
        y = torch.tensor(label, dtype=torch.long)
        self.tensor_cache.set(index, (y, x))
        return y, x

    def __len__(self):
        return len(self.x)
