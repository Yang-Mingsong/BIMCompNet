import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


# reverse w/o self-loop
def custom_aggregate_3(inputs, dsttype):
    if dsttype == 'class_node':
        part1 = inputs[0]
        part2 = torch.zeros_like(inputs[0])
        part3 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'attribute_node' or dsttype == 'type_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.max(torch.stack([inputs[0], inputs[1], inputs[2]]), dim=0)[0]
        part3 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    else:
        print("______大错特错")
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        part3 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    return aggregated_feats


# forward w/o self-loop
def custom_aggregate_2(inputs, dsttype):
    if dsttype == 'class_node' or dsttype == 'value_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.max(torch.stack([inputs[0], inputs[1]]), dim=0)[0]
        aggregated_feats = torch.cat((part1, part2), dim=1)

    elif dsttype == 'attribute_node':
        part1 = inputs[0]
        part2 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2), dim=1)

    elif dsttype == 'type_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = inputs[0]
        aggregated_feats = torch.cat((part1, part2), dim=1)
    else:
        print("______大错特错")
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2), dim=1)

    return aggregated_feats


# forward with self-loop
def custom_aggregate_1(inputs, dsttype):
    if dsttype == 'class_node' or dsttype == 'value_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.max(torch.stack([inputs[0], inputs[1]]), dim=0)[0]
        part3 = inputs[2]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'attribute_node':
        part1 = inputs[0]
        part2 = torch.zeros_like(inputs[0])
        part3 = inputs[1]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'type_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = inputs[0]
        part3 = inputs[1]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)
    else:
        print("______大错特错")
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        part3 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    return aggregated_feats


# reverse with self-loop
def custom_aggregate(inputs, dsttype):
    if dsttype == 'class_node':
        part1 = inputs[0]
        part2 = torch.zeros_like(inputs[0])
        part3 = inputs[1]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'attribute_node' or dsttype == 'type_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.max(torch.stack([inputs[0], inputs[1], inputs[2]]), dim=0)[0]
        part3 = inputs[3]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'value_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        part3 = inputs[0]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)
    else:
        print("______大错特错")
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        part3 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    return aggregated_feats


class RGCNLayer(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super(RGCNLayer, self).__init__()
        # forward用custom_aggregate_1,custom_aggregate_2  2不带自指
        # reverse用custom_aggregate,custom_aggregate_3  3不带自指
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate=custom_aggregate)
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(3 * hid_feats, out_feats)  # w/o self-loop reverse 和 with self-loop
            # rel: dglnn.GraphConv(2 * hid_feats, out_feats)  # w/o self-loop forward
            for rel in rel_names}, aggregate=custom_aggregate)

        # # reserve w/o self-loop：value_node
        # self.linear_value_node = nn.Linear(in_feats, hid_feats)  # 添加线性层升维，将 value_node 特征升维

    def forward(self, graph, inputs):
        # inputs是节点的特征
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}

        # # reserve w/o self-loop：value_node
        # part3 = F.relu(self.linear_value_node(inputs['value_node']))
        # part1 = torch.zeros_like(part3)
        # part2 = torch.zeros_like(part3)
        # vn_feats = torch.cat((part1, part2, part3), dim=1)
        # h['value_node'] = vn_feats

        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}

        # # reserve w/o self-loop：value_node
        # h['value_node'] = vn_feats

        return h


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super(RGCN, self).__init__()
        self.rgcn = RGCNLayer(in_dim, hidden_dim, hidden_dim, rel_names)

        self.fc1 = nn.Linear(12*hidden_dim, 3*hidden_dim)
        self.bn = nn.BatchNorm1d(3*hidden_dim)
        self.fc2 = nn.Linear(3*hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

        # # w/o self-loop FORWARD
        # self.fc1 = nn.Linear(8 * hidden_dim, 2 * hidden_dim)
        # self.bn = nn.BatchNorm1d(2 * hidden_dim)
        # self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.classify = nn.Linear(hidden_dim, n_classes)

    # # 初始化特征存储
        # self.features = None
        # self.hook = self.fc1.register_forward_hook(self.get_features)

    # def get_features(self, module, input, output):
    #     self.features = output.detach()
    #
    # def close(self):
    #     self.hook.remove()

    def forward(self, g):
        h = g.ndata['features']
        h = self.rgcn(g, h)

        with g.local_scope():
            g.ndata['features'] = h
            features_list = []
            for ntype in g.ntypes:
                features_list.append(dgl.max_nodes(g, 'features', ntype=ntype))

            hg = torch.cat(features_list, dim=1)
            hg = F.relu(self.bn(self.fc1(hg)))
            hg = F.relu(self.bn2(self.fc2(hg)))
        return self.classify(hg)
