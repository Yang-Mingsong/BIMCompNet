import csv
import ifcopenshell
import pandas as pd
import os
import concurrent.futures
import logging
from ifcopenshell import entity_instance
import psutil


def get_ifc_schem_infor(IFC_SCHEMA_PATH):
    # 读取ifcSchema，解析成ifc_schema_dict{item：n_type}

    attribute_nodes_set = set()
    class_nodes_set = set()
    value_nodes_set = set()
    ifc_schema_dict = {}
    ifc_schema_graph = pd.read_csv(IFC_SCHEMA_PATH, names=['src', 'edge_type', 'dst'])

    # 根据规则分类
    for _, row in ifc_schema_graph.iterrows():
        if row['edge_type'] == 'hasAttribute':
            attribute_nodes_set.add(row['dst'])
            class_nodes_set.add(row['src'])
        elif row['edge_type'] == 'hasSubClass':
            class_nodes_set.add(row['src'])
            class_nodes_set.add(row['dst'])
        elif row['edge_type'] == 'hasValue':
            value_nodes_set.add(row['dst'])

    # 创建 TypeNode 集合
    all_nodes = set(ifc_schema_graph['src']).union(set(ifc_schema_graph['dst']))
    type_nodes_set = all_nodes - attribute_nodes_set - class_nodes_set - value_nodes_set

    ifc_schema_dict['CN'] = class_nodes_set
    ifc_schema_dict['AN'] = attribute_nodes_set
    ifc_schema_dict['TN'] = type_nodes_set
    ifc_schema_dict['VN'] = value_nodes_set

    return ifc_schema_dict


def get_related_instances(ifc_instance, visited=None):
    if visited is None:
        visited = set()

    instance_id = ifc_instance.id()
    if instance_id in visited:
        return []

    visited.add(instance_id)
    related_instances = [ifc_instance]

    attrs = ifc_instance.get_info()
    for attr_name in list(attrs.keys()):
        attr_value = attrs[attr_name]
        if attr_value is None or attr_name in ["GlobalId", "OwnerHistory", "Name", "id", "type"]:
            continue
        if isinstance(attr_value, ifcopenshell.entity_instance) and attr_value.id() != 0:  # 属性值 = ifc实例
            related_instances.extend(get_related_instances(attr_value, visited))
        elif isinstance(attr_value, tuple):  # 属性值 = 集合
            for value in attr_value:
                if isinstance(value, ifcopenshell.entity_instance) and value.id() != 0:  # 集合里是实例
                    related_instances.extend(get_related_instances(value, visited))
                elif isinstance(value, tuple):  # 集合里是集合
                    for item in value:
                        if isinstance(item, ifcopenshell.entity_instance) and item.id() != 0:  # 集合的集合里是实例
                            related_instances.extend(get_related_instances(item, visited))

    return related_instances


def get_out_pth(ifc_file_path):
    new_ifc_file_infor_path = os.path.dirname(ifc_file_path.replace('\IFC', '\GRAPH'))
    return new_ifc_file_infor_path


def process_ifc_model(ifcfilepath):
    # 初始化列表字典和计数器
    node_list = []
    edge_list = []
    # 用于记录class节点原始id和新id的关系
    class_node_id_dict = {}

    # 用于生成连续 ID 的计数器
    class_id_counter = 0
    attr_id_counter = 0
    type_id_counter = 0
    value_id_counter = 0

    ifcmodel = ifcopenshell.open(ifcfilepath)

    # 仅构建几何数据的图
    ifcElements = ifcmodel.by_type("IFCELEMENT")
    if len(ifcElements) == 0:
        print("No IFC elements found {}".format(ifcfilepath))
    elif len(ifcElements) == 1:

        ifcRepresentation = ifcElements[0].Representation
        related_instances = get_related_instances(ifcRepresentation)
    else:
        related_instances = []
        for element in ifcElements:
            ifcRepresentation = element.Representation
            if ifcRepresentation is not None:
                related_instance = get_related_instances(ifcRepresentation)
                related_instances.extend(related_instance)

    # ifcElement = ifcmodel.by_type("IFCELEMENT")[0]
    # related_instances = get_related_instances(ifcElement)

    # ifcElements = ifcmodel.by_type("IFCRELATIONSHIP")
    # related_instances = []
    # for ifcElement in ifcElements:
    #     related_instance = get_related_instances(ifcElement)
    #     related_instances.extend(related_instance)

    # 迭代 IFC 文件中的所有元素
    for element in related_instances:  # IfcRoot 是所有 IFC 类的父类
        class_name = element.is_a()  # 获取类名

        # 分配ID给类名
        node_list.append(('class_node', class_id_counter, class_name))
        class_node_id_dict[element.id()] = class_id_counter
        class_id_counter += 1

    for element in related_instances:
        # 获取实体的属性和值
        attributes = element.get_info()
        class_id = class_node_id_dict[element.id()]

        for attr, attr_value in attributes.items():
            # 跳过 IFC 内部属性
            if attr in ('type', 'id', 'OwnerHistory'):
                continue
            # 跳过空属性值
            if attr_value is None or (isinstance(attr_value, str) and attr_value.strip() == '') or (isinstance(attr_value, str) and attr_value.strip() == '*'):
                continue
            if isinstance(attr_value, entity_instance) and attr_value.id() == 0:
                if attr_value.wrappedValue == '':
                    continue

            # 每次都分配新的 ID 给属性名
            node_list.append(('attribute_node', attr_id_counter, attr))
            attr_id = attr_id_counter
            attr_id_counter += 1
            # 五元组
            edge_list.append(('class_node', class_id, 'attribute_node', attr_id, 'hasAttribute'))

            # 处理属性值并分配新 ID
            if isinstance(attr_value, (str, int, float, bool)):
                node_list.append(('value_node', value_id_counter, attr_value))
                value_id = value_id_counter
                value_id_counter += 1
                # 五元组
                edge_list.append(('attribute_node', attr_id, 'value_node', value_id, 'hasValue'))

            elif isinstance(attr_value, entity_instance):
                if attr_value.id() == 0:
                    node_list.append(('type_node', type_id_counter, attr_value.is_a()))
                    w_type_id = type_id_counter
                    type_id_counter += 1
                    # 五元组
                    edge_list.append(('attribute_node', attr_id, 'type_node', w_type_id, 'hasValue'))

                    node_list.append(('value_node', value_id_counter, attr_value.wrappedValue))
                    value_id = value_id_counter
                    value_id_counter += 1
                    # 五元组
                    edge_list.append(('type_node', w_type_id, 'value_node', value_id, 'hasValue'))
                else:
                    value_id = class_node_id_dict[attr_value.id()]
                    # 五元组
                    edge_list.append(('attribute_node', attr_id, 'class_node', value_id, 'hasValue'))

            elif isinstance(attr_value, tuple):
                # typenodes
                node_list.append(('type_node', type_id_counter, 'TUPLE'))
                type_id = type_id_counter
                type_id_counter += 1
                # 五元组
                edge_list.append(('attribute_node', attr_id, 'type_node', type_id, 'hasValue'))

                # 展开元组类型的属性值,也应分为三种进行判断，同时添加typenodes
                for i, value in enumerate(attr_value):
                    if isinstance(value, (str, int, float, bool)):
                        node_list.append(('value_node', value_id_counter, value))
                        value_id = value_id_counter
                        value_id_counter += 1
                        # 五元组
                        edge_list.append(('type_node', type_id, 'value_node', value_id, 'hasValue'))
                    elif isinstance(value, entity_instance):
                        if value.id() == 0:
                            node_list.append(('type_node', type_id_counter, value.is_a()))
                            w_type_id = type_id_counter
                            type_id_counter += 1
                            # 五元组
                            edge_list.append(('type_node', type_id, 'type_node', w_type_id, 'hasValue'))

                            node_list.append(('value_node', value_id_counter, value.wrappedValue))
                            value_id = value_id_counter
                            value_id_counter += 1
                            # 五元组
                            edge_list.append(('type_node', w_type_id, 'value_node', value_id, 'hasValue'))
                        else:
                            value_id = class_node_id_dict[value.id()]
                            # 五元组
                            edge_list.append(('type_node', type_id, 'class_node', value_id, 'hasValue'))
                    elif isinstance(value, tuple):
                        # typenodes
                        node_list.append(('type_node', type_id_counter, 'TUPLETUPLE'))
                        s_type_id = type_id_counter
                        type_id_counter += 1
                        # 五元组
                        edge_list.append(('type_node', type_id, 'type_node', s_type_id, 'hasValue'))

                        for _, v in enumerate(value):
                            if isinstance(v, (str, int, float, bool)):
                                node_list.append(( 'value_node', value_id_counter, v))
                                value_id = value_id_counter
                                value_id_counter += 1
                                # 五元组
                                edge_list.append(('type_node', s_type_id, 'value_node', value_id, 'hasValue'))
                            elif isinstance(v, ifcopenshell.entity_instance):
                                if v.id() == 0:
                                    node_list.append(('type_node', type_id_counter, v.is_a()))
                                    sw_type_id = type_id_counter
                                    type_id_counter += 1
                                    # 五元组
                                    edge_list.append(('type_node', s_type_id, 'type_node', sw_type_id, 'hasValue'))

                                    node_list.append(('value_node', value_id_counter, v.wrappedValue))
                                    value_id = value_id_counter
                                    value_id_counter += 1
                                    # 五元组
                                    edge_list.append(('type_node', sw_type_id, 'value_node', value_id, 'hasValue'))
                                else:
                                    value_id = class_node_id_dict[v.id()]
                                    # 五元组
                                    edge_list.append(('type_node', s_type_id, 'class_node', value_id, 'hasValue'))
                            else:
                                print("{}的属性值{}的{}是{},类型是tuple{}".format(attr, attr_value, value, v, type(v)))
            else:
                print("{}的属性值{}的类型是tuple{}".format(attr, attr_value, type(attr_value)))

    # 添加自指关系
    for node in node_list:
        node_type, node_id, node_item = node
        edge_list.append((node_type, node_id, node_type, node_id, 'selfLoop'))

    # 输出节点和边的csv文件
    out_path = get_out_pth(ifcfilepath)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, 'geo_node.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(node_list)
    with open(os.path.join(out_path, 'geo_edge.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(edge_list)


def classify_files_by_dynamic_size(dataset_folder_path):
    files = []

    # 遍历文件夹中的文件，收集文件路径和大小
    for root, _, file_names in os.walk(dataset_folder_path):
        for file_name in file_names:
            if file_name.endswith(".ifc"):
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # 将文件大小转换为KB
                    files.append((file_path, file_size))

    # if not files:
    #     return [], [], [], [], [], [], [], [], [], [], []  # 如果没有文件，返回空列表

    # 获取最大和最小文件大小
    min_size = min(files, key=lambda x: x[1])[1]
    print("最小值{}".format(min_size))
    max_size = max(files, key=lambda x: x[1])[1]
    print("最大值{}".format(max_size))

    # 初始化10个列表
    list0, list1, list2, list3, list4, list5, list6, list7, list8, list9, list10 \
        = [], [], [], [], [], [], [], [], [], [], []

    # 将文件路径按照大小分类到不同列表中
    for file_path, file_size in files:

        if file_size < 20:
            list0.append(file_path)
        elif 20 <= file_size < 30:
            list1.append(file_path)
        elif 30 <= file_size < 80:
            list2.append(file_path)
        elif 80 <= file_size < 200:
            list3.append(file_path)
        elif 200 <= file_size < 357:
            list4.append(file_path)
        elif 357 <= file_size < 432:
            list5.append(file_path)
        elif 432 <= file_size < 545:
            list6.append(file_path)
        elif 545 <= file_size < 725:
            list7.append(file_path)
        elif 725 <= file_size < 1070:
            list8.append(file_path)
        elif 1070 <= file_size < 1400:
            list9.append(file_path)
        else:
            list10.append(file_path)

    print("十一个列表的长度分别为list0:{},list1:{},list2:{},list3:{},list4:{},list5:{},list6:{},list7:{},list8:{},list9:{},list10:{}".format
          (len(list0), len(list1), len(list2), len(list3), len(list4), len(list5), len(list6), len(list7), len(list8), len(list9), len(list10)))

    ifc_model_files_list = [list0, list1, list2, list3, list4, list5, list6, list7, list8, list9, list10]
    return ifc_model_files_list


def check_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f'Memory usage: {mem_info.rss / (1024 * 1024)} MB')


if __name__ == '__main__':
    DATASET_PATH = 'H:\BIMCompNet'
    for bim_class in os.listdir(DATASET_PATH):
        RAW_PATH = os.path.join(DATASET_PATH, bim_class)
        ifc_model_files_list = classify_files_by_dynamic_size(RAW_PATH)
        for i, ifc_model_files in enumerate(ifc_model_files_list):
            num_processes = 8
            while True:
                results_temp = {}  # 存储临时结果
                failed_files = []  # 记录失败的文件
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                    future_to_file = {executor.submit(process_ifc_model, file): file for file in ifc_model_files}
                    for future in concurrent.futures.as_completed(future_to_file):
                        file = future_to_file[future]
                        try:
                            data = future.result()
                            logging.info(f'{file} processed successfully')
                        except Exception as exc:
                            logging.error(f'{file} generated an exception: {exc}')
                            failed_files.append(file)

                # 如果没有失败的文件，退出循环
                if not failed_files:
                    print("{}list{}已完成，处理了{}个文件".format(RAW_PATH, i, len(ifc_model_files)))
                    break
