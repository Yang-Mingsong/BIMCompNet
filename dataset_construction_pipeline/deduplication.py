import hashlib
import os
import shutil
import torch
import numpy as np
import subprocess


# 读取ifc文件，现有解析器以list的形式加载，在文件较大的情况下列表长度不好获取
def load_obj_file(_file_path):
    with open(_file_path, encoding='utf-8') as file:  # 打开文件
        points = {}
        vertices = []
        index = 0
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":  # obj文件中以v开头的是点的坐标，提取顶点坐标
                index += 1
                vertices.append((float(strs[1]), float(strs[2]), float(strs[3])))
                points.update({index: [float(strs[1]), float(strs[2]), float(strs[3])]}) #将坐标存储在字典points中
            elif strs[0] == "vt":
                break
    vertices = np.array(vertices)
    return points, vertices #返回顶点列表和字典


# 遍历文件后按照顶点数量进行分组
def classify_obj_file_by_vertex_num(_obj_folder):
    # 创建一个字典，用于存储顶点数量相同的文件名
    vertex_dict = {}  # {顶点个数1：[obj文件路径1,obj文件路径2]}
    objs_dict = {}
    obj_files = [file for file in os.listdir(_obj_folder) if file.endswith('.obj')]

    # 遍历文件并将它们分组
    for file in obj_files:
        file_path = os.path.join(_obj_folder, file)
        with open(file_path, 'r'):
            points, vertices = load_obj_file(file_path)
            objs_dict[file_path] = points
            vertices_count = len(points)
        if vertices_count == 0:
            print("{}这个文件的顶点数为0".format(file_path))
        if vertices_count not in vertex_dict:
            vertex_dict[vertices_count] = []
        vertex_dict[vertices_count].append(file_path)
    return vertex_dict, objs_dict


# 计算obj模型中点之间的距离特征，用于判断是否是重复
def compute_distance(points, _vertices_count):
    points_array = np.array(list(points.values()))
    points_tensor = torch.tensor(points_array, device='cuda')
    if _vertices_count > 15000:
        # 计算每个点与几何中心点的距离
        center = np.mean(points_array, axis=0)
        ca_tensor = torch.tensor(center, device='cuda')
        diffs_tensor = points_tensor - ca_tensor
        length_tensor = torch.linalg.norm(diffs_tensor, dim=1)
        cen_dis_mean_diff = torch.abs(torch.mean(length_tensor))
        return cen_dis_mean_diff.item(), 0
    else:
        # 计算每个点与几何中心点的距离
        center = np.mean(points_array, axis=0)
        ca_tensor = torch.tensor(center, device='cuda')
        diffs_tensor = points_tensor - ca_tensor
        length_tensor = torch.linalg.norm(diffs_tensor, dim=1)
        cen_dis_mean_diff = torch.abs(torch.mean(length_tensor))

        # 计算每个点与其他点之间的距离
        lengths = []
        for i in range(len(points_tensor)):
            point = points_tensor[i]
            diffs = points_tensor - point
            length = torch.linalg.norm(diffs, dim=1)
            length = length[length != 0]
            lengths.append(length)
        # 计算距离平均值的绝对值
        lengths_tensor = torch.cat(lengths)
        point_dis_mean_diff = torch.abs(torch.mean(lengths_tensor))

        return cen_dis_mean_diff.item(), point_dis_mean_diff.item()


# 根据两种距离计算MD5
def calculate_distance_features(cen_distance, point_distance): #根据中心距离和点对距离计算MD5哈希值
        mean_diff_1 = cen_distance + point_distance
        combined_value = "{}{}".format(cen_distance, point_distance)
        mean_diff_2 = hashlib.md5(combined_value.encode()).hexdigest()
        combined_value_2 = "{}{}".format(mean_diff_1, mean_diff_2)
        distance_feature_value = hashlib.md5(combined_value_2.encode()).hexdigest()
        return distance_feature_value


# 筛选出给定文件夹中不重复的obj文件路径
def execute_program(_obj_folder):
    print("开始处理{}文件夹".format(_obj_folder))
    for _file in os.listdir(_obj_folder):
        if _file.endswith('.mtl'):
            os.remove(os.path.join(_obj_folder, _file))
    # 存储哈希值和路径
    nonredundant_obj_files_dict = {}
    # 按顶点数量对obj文件分类
    vertex_dict, objs_dict = classify_obj_file_by_vertex_num(_obj_folder)
    # 从顶点数相同的obj文件中抽取出不重复的obj文件，并记录它们的路径
    for vertices_count, file_paths in vertex_dict.items():
        print("顶点数为 {} 的文件有:{}个".format(vertices_count, len(file_paths)))
        if vertices_count == 0:
            continue
        else:
            for file_path in file_paths:
                cen_dis_mean_diff, point_dis_mean_diff = compute_distance(objs_dict[file_path], vertices_count)
                dis_feature_value = calculate_distance_features(cen_dis_mean_diff, point_dis_mean_diff)
                nonredundant_obj_files_dict[dis_feature_value] = file_path

    # 其实不是删除，就是记个数而已
    print("----------{}文件夹处理结束,删除重复模型{}个".format(_obj_folder, len(objs_dict)-len(nonredundant_obj_files_dict)))
    return nonredundant_obj_files_dict


# 将某文件复制到指定文件夹，如果文件夹不存在就新建
def copy_file(src_file, dst_dir):
    dst_folder_path = os.path.dirname(dst_dir)
    os.makedirs(dst_folder_path, exist_ok=True)
    shutil.copy(src_file, dst_folder_path)


# 删除obj文件中的g和s行。现在已经重写了加载obj文件的方法，没啥用
def remove_g_and_s_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 移除包含 'g' 和 's' 的行
    lines = [line for line in lines if not line.startswith(('g', 's'))]

    # 重新写入文件
    with open(file_path, 'w') as f:
        f.writelines(lines)


# 批量移除dict中obj文件们的g和s行 #g指组，s指平滑着色组
def batch_remove_g_and_s(directory):
    """Batch remove 'g' and 's' lines from all OBJ files in a directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                remove_g_and_s_lines(file_path)
                print(f"Removed 'g' and 's' lines from {file_path}")


def convert(_ifc_path, _obj_path):
    ifc_convert_path = r"Q:\pychem_project\BIMCompNet\untils\IfcConvert.exe"
    command = [ifc_convert_path, _ifc_path, _obj_path]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return "Conversion of {} failed with error code {}.".format(_ifc_path, e.returncode)


if __name__ == '__main__':
    ifc_folder = r"Q:\pychem_project\BIMCompNet\case\correction"  # 根目录路径
    temp_folder = r"Q:\pychem_project\BIMCompNet\case\deduplication_temp"  # 临时OBJ文件目录
    deduplication_folder = r"Q:\pychem_project\BIMCompNet\case\deduplication"  # 去重后的文件目录
    # # batch_remove_g_and_s(root_folder)  # 删除g和s的
    # for root, dirs, files in os.walk(ifc_folder):
    #     for file in files:
    #         if file.endswith('.ifc'):
    #             ifc_path = os.path.join(root, file)
    #             temp_obj_folder = os.path.join(temp_folder, file.split("_")[0])
    #             if not os.path.exists(temp_obj_folder):
    #                 os.mkdir(temp_obj_folder)
    #             obj_path = os.path.join(temp_obj_folder, file.replace('.ifc', '.obj'))
    #             convert(ifc_path, obj_path)

    # # 对根目录下文件进行去重
    for folder in os.listdir(temp_folder):
        obj_folder = os.path.join(temp_folder, folder)
        results = execute_program(obj_folder)
        for _, obj_file_path in results.items():
            old_ifc_path = obj_file_path.replace('deduplication_temp', 'correction').replace('.obj', '.ifc')
            new_ifc_path = old_ifc_path.replace('correction', 'deduplication')
            copy_file(old_ifc_path, new_ifc_path)
