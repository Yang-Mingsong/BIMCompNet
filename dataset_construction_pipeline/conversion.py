import math
import bpy
import csv
import subprocess
import os
import os.path
from plyfile import PlyData, PlyElement
# import cloudComPy as cc  # 需要切换conda环境至obj2ply
import mathutils
import numpy as np
from ifcopenshell import entity_instance
import ifcopenshell


# 通过相机视角设定生成相机拍摄信息
def generate_pic_info(_render, obj_name, ortho_scale):
    # 创建相机，渲染并保存图像
    for key, positions in _render.camera_positions.items():
        for i, cam_pos in enumerate(positions):
            cam_name = '{}Camera_{}'.format(key.capitalize(), i)
            direction = cam_pos.normalized()  # 指定相机朝向
            rot_quat = direction.to_track_quat('Z', 'Y')
            cam = _render.create_camera(cam_name, cam_pos,
                                        rot_quat.to_euler(),
                                        ortho_scale)
            # obj文件名\12view_Edges\0.png
            png_path = os.path.join(obj_name, key, str(i))
            yield cam, png_path


# 获取render对象，并初始化场景
def make_render(_vector):
    # 创建render对象并初始化
    render = Render(_vector)
    render.clear_scene()
    render.setup_scene()
    render.add_white_backgrand()
    render.set_light('Light', [4.07625, 1.00545, 5.90386])
    render.set_light('Light2', [-4.07625, -1.00545, -5.90386])
    render.set_light('Light3', [4.07625, -1.00545, -5.90386])
    render.set_light('Light4', [-4.07625, 1.00545, 5.90386])
    render.camera_positions = {
        'Vertices': [_vector(comb) for comb in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]],
        'Edges': [_vector((x if x != 0 else 0, y if y != 0 else 0, z if z != 0 else 0)) for x in (-1, 0, 1) for y in
                  (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z).count(0) == 1],
        'Faces': [_vector((0, 0, z)) for z in (-1, 1)] +
                 [_vector((0, y, 0)) for y in (-1, 1)] +
                 [_vector((x, 0, 0)) for x in (-1, 1)],
        'IfcNet': [_vector((math.cos(w) * math.sqrt(3), math.sin(w) * math.sqrt(3), math.sqrt(3) / 2)) for w in
                   np.linspace(0, 2 * math.pi, 12, endpoint=False)],
        'ArchShapesNet': [_vector((math.cos(w) * math.sqrt(3), math.sin(w) * math.sqrt(3), 0)) for w in
                          np.linspace(0, 2 * math.pi, 10, endpoint=False)]
    }
    return render


# 获取所有物体的边界框坐标
def get_scene_bounds(render, obj):
    Vector = render.Vector
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((-float('inf'), -float('inf'), -float('inf')))

    if obj.type == 'MESH':
        for corner in [(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]:
            min_coords = Vector(map(min, min_coords, corner))
            max_coords = Vector(map(max, max_coords, corner))

    return min_coords, max_coords


# 计算正交比例
def compute_ortho_scale(render, imported_obj, camera_distance_multiplier=1.8):
    # 获取场景边界框
    min_coords, max_coords = get_scene_bounds(render, imported_obj)
    scene_dimensions = max_coords - min_coords
    max_dimension = max(scene_dimensions)

    # 计算正交比例
    return max_dimension * camera_distance_multiplier


# 设置物体可见
def set_object_visible(imported_obj):
    imported_obj.hide_render = False


def _get_related_instances(ifc_instance, visited=None):
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
            related_instances.extend(_get_related_instances(attr_value, visited))
        elif isinstance(attr_value, tuple):  # 属性值 = 集合
            for value in attr_value:
                if isinstance(value, ifcopenshell.entity_instance) and value.id() != 0:  # 集合里是实例
                    related_instances.extend(_get_related_instances(value, visited))
                elif isinstance(value, tuple):  # 集合里是集合
                    for item in value:
                        if isinstance(item, ifcopenshell.entity_instance) and item.id() != 0:  # 集合的集合里是实例
                            related_instances.extend(_get_related_instances(item, visited))

    return related_instances


class Render:
    def __init__(self, vector):
        self.Vector = vector
        # 相机对象缓存
        self.camera_dict = {}
        for obj in bpy.context.scene.objects:
            # 检查对象是否是相机
            if obj.type == 'CAMERA':
                self.camera_dict[obj.name] = obj

    # 清理场景中的所有物体
    def clear_scene(self):
        for obj in bpy.context.scene.objects:
            print("清理物体：" + obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)

    def remove_obj(self, imported_obj):
        # 如果物体被链接到多个集合，需要先从集合中解除链接
        for collection in bpy.data.collections:
            if imported_obj.name in collection.objects:
                collection.objects.unlink(imported_obj)
        # 删除物体
        bpy.data.objects.remove(imported_obj)

    # 给模型添加白色背景
    def add_white_backgrand(self):
        # 检查是否有有效的节点树
        if bpy.context.scene.node_tree is None:
            print("当前上下文没有有效的节点树。启用节点模式")
            bpy.context.scene.use_nodes = True

        nodes = bpy.context.scene.node_tree.nodes

        # 清除所有节点（注意：在循环中直接修改列表可能会导致问题，这里只是示例）
        for node in nodes:
            nodes.remove(node)
        # 创建一个新的“渲染层”节点
        render_layers_node = nodes.new("CompositorNodeRLayers")
        # 创建一个Alpha Over节点
        alpha_over_node = nodes.new("CompositorNodeAlphaOver")
        # 创建一个新的“合成”节点
        composite_layers_node = nodes.new("CompositorNodeComposite")

        # 创建链接
        links = bpy.context.scene.node_tree.links
        links.new(render_layers_node.outputs['Image'],
                  alpha_over_node.inputs[2])
        links.new(alpha_over_node.outputs['Image'],
                  composite_layers_node.inputs['Image'])

    # 设置场景分辨率和世界颜色 output
    def setup_scene(self):
        scene = bpy.context.scene
        scene.render.resolution_x = 224
        scene.render.resolution_y = 224
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        scene.render.film_transparent = True
        # add out-line
        scene.render.use_freestyle = True
        scene.render.line_thickness_mode = 'ABSOLUTE'
        scene.render.line_thickness = 0.1
        scene.render.use_high_quality_normals = True
        scene.view_settings.view_transform = 'Standard'
        # bpy.data.linestyles["LineStyle"].color = (0.143435, 0.143435, 0.143435)
        bpy.data.linestyles["LineStyle"].color = (0, 0, 0)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
            0.0874278, 0.0874278, 0.0874278, 1)

    # 设置灯光
    def set_light(self, name, location):
        # 若指定名称的灯光为空则创建
        light = bpy.data.objects.get(name)
        if light is None:
            bpy.ops.object.light_add(type='POINT', align='WORLD',
                                     location=location, scale=(1, 1, 1))
            light = bpy.context.object
            light.name = name
            light.data.energy = 750
            light.data.shadow_soft_size = 0.01
            light.data.use_shadow = False
            light.rotation_euler = [math.radians(37.261), math.radians(3.16371), math.radians(106.936)]
        light.location = location

    # 将obj对象移动到中心
    def reset_object(self, imported_obj):
        imported_obj.hide_render = True
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        imported_obj.rotation_euler = (0, 0, 0)
        imported_obj.location = (0, 0, 0)

    def scale(self, obj):
        bbox = obj.bound_box
        width = (bbox[6][0] - bbox[0][0]) * obj.scale.x
        height = (bbox[6][1] - bbox[0][1]) * obj.scale.y
        depth = (bbox[6][2] - bbox[0][2]) * obj.scale.z
        scale_factor = 2 / max(width, height, depth)
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.context.view_layer.update()

        return scale_factor

    # 通过名称使物体可见
    def set_object_visible_by_name(self, name):
        obj = bpy.data.objects.get(name)
        obj.hide_render = False
        return obj

    # 创建相机函数
    def create_camera(self, name, location, rotation, ortho_scale):
        camera_dict = self.camera_dict
        cam_object = camera_dict.get(name)
        if cam_object is None:
            # 创建相机数据
            cam_data = bpy.data.cameras.new(name=name)
            cam_data.type = 'ORTHO'
            cam_data.ortho_scale = ortho_scale
            cam_data.clip_end = 3000

            # 创建相机对象
            cam_object = bpy.data.objects.new(name=name, object_data=cam_data)
            bpy.context.scene.collection.objects.link(cam_object)
            camera_dict[name] = cam_object
        cam_object.location = location
        cam_object.rotation_euler = rotation

        return cam_object

    # 渲染并保存图像
    def render_and_save(self, camera, output_path):
        bpy.context.scene.camera = camera
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

    # 将obj文件加载到场景中
    def load_obj_model(self, obj_path):
        # 假设这是您当前场景中的对象列表
        old_objects = list(bpy.context.scene.objects)
        bpy.ops.import_scene.obj(filepath=obj_path)
        # 获取新导入的对象
        new_objects = [obj for obj in bpy.context.scene.objects if obj not in old_objects]
        obj_name = new_objects[0].name
        return bpy.data.objects.get(obj_name), obj_name


def _trans_obj_to_numpy_array(obj_path):
    cc.initCC()
    mesh = cc.loadMesh(obj_path)
    name = mesh.getName()
    print(mesh.getName())

    # 对原始的进行位移
    mesh.setGlobalShift(-50000.00, -40000.00, 0.00)
    vertices_cloud = mesh.getAssociatedCloud()
    sample_point = mesh.samplePoints(False, 10000)
    sample_point.fuse(vertices_cloud)
    py_arr = sample_point.toNpArrayCopy()
    cc.deleteEntity(sample_point)
    cc.deleteEntity(vertices_cloud)
    cc.deleteEntity(mesh)

    point_tuples = [(x, y, z) for x, y, z in py_arr]
    point_array = np.array(point_tuples, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    return name, point_array


def convert_to_mutil_views(_obj_path, _out_path):
    render = make_render(mathutils.Vector)
    obj, obj_name = render.load_obj_model(_obj_path)
    # 重设位置并放大图像
    render.reset_object(obj)
    render.scale(obj)
    render.reset_object(obj)
    ortho_scale = compute_ortho_scale(render, obj)
    # 使模型可见
    render.set_object_visible_by_name(obj_name)
    cam_pic_iterator = generate_pic_info(render, obj_name, ortho_scale)
    for camera, pic_path in cam_pic_iterator:
        render.render_and_save(camera, os.path.join(str(_out_path), str(pic_path)))
    return obj_name


def convert_to_obj(_input_file, _output_file):
    # 从input_file中提取文件名
    ifc_name = os.path.basename(_input_file)
    # 构造转换后的obj文件的完整路径
    obj_name = ifc_name.replace('.ifc', '.obj')
    output_file = os.path.join(str(_output_file), str(obj_name))
    ifc_convert_path = r"Q:\pychem_project\BIMCompNet\untils\IfcConvert.exe"
    command = [ifc_convert_path, _input_file, output_file]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return "Conversion of {} failed with error code {}.".format(_input_file, e.returncode)


def convert_to_graph(_input_file, _output_file):
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

    ifcmodel = ifcopenshell.open(_input_file)

    # 仅构建几何数据的图
    # ifcElements = ifcmodel.by_type("IFCELEMENT")
    # if len(ifcElements) == 0:
    #     print("No IFC elements found {}".format(ifcfilepath))
    # elif len(ifcElements) == 1:
    #
    #     ifcRepresentation = ifcElements[0].Representation
    #     related_instances = _get_related_instances(ifcRepresentation)
    # else:
    #     related_instances = []
    #     for element in ifcElements:
    #         ifcRepresentation = element.Representation
    #         if ifcRepresentation is not None:
    #             related_instance = _get_related_instances(ifcRepresentation)
    #             related_instances.extend(related_instance)

    # ifcElement = ifcmodel.by_type("IFCELEMENT")[0]
    # related_instances = _get_related_instances(ifcElement)

    ifcElements = ifcmodel.by_type("IFCRELATIONSHIP")
    related_instances = []
    for ifcElement in ifcElements:
        related_instance = _get_related_instances(ifcElement)
        related_instances.extend(related_instance)

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
            if attr_value is None or (isinstance(attr_value, str) and attr_value.strip() == '') or (
                    isinstance(attr_value, str) and attr_value.strip() == '*'):
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
                                node_list.append(('value_node', value_id_counter, v))
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
    with open(os.path.join(_output_file, 'node.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(node_list)
    with open(os.path.join(_output_file, 'edge.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(edge_list)


def convert_to_pointnet(_obj_file_path):
    file_name, points = _trans_obj_to_numpy_array(_obj_file_path)
    # 创建一个新的PlyElement实例，使用原始vertex元素的数据
    vertex_element = PlyElement.describe(points, 'vertex')

    # 创建一个新的PlyData实例，包含ASCII格式的元素列表，并设置text=True
    ply_data_ascii = PlyData([vertex_element], text=True)

    # 将转换后的数据写入ASCII格式的PLY文件
    output_file = os.path.join(file_name + '.ply')
    ply_data_ascii.write(output_file)


def convert_to_binvox(_obj_file_path, resolution=256):
    obj_convert_path = r"Q:\pychem_project\BIMCompNet\utils\vox\cuda_voxelizer.exe"
    command = [obj_convert_path, '-f', _obj_file_path, '-s', str(resolution), '-o', 'binvox']
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return "Conversion of {} failed with error code {}.".format(_obj_file_path, e.returncode)


if __name__ == '__main__':
    ifc_file_path = r"Q:\pychem_project\BIMCompNet\case\correction\IfcStair\IfcStair_0416184708348846.ifc"
    obj_file_path = r"Q:\pychem_project\BIMCompNet\case\conversion\IfcStair_0416184708348846.obj"
    out_path = r"Q:\pychem_project\BIMCompNet\case\conversion"
    # convert_to_obj(ifc_file_path, out_path)
    # convert_to_graph(ifc_file_path, out_path)
    # convert_to_mutil_views(obj_file_path, out_path)
    # convert_to_pointnet(obj_file_path)
    convert_to_binvox(obj_file_path)
