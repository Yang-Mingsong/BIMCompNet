import os
import shutil

import ifcopenshell
from ifcopenshell.util.element import remove_deep2


def _find_zero_point(_ifc_model):
    # 该函数会搜索IFC模型中的所有IfcCartesianPoint实体，找到坐标为(0.0, 0.0, 0.0)的点，通常代表原点
    _entity_items = _ifc_model.by_type("IfcCartesianPoint")  # 获取IFC模型中所有类型为IfcCartesianPoint的实体
    _zero_point = None
    for _entity in _entity_items:
        if _entity.Coordinates == (0.0, 0.0, 0.0):  # 遍历这些实体，检查每个点的坐标是否为(0.0, 0.0, 0.0)
            _zero_point = _entity  # 若找到原点，将赋值给_zero_point
            break
    return _zero_point


def _location_setup(_entity, _ifc_model):
    _object_placement = _entity.ObjectPlacement
    _object_placement.PlacementRelTo = None
    _placement = _object_placement.RelativePlacement
    _location = _placement.Location
    _z_direction = _placement.Axis
    _x_direction = _placement.RefDirection

    # 这两步不冲突！第一步确保模型中的元素相对于彼此正确对齐或定位，
    # 第二步则是确保所有元素相对于全局坐标系的原点对齐
    _placement.Location = _find_zero_point(_ifc_model)
    _location.Coordinates = [0.0, 0.0, 0.0]
    _z_direction = None  #方向的清除，重置z轴方向为无方向
    _x_direction = None  # 重置x轴方向为无方向


def location_setup(_ifc_model):
    four_types = ("IfcCurtainWall", "IfcStair", "IfcRamp", "IfcRoof")
    _entities = _ifc_model.by_type("IfcElement")
    if len(_entities) == 1:
        _location_setup(_entities[0], _ifc_model)  # 则对这个元素调用 __location_setup__ 函数
    else:
        has_four_types = any(e.is_a(t) for e in _entities for t in four_types)
        if has_four_types:
            for _entity in _entities:  # 幕墙、楼梯、坡道、屋顶
                if _entity.is_a("IfcCurtainWall") or _entity.is_a("IfcStair") or _entity.is_a("IfcRamp") or _entity.is_a("IfcRoof"):
                    _location_setup(_entity, _ifc_model)
        else:
            for _entity in _entities:
                if _entity.is_a("IfcFeatureElement") or _entity.is_a("IfcVirtualElement") or _entity.is_a("IfcElementAssembly"):
                    continue
                else:
                    _location_setup(_entity, _ifc_model)


def remove_geos(_ifc_model):  # 去除包围盒以及其他形式的几何表达
    _entities = _ifc_model.by_type("IfcElement")
    for _entity in _entities:
        product_representation = _entity.Representation
        if product_representation is not None:
            representations = list(product_representation.Representations)
            for k, _representation in enumerate(representations):
                if _representation.RepresentationIdentifier == "Body":
                    continue
                else:
                    representations.remove(_representation)
                    remove_deep2(_ifc_model, _representation)
                    _ifc_model.remove(_representation)
            if len(representations) > 1:
                for k, _representation in enumerate(representations):
                    if _representation.RepresentationIdentifier == "Body":
                        continue
                    else:
                        representations.remove(_representation)
                        remove_deep2(_ifc_model, _representation)
                        _ifc_model.remove(_representation)
            product_representation.Representations = representations


def label_setup(_path):
    for file_name in os.listdir(_path):
        file_path = os.path.join(_path, file_name)
        # 判断是文件并且以.ifc结尾
        if os.path.isfile(file_path) and file_name.endswith(".ifc"):
            # 通过"_"分割，取第一个部分作为前缀
            prefix = file_name.split("_")[0]
            # 构建目标文件夹路径，例如 aa/jj
            target_folder = os.path.join(_path, prefix)
            # 如果目标文件夹不存在，则新建文件夹
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            # 构建目标文件路径
            target_file_path = os.path.join(target_folder, file_name)
            # 移动文件到目标文件夹中
            shutil.move(file_path, target_file_path)


if __name__ == '__main__':
    IFC_FOLDER_PATH = r"Q:\pychem_project\BIMCompNet\case\extraction"
    NEW_IFC_FOLDER_PATH = r"Q:\pychem_project\BIMCompNet\case\correction"
    for root, dirs, files in os.walk(IFC_FOLDER_PATH):
        for file in files:
            ifc_file_path = os.path.join(root, file)
            ifc_model = ifcopenshell.open(ifc_file_path)
            location_setup(ifc_model)
            remove_geos(ifc_model)
            new_ifc_file_path = os.path.join(NEW_IFC_FOLDER_PATH, file)
            ifc_model.write(new_ifc_file_path)
    label_setup(NEW_IFC_FOLDER_PATH)
