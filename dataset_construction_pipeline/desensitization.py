import os
import ifcopenshell


def header_desensitization(_ifc_model):
    # 头部信息的脱敏，包括文件描述、作者、组织、预处理器版本等，将它们全都设置为默认值或匿名值
    _header = _ifc_model.header
    _header.file_description.description = ["GeoIFCNet file"]
    _header.file_description.implementation_level = ""
    _header.file_name.name = ""
    _header.file_name.time_stamp = "1995-11-18T09:00:00"
    _header.file_name.author = ["YaMiSo"]
    _header.file_name.organization = ["XUT_BIM606"]
    _header.file_name.preprocessor_version = "ifcopenshell 0.7.0"
    _header.file_name.originating_system = ""
    _header.file_name.authorization = "None"
    _header.file_schema.schemas = ["IFC4"]


def _find_zero_point(_ifc_model):
    # 该函数会搜索IFC模型中的所有IfcCartesianPoint实体，找到坐标为(0.0, 0.0, 0.0)的点，通常代表原点
    _entity_items = _ifc_model.by_type("IfcCartesianPoint")  # 获取IFC模型中所有类型为IfcCartesianPoint的实体
    _zero_point = None
    for _entity in _entity_items:
        if _entity.Coordinates == (0.0, 0.0, 0.0):  # 遍历这些实体，检查每个点的坐标是否为(0.0, 0.0, 0.0)
            _zero_point = _entity  # 若找到原点，将赋值给_zero_point
            break
    return _zero_point


def model_desensitization(_ifc_model):
    # 处理IFC模型中的各种实体，如角色、地址、组织、人员等，移除或修改敏感信息
    _entity_items = _ifc_model.by_type("IfcActorRole")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Role = "USERDEFINED"
            _entity.UserDefinedRole = "Researcher"
            _entity.Description = None

    _entity_items = _ifc_model.by_type("IfcAddress")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Purpose = None
            _entity.Description = None
            _entity.UserDefinedPurpose = None

    _entity_items = _ifc_model.by_type("IfcOrganization")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Description = None
            _entity.Name = "XUT_BIM606"
            _entity.Identification = None

    _entity_items = _ifc_model.by_type("IfcPerson")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Identification = None
            _entity.FamilyName = None
            _entity.GivenName = None
            _entity.MiddleNames = None
            _entity.PrefixTitles = None
            _entity.SuffixTitles = None

    _entity_items = _ifc_model.by_type("IfcPostalAddress")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Purpose = None
            _entity.Description = None
            _entity.UserDefinedPurpose = None
            _entity.InternalLocation = None
            _entity.AddressLines = None
            _entity.PostalBox = None
            _entity.Town = None
            _entity.Region = None
            _entity.PostalCode = None
            _entity.Country = None

    _entity_items = _ifc_model.by_type("IfcTelecomAddress")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Purpose = None
            _entity.Description = None
            _entity.UserDefinedPurpose = None
            _entity.TelephoneNumbers = None
            _entity.FacsimileNumbers = None
            _entity.PagerNumber = None
            _entity.ElectronicMailAddresses = None
            _entity.WWWHomePageURL = None
            _entity.MessagingIDs = None

    _entity_items = _ifc_model.by_type("IfcApplication")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.Version = "0.7.0"
            _entity.ApplicationFullName = "ifcopenshell 0.7.0"
            _entity.ApplicationIdentifier = "ifcopenshell"

    _entity_items = _ifc_model.by_type("IfcOwnerHistory")
    if len(_entity_items) > 0:
        for _entity in _entity_items:
            _entity.LastModifiedDate = None
            _entity.CreationDate = 818056800


def spatial_element_desensitization(_ifc_model):
    # 修改几何表达上下文中的位置信息，将所有位置信息重置为原点或匿名值
    _entity_items = _ifc_model.by_type("IfcGeometricRepresentationContext")  # 获取IFC模型中所有几何表达上下文实体
    for _entity_item in _entity_items:
        if _entity_item.is_a() == "IfcGeometricRepresentationContext":  # 确认实体类型
            true_north = _entity_item.TrueNorth  # 获取TrueNorth属性
            _entity_item.TrueNorth = None  # 将TrueNorth设置为None，移除方向信息
            true_north.DirectionRatios = [0.0, 1.0]  # 重置方向比率为(0.0, 1.0)，这通常表示一个默认方向
            world_coordinate_system = _entity_item.WorldCoordinateSystem  # 获取世界坐标系统
            if world_coordinate_system.is_a("IfcAxis2Placement3D"):  # 确认坐标系统类型
                _entity_item.WorldCoordinateSystem.Axis = None  # 将坐标系统的轴设置为None，移除轴信息
                _entity_item.WorldCoordinateSystem.RefDirection = None  # 将参考方向设置为None，移除方向信息
                if world_coordinate_system.Location.Coordinates != (0.0, 0.0, 0.0):  # 检查位置坐标是否为原点
                    _entity_item.WorldCoordinateSystem.Location = _find_zero_point(_ifc_model)  # 若不是原点，则将位置设置为找到的原点

    # 移除项目以及空间结构信息
    _entities = _ifc_model.by_type("IfcProject")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.LongName = None
        _entity.Phase = None
        _entity.ObjectType = None
    _entities = _ifc_model.by_type("IfcSite")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.ObjectType = None
        _entity.LandTitleNumber = None
        _entity.LongName = None
        _entity.RefLatitude = None
        _entity.RefLongitude = None
    _entities = _ifc_model.by_type("IfcBuilding")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.ObjectType = None
        _entity.LongName = None
    _entities = _ifc_model.by_type("IfcBuildingStorey")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.ObjectType = None
        _entity.LongName = None
    _entities = _ifc_model.by_type("IfcSpace")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.ObjectType = None
        _entity.LongName = None


def entity_desensitization(_ifc_model):
    # 进一步处理IFC模型中的实体，如元素、属性、配置文件等，移除或修改敏感信息。
    # 移除对象自身的一些属性
    _entities = _ifc_model.by_type("IfcElement")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.ObjectType = None
        _entity.Tag = None

    # 移除对象自身的一些属性
    _entities = _ifc_model.by_type("IfcElementType")
    for _entity in _entities:
        _entity.Name = None
        _entity.Description = None
        _entity.Tag = None

    # 移除对象IFC的自定义属性集中的属性信息
    _property_sets = _ifc_model.by_type("IfcPropertySet")
    for _property_set in _property_sets:
        if "Pset_" not in _property_set.Name:
            properties = _property_set.HasProperties
            for _property in properties:
                if _property.is_a("IfcPropertySingleValue"):
                    _property.Name = ""
                    _property.Description = None
                    _property.NominalValue = None

    # 移除对象IFC预定义属性集中的指定属性值
    _properties = _ifc_model.by_type("IfcPropertySingleValue")
    for _property in _properties:
        if _property.Name == "Reference":
            _property.NominalValue = None

    # 移除对象IfcProfileDef中的name属性
    _entities = _ifc_model.by_type("IfcProfileDef")
    for _entity in _entities:
        _entity.ProfileName = None


if __name__ == '__main__':
    ifc_file_path = r"Q:\pychem_project\BIMCompNet\case\test.ifc"
    ifc_model = ifcopenshell.open(ifc_file_path)
    header_desensitization(ifc_model)
    model_desensitization(ifc_model)
    spatial_element_desensitization(ifc_model)
    entity_desensitization(ifc_model)
    new_ifc_file_path = r"Q:\pychem_project\BIMCompNet\case\desensitization_test.ifc"
    ifc_model.write(new_ifc_file_path)

