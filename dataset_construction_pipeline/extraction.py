from datetime import datetime
import ifcopenshell
from ifcopenshell.api import run
from ifcopenshell.util import element
import os


def entity_extract(_ifcmodel):
    entity_list = []
    problem_set = set()
    entity_dict = {}
    entity_has_openings = {}
    _ifc_curtain_walls = _ifcmodel.by_type("IfcCurtainWall")
    _ifc_stairs = _ifcmodel.by_type("IfcStair")
    _ifc_ramps = _ifcmodel.by_type("IfcRamp")
    _ifc_roofs = _ifcmodel.by_type("IfcRoof")
    for _special_entity in _ifc_curtain_walls+_ifc_stairs+_ifc_ramps+_ifc_roofs:
        if _special_entity.Representation is None:
            _aggregates_rel = _special_entity.IsDecomposedBy
            if len(_aggregates_rel) == 1:
                _aggregates_entities = _aggregates_rel[0].RelatedObjects
                _aggregates_entities = list(_aggregates_entities)
                entity_dict[_special_entity.id()] = _aggregates_entities
            elif len(_aggregates_rel) == 0:
                problem_set.add(_special_entity)
                print("{}实体没有聚合关系".format(_special_entity.is_a()))
            else:
                problem_set.add(_special_entity)
                print("{}实体有多个聚合关系".format(_special_entity.is_a()))
        else:
            continue

    _entities = _ifcmodel.by_type("IfcElement")
    for _key_entity, _value_entities in entity_dict.items():
        _entity_item = _ifcmodel.by_id(_key_entity)
        if _entity_item.is_a("IfcCurtainWall"):
            _entities = set(_entities) - set(_value_entities) - problem_set
            _entities.remove(_entity_item)
        else:
            _entities.remove(_entity_item)

    for _entity in _entities:
        #_entity.Representation指的是实体的几何表示
        if _entity.HasOpenings is not None and _entity.Representation is not None and len(_entity.HasOpenings) > 0:
            _voids_rels = _entity.HasOpenings
            for _voids_rel in _voids_rels:
                _feature_element_subtraction = _voids_rel.RelatedOpeningElement
                entity_has_openings[_entity.id()] = _feature_element_subtraction

    for _key_entity, _ in entity_has_openings.items():
        _entity_item = _ifcmodel.by_id(_key_entity)
        _entities.remove(_entity_item)

    for _entity in _entities:
        if _entity.Representation is None or _entity.is_a("IfcFeatureElement") or _entity.is_a("IfcVirtualElement") or _entity.is_a("IfcElementAssembly"):
            continue
        else:
            entity_list.append(_entity)
    return entity_list, entity_dict, entity_has_openings


def special_entity_segment(_ifcmodel, _entity):
    # 在方法special_entity_segment(_ifcmodel, _entity)下面增加设置用户的代码
    _model = ifcopenshell.file(schema="IFC4")

    application = ifcopenshell.api.run("owner.add_application", _model)
    person = ifcopenshell.api.run("owner.add_person", _model,
                                  identification="LPARTEE", family_name="Partee", given_name="Leeable")
    organisation = ifcopenshell.api.run("owner.add_organisation", _model,
                                        identification="AWB", name="Architects Without Ballpens")
    user = ifcopenshell.api.run("owner.add_person_and_organisation", _model,
                                person=person, organisation=organisation)
    ifcopenshell.api.owner.settings.get_user = lambda x: user
    ifcopenshell.api.owner.settings.get_application = lambda x: application

    # 创建空的IfcProject实体
    project = run("root.create_entity", _model, ifc_class="IfcProject", name="My Project")
    # 完全复制单位信息实体
    unitassignment = _ifcmodel.by_type('IfcUnitAssignment')[0]
    # 复制的实体增加到新建的Ifc文件中
    newunit = ifcopenshell.file.add(_model, unitassignment)
    # 关联IfcProject和单位信息
    project.UnitsInContext = newunit
    # 完全复制这个实体
    decomposes = _entity.IsDecomposedBy
    if len(decomposes) > 0:
        for decompose in decomposes:
            ifcopenshell.file.add(_model, decompose)
    else:
        print("{},{}没有包含的元素".format(_entity.id(), _entity.is_a()))
    # 关联IfcProject和几何表达精度信息
    representation_contexts = _model.by_type('IfcRepresentationContext')
    for _context in representation_contexts:
        if _context.is_a() == "IfcGeometricRepresentationContext":
            project.RepresentationContexts = [_context]
    # 每个构件实体关联project和site
    site = run("root.create_entity", _model, ifc_class="IfcSite", name="My Site")
    run("aggregate.assign_object", _model, relating_object=project, product=site)
    _new_entities = _model.by_type("IfcElement")
    for _new_entity in _new_entities:
        run("spatial.assign_container", _model, relating_structure=site, product=_new_entity)
        # 获得这个IFC构件的所有属性集
        psets = ifcopenshell.util.element.get_psets(_entity)
        # 给新加的实体增加所有属性
        for pset in list(psets.items()):
            if "Pset_" in pset[0] or pset[0] == "BaseQuantities":
                psetname = pset[0]
                properties = pset[1]
                properties.pop("id")
                addedpset = ifcopenshell.api.run("pset.add_pset", _model, product=_new_entity, name=psetname)
                ifcopenshell.api.run("pset.edit_pset", _model, pset=addedpset, properties=properties)
    return _model


def entity_segment(_ifcmodel, _entity): #用于将单个实体分割到一个新的IFC文件中
    _model = ifcopenshell.file()
    # 创建空的IfcProject实体
    project = run("root.create_entity", _model, ifc_class="IfcProject", name="My Project")
    # 完全复制单位信息实体
    unitassignment = _ifcmodel.by_type('IfcUnitAssignment')[0]
    # 复制的实体增加到新建的Ifc文件中
    newunit = ifcopenshell.file.add(_model, unitassignment)
    # 关联IfcProject和单位信息
    project.UnitsInContext = newunit
    # 完全复制这个实体
    addedelement = ifcopenshell.file.add(_model, _entity)
    # 关联IfcProject和几何表达精度信息
    representation_contexts = _model.by_type('IfcRepresentationContext')
    for _context in representation_contexts:
        if _context.is_a() == "IfcGeometricRepresentationContext":
            project.RepresentationContexts = [_context]
    # 每个构件实体关联project和site
    site = run("root.create_entity", _model, ifc_class="IfcSite", name="My Site")
    run("aggregate.assign_object", _model, relating_object=project, product=site)
    run("spatial.assign_container", _model, relating_structure=site, product=addedelement)
    # 获得这个IFC构件的所有属性集
    psets = ifcopenshell.util.element.get_psets(_entity)
    # 给新加的实体增加所有属性
    for pset in list(psets.items()):
        if "Pset_" in pset[0] or pset[0] == "BaseQuantities":
            psetname = pset[0]
            properties = pset[1]
            properties.pop("id")
            addedpset = ifcopenshell.api.run("pset.add_pset", _model, product=addedelement, name=psetname)
            ifcopenshell.api.run("pset.edit_pset", _model, pset=addedpset, properties=properties)
    return _model


def has_openings_element_segment(_ifcmodel, _entity):
    _model = ifcopenshell.file()
    # 创建空的IfcProject实体
    project = run("root.create_entity", _model, ifc_class="IfcProject", name="My Project")
    # 完全复制单位信息实体
    unitassignment = _ifcmodel.by_type('IfcUnitAssignment')[0]
    # 复制的实体增加到新建的Ifc文件中
    newunit = ifcopenshell.file.add(_model, unitassignment)
    # 关联IfcProject和单位信息
    project.UnitsInContext = newunit
    # 完全复制这个实体
    decomposes = _entity.HasOpenings
    if len(decomposes) > 0:
        for decompose in decomposes:
            ifcopenshell.file.add(_model, decompose)
    else:
        print("{},{}没有包含的元素".format(_entity.id(), _entity.is_a()))
        # 关联IfcProject和几何表达精度信息
    representation_contexts = _model.by_type('IfcRepresentationContext')
    for _context in representation_contexts:
        if _context.is_a() == "IfcGeometricRepresentationContext":
            project.RepresentationContexts = [_context]
    # 每个构件实体关联project和site
    site = run("root.create_entity", _model, ifc_class="IfcSite", name="My Site")
    run("aggregate.assign_object", _model, relating_object=project, product=site)
    _new_entities = _model.by_type("IfcElement")
    for _new_entity in _new_entities:
        if not _new_entity.is_a("IfcFeatureElementSubtraction"):
            run("spatial.assign_container", _model, relating_structure=site, product=_new_entity)
        # 获得这个IFC构件的所有属性集
        psets = ifcopenshell.util.element.get_psets(_entity)
        # 给新加的实体增加所有属性
        for pset in list(psets.items()):
            if "Pset_" in pset[0] or pset[0] == "BaseQuantities":
                psetname = pset[0]
                properties = pset[1]
                properties.pop("id")
                addedpset = ifcopenshell.api.run("pset.add_pset", _model, product=_new_entity, name=psetname)
                ifcopenshell.api.run("pset.edit_pset", _model, pset=addedpset, properties=properties)
    return _model


if __name__ == '__main__':
    IFC_FILE_PATH = r"Q:\pychem_project\BIMCompNet\case\desensitization_test.ifc"
    ifc_model = ifcopenshell.open(IFC_FILE_PATH)
    entity_list, entity_dict, entity_has_openings = entity_extract(ifc_model)
    path = r"Q:\pychem_project\BIMCompNet\case\extraction"

    for entity in entity_list:
        model = entity_segment(ifc_model, entity)
        now_time = datetime.now().strftime("%m%d%H%M%S%f")
        filename = entity.is_a() + "_" + now_time + ".ifc"
        model.write(os.path.join(path, filename))

    for key, entities in entity_dict.items():
        entity = ifc_model.by_id(key)
        model = special_entity_segment(ifc_model, entity)
        now_time = datetime.now().strftime("%m%d%H%M%S%f")
        filename = entity.is_a() + "_" + now_time + ".ifc"
        model.write(os.path.join(path, filename))

    for key, entities in entity_has_openings.items():
        entity = ifc_model.by_id(key)
        model = has_openings_element_segment(ifc_model, entity)
        now_time = datetime.now().strftime("%m%d%H%M%S%f")
        filename = entity.is_a() + "_" + now_time + ".ifc"
        model.write(os.path.join(path, filename))


