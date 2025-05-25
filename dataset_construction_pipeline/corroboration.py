import ifcopenshell


def corroborate(_path):
    _messages = []
    model = ifcopenshell.open(_path)
    # 几何精度判断
    geo_contexts = model.by_type("IfcGeometricRepresentationContext")
    for geo_context in geo_contexts:
        if geo_context.ContextType == "Model" and not geo_context.is_a("IfcGeometricRepresentationSubContext"):
            if geo_context.Precision > 0.01:
                _message = "当前模型的几何精度为{},不符合要求".format(geo_context.Precision)
                _messages.append(_message)

    # 基本单位判断
    context_units = model.by_type("IfcUnitAssignment")
    units = list(context_units[0].Units)
    for _unit in units:
        if _unit.UnitType == "PLANEANGLEUNIT":
            if _unit.is_a("IfcConversionBasedUnit") and _unit.Name == 'DEGREE':
                continue
            else:
                _message = "当前模型的角度单位{}信息为{},不符合要求".format(_unit.UnitType, _unit)
                _messages.append(_message)
        elif _unit.UnitType == "LENGTHUNIT":
            if _unit.is_a("IfcSIUnit") and _unit.Prefix == "MILLI" and _unit.Name == 'METRE':
                continue
            else:
                _message = "当前模型的长度单位{}信息为{},不符合要求".format(_unit.UnitType, _unit)
                _messages.append(_message)
        elif _unit.UnitType == "AREAUNIT":
            if _unit.is_a("IfcSIUnit") and _unit.Prefix is None and _unit.Name == 'SQUARE_METRE':
                continue
            else:
                _message = "当前模型的面积单位{}信息为{},不符合要求".format(_unit.UnitType, _unit)
                _messages.append(_message)
        elif _unit.UnitType == "VOLUMEUNIT" and _unit.is_a("IfcSIUnit") and _unit.Prefix is None and _unit.Name == 'CUBIC_METRE':
            if _unit.is_a("IfcSIUnit") and _unit.Prefix is None and _unit.Name == 'CUBIC_METRE':
                continue
            else:
                _message = "当前模型的体积单位{}信息为{},不符合要求".format(_unit.UnitType, _unit)
                _messages.append(_message)
    return _messages


if __name__ == '__main__':
    path = r"Q:\pychem_project\BIMCompNet\case\test.ifc"
    messages = corroborate(path)
    for message in messages:
        print(message)
