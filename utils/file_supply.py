import os
import csv


def get_new_name(part1, part2, part3):

    name = str(part1) + "_" + str(part2) + "_" + str(part3)
    return name


def process_folder(folder_path, f_index):
    mapping = []  # 用于记录 (旧路径, 新路径)

    for ifc_class in os.listdir(folder_path):
        class_path = os.path.join(folder_path, ifc_class)
        for element_index, element_items in enumerate(os.listdir(class_path)):
            element_path = os.path.join(class_path, element_items)
            ifc_path = os.path.join(element_path, "IFC")
            obj_path = os.path.join(element_path, "OBJ")
            ply_path = os.path.join(element_path, "PLY")
            new_file_name = get_new_name(f_index, ifc_class, element_index)
            try:
                for ifc_file in os.listdir(ifc_path):
                    old_ifc_dir = os.path.join(ifc_path, ifc_file)
                    new_ifc_dir = os.path.join(ifc_path, new_file_name + ".ifc")
                    os.rename(old_ifc_dir, new_ifc_dir)
                for obj_file in os.listdir(obj_path):
                    old_obj_dir = os.path.join(obj_path, obj_file)
                    new_obj_dir = os.path.join(obj_path, new_file_name + ".obj")
                    os.rename(old_obj_dir, new_obj_dir)
                for ply_file in os.listdir(ply_path):
                    old_ply_dir = os.path.join(ply_path, ply_file)
                    new_ply_dir = os.path.join(ply_path, new_file_name + ".ply")
                    os.rename(old_ply_dir, new_ply_dir)
            except Exception as e:
                print("Error renaming file {}: {}".format(element_path, e))
        print("文件重命名：{}的{}已完成".format(f_index, ifc_class))

    for ifc_class in os.listdir(folder_path):
        class_path = os.path.join(folder_path, ifc_class)
        for element_index, element_items in enumerate(os.listdir(class_path)):
            element_path = os.path.join(class_path, element_items)
            new_name = get_new_name(f_index, ifc_class, element_index)
            new_element_path = os.path.join(class_path, new_name)
            try:
                os.rename(element_path, new_element_path)
                mapping.append((element_path, new_element_path))
            except Exception as e:
                print("Error renaming folder {}: {}".format(element_path, e))
        print("对象文件夹重命名：{}的{}已完成".format(f_index, ifc_class))

    # 保存映射关系到 CSV 文件（保存于当前主文件夹下）
    csv_file = os.path.join(r"Q:\pychem_project\BIMCompNet\change_csv", str(f_index) + "_rename_mapping.csv")
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Original Path", "New Path"])
            for orig, new in mapping:
                writer.writerow([orig, new])
        print("CSV mapping saved: {}".format(csv_file))
    except Exception as e:
        print("Error writing CSV file in {}: {}".format(csv_file, e))


def process_all_folders(path):
    folders = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]
    print("Found {} folders.".format(len(folders)))
    for f_index, folder in enumerate(folders):
        print("Processing folder:{}: {}".format(f_index, folder))
        process_folder(folder, f_index)


if __name__ == "__main__":
    # 请将此路径替换为你的包含15个文件夹的基目录路径
    base_path = r"H:\G"
    process_all_folders(base_path)
