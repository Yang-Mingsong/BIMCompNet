import pymeshlab
from pymeshlab import PyMeshLabException
import os
import numpy as np


def get_files_with_extension(directory, extension):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matched_files.append(os.path.join(root, file))
    return matched_files


# fill for n < max_faces with randomly picked faces
def fill_face_to_max(max_faces, face, neighbor_index):
    num_point = len(face)
    if not num_point < max_faces:
        return face, neighbor_index
    fill_face = []
    fill_neighbor_index = []
    for i in range(max_faces - num_point):
        index = np.random.randint(0, num_point)
        fill_face.append(face[index])
        fill_neighbor_index.append(neighbor_index[index])
    face = np.concatenate((face, np.array(fill_face)))
    neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))
    return face, neighbor_index


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


def process_mesh(path, max_faces):
    ms = pymeshlab.MeshSet()
    ms.clear()
    # load mesh
    try:
        ms.load_new_mesh(path)
    except PyMeshLabException as e:
        print('读取错误', e, path)
        return None, None
    
    # 降采样到max_faces
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=max_faces)
    mesh = ms.current_mesh()
    
    # # clean up
    # mesh, _ = pymesh.remove_isolated_vertices(mesh)
    # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

    # get elements
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    if faces.shape[0] > max_faces:     # only occur once in train set of Manifold40
        print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], path))
        return None, None

    # move to center
    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center

    # normalize
    max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
    vertices /= np.sqrt(max_len)

    # get normal vector
    ms.clear()
    mesh = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(mesh)
    face_normal = ms.current_mesh().face_normal_matrix()

    # get neighbors
    faces_contain_this_vertex = []
    for i in range(len(vertices)):
        faces_contain_this_vertex.append(set([]))
    centers = []
    corners = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    faces = np.concatenate([centers, corners, face_normal], axis=1)
    neighbors = np.array(neighbors)
    # 调整面的数量
    faces, neighbors = fill_face_to_max(max_faces, faces, neighbors)
    return faces, neighbors
