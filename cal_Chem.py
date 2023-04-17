import numpy as np
import sys
import time
import math
# import lsgeom
import treelib

# elements
elemord = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
           'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
           'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
           'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
           'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
           'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
           'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
           'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
           'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
           'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
           'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
           'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
           'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
           'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
           'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75,
           'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
           'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
           'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
           'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
           'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
           'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
           'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
           'Rg': 111, 'Uub': 112, 'Uut': 113, 'Uuq': 114, 'Uup': 115,
           'Uuh': 116
           }

# atomic radius
radius = [0.0,  # let [0] as 0
          0.30, 1.16, 1.23, 0.89, 0.88,
          0.77, 0.70, 0.66, 0.58, 0.55,
          1.40, 1.36, 1.25, 1.17, 1.10,
          1.11, 0.99, 1.58, 2.03, 1.74,
          1.44, 1.32, 1.20, 1.13, 1.17,
          1.16, 1.16, 1.15, 1.17, 1.25,
          1.25, 1.22, 1.21, 1.17, 1.14,
          1.89, 2.25, 1.92, 1.62, 1.45,
          1.34, 1.29, 1.23, 1.24, 1.25,
          1.28, 1.34, 1.41, 1.50, 1.40,
          1.41, 1.37, 1.33, 2.09, 2.35,
          1.98, 1.69, 1.65, 1.65, 1.64,
          1.64, 1.66, 1.85, 1.61, 1.59,
          1.59, 1.58, 1.57, 1.56, 1.70,
          1.56, 1.44, 1.34, 1.30, 1.28,
          1.26, 1.26, 1.29, 1.34, 1.44,
          1.55, 1.54, 1.52, 1.53, 1.52,
          1.53, 2.45, 2.02, 1.70, 1.63,
          1.46, 1.40, 1.36, 1.25, 1.57,
          1.58, 1.54, 1.53, 1.84, 1.61,
          1.50, 1.49, 1.38, 1.36, 1.26,
          1.20, 1.16, 1.14, 1.06]

# atomic mass
atomwt = [0.0,
          1.00794, 4.002602, 6.941, 9.012182, 10.81,  # H - B
          12.011, 14.00674, 15.9994, 18.9984032, 20.1797,  # C - Ne
          22.989768, 24.305, 26.981539, 28.0855, 30.973762,  # Na - P
          32.066, 35.4527, 39.948, 39.0983, 40.078,  # S - Ca
          44.955910, 47.88, 50.9415, 51.9961, 54.93805,  # Sc - Mn
          55.847, 58.93320, 58.6934, 63.546, 65.38,  # Fe - Zn
          69.723, 72.61, 74.92159, 78.96, 79.904,  # Ga - Br
          83.80, 85.4678, 87.62, 88.90585, 91.224,  # Kr - Zr
          92.9064, 95.94, 98.9062, 101.07, 102.9055,  # Nb - Rh
          106.42, 107.868, 112.411, 114.818, 118.710,  # Pd - Sn
          121.75, 127.60, 126.9045, 131.29, 132.9054,  # Sb - Cs
          137.327, 138.9055, 140.115, 140.90765, 144.24,  # Ba - Nd
          146.9151, 150.36, 151.965, 157.25, 158.92534,  # Pm - Tb
          162.50, 164.93032, 167.26, 168.9342, 173.04,  # Dy - Yb
          174.967, 178.49, 180.9479, 183.85, 186.207,  # Lu - Re
          190.23, 192.22, 195.08, 196.96654, 200.59,  # Os - Hg
          204.3833, 207.2, 208.98037, 208.9824, 209.9871,  # Tl - At
          222.0176, 223.0197, 226.0254, 227.0278, 232.0381,  # Rn - Th
          231.0359, 238.0289, 237.0482, 244.0642, 243.0614,  # Pa - Am
          247.0703, 247.0703, 251.0796, 252.0829, 257.0951,  # Cm - Fm
          258.0986, 259.1009, 262.11, 261.1087, 262.1138,  # Md - Db
          263.1182, 262.1229, 265., 266.]  # Sg - Mt


def calDistance(coord1, coord2):
    distance = 0
    """计算两个坐标之间的距离"""
    for i in range(len(coord1)):
        distance += (coord1[i] - coord2[i]) ** 2
    distance = np.sqrt(distance)
    return distance


if __name__ == "__main__":

    time.clock()
    '''.gif文件路径'''
    file_full_name = 'ion.gjf'  # gif文件路径
    Frg_save_path = ''  # frg文件结果保存路径

    isWriter2Frg = True  # 是否生成frg文件
    isShowRes = True  # 是否控制台输出结果

    Frg_save_name = file_full_name.split('\\')[-1][:-4]
    Frg_save_full_path = Frg_save_path + '\\' + Frg_save_name + '.frg'

    fo = open(file_full_name, "r")
    lines = fo.readlines()

    start_line_idx = 0
    blank_line_num = 0
    for idx, line in enumerate(lines):
        if line == '\n':
            blank_line_num += 1
        if blank_line_num == 2:
            start_line_idx = idx + 1
            break

    end_line_idx = 0
    for idx, line in enumerate(lines[start_line_idx + 1:]):
        if line == '\n':
            end_line_idx = start_line_idx + idx + 1
            atom_num = idx  # 获取文本文件内原子总数
            break
    if end_line_idx == 0:
        lines = lines[start_line_idx:]
    else:
        lines = lines[start_line_idx:end_line_idx]

    eletron_spin_line = lines[0].strip().split()
    eletron = int(eletron_spin_line[0])
    spin_multiplicity = int(eletron_spin_line[1])

    # atom_num = int(len(lines[start_line_idx + 3 : ]))  # 获取文本文件内原子总数
    atom_name = []
    atom_coord = np.zeros((atom_num, 3))

    # 遍历文件.xyz第三行到最后一行
    for i, line in enumerate(lines[1:]):
        line = line.strip().split()
        # 读原子名称,即每行第一个元素
        atom_name.append(line[0])
        # 读每个原子的坐标，即每行第2-4个元素
        atom_coord[i][0] = float(line[1])
        atom_coord[i][1] = float(line[2])
        atom_coord[i][2] = float(line[3])

    # 存储不同原子之间的距离，比如二维数组的第(i, j)元素就是第i+1个原子和第j+1个原子之间的距离
    # 这也说明，该矩阵是个上三角矩阵。下三角部分和上三角一样，不需要关注，对角线也不用管
    distance_matrix = np.zeros((atom_num, atom_num))

    for i in range(atom_num):
        for j in range(i + 1, atom_num):
            distance_matrix[i][j] = calDistance(atom_coord[i], atom_coord[j])

    # 存储不同原子的距离临界条件，也是一个二维数组
    # 二维数组的第(i, j)元素就是第i+1个原子和第j+1个原子之间的临界距离
    # 如果实际距离大于临界距离,就说明这两个原子不是一个分子,反之,则是同一个分子
    # 这里的临界距离等于两个原子半径之和的1.25倍
    threshold_matrix = np.zeros((atom_num, atom_num))

    element_unique = list(set(atom_name))
    element_unique_num = len(element_unique)
    atompair_distance = {}
    for i in range(element_unique_num):
        for j in range(element_unique_num):
            dict_key = element_unique[i] + element_unique[j]
            dict_value = (radius[elemord[element_unique[i]]]
                          + radius[elemord[element_unique[j]]]) * 1.25
            atompair_distance[dict_key] = dict_value
    # 计算不同原子之间的临近距离,与distance_matrix同理,threshold_matrix 也是一个上三角矩阵
    for i in range(atom_num):
        for j in range(i + 1, atom_num):
            threshold_matrix[i][j] = atompair_distance[atom_name[i] + atom_name[j]]

    temp = treelib.Tree()
    temp.create_node(tag=str(0 + 1), identifier=str(0 + 1), data=atom_name[0])
    cluster_list = [temp]
    logit = (distance_matrix < threshold_matrix)
    for i in range(atom_num):
        isTotalRowFalse = True  # 判断是否整行都是False,如果是,则原子i孤立,不与任何原子相连
        for j in range(i + 1, atom_num):
            if logit[i][j]:
                isTotalRowFalse = False
                isNocluster_contain_i = True  # 是否所有的树都没有原子i
                for k in range(len(cluster_list)):
                    # 检查第k个树中是否有原子i
                    if cluster_list[k].contains(str(i + 1)):
                        # 如果第k个树中有原子i,则将原子j添加成为其子节点
                        cluster_list[k].create_node(tag=str(j + 1), identifier=str(j + 1), parent=str(i + 1),
                                                    data=atom_name[j])
                        isNocluster_contain_i = False
                        break
                # 如果所有的树都没有原子i,则说明原子i尚未和任何原子相连
                # 则创建新的tree,此时原子i为第一个节点,且有子节点j
                if isNocluster_contain_i:
                    temp = treelib.Tree()
                    temp.create_node(tag=str(i + 1), identifier=str(i + 1), data=atom_name[i])
                    temp.create_node(tag=str(j + 1), identifier=str(j + 1), parent=str(i + 1), data=atom_name[j])
                    cluster_list.append(temp)
        if i != 0 and isTotalRowFalse:
            for k in range(len(cluster_list)):
                # 检查第k个树中是否有原子i
                if cluster_list[k].contains(str(i + 1)):
                    isTotalRowFalse = False
            if isTotalRowFalse:
                temp = treelib.Tree()
                temp.create_node(tag=str(i + 1), identifier=str(i + 1), data=atom_name[i])
                cluster_list.append(temp)
    if isShowRes:
        for idx in range(len(cluster_list)):
            cluster_list[idx].show()

    # 第一种原子序号标记方式,如(1,2,3,4)
    if isWriter2Frg:
        with open(Frg_save_full_path, 'w') as f:
            for idx, cluster in enumerate(cluster_list):
                f.write(str(idx + 1))
                f.write('   ')
                f.write(str(spin_multiplicity))
                f.write('  (')
                write_line = []
                for node_id in cluster.expand_tree():
                    write_line.append(int(node_id))
                write_line.sort()
                for i in range(len(write_line)):
                    f.write(str(write_line[i]))
                    if i != len(write_line) - 1:
                        f.write(',')
                f.write(')   ')
                f.write(str(eletron))
                f.write('\n')

    # 第二种原子序号标记方式,如(1-4)
    # if isWriter2Frg:
    #     with open(Frg_save_full_path, 'w') as f:
    #         for idx, cluster in enumerate(cluster_list):
    #             f.write(str(idx + 1))
    #             f.write('   ')
    #             f.write(str(spin_multiplicity))
    #             f.write('  (')
    #             write_line = []
    #             for node_id in cluster.expand_tree():
    #                 write_line.append(int(node_id))
    #             f.write(str(min(write_line)))
    #             if len(write_line) != 1:
    #                 f.write('-')
    #                 f.write(str(max(write_line)))
    #             f.write(')   ')
    #             f.write(str(eletron))
    #             f.write('\n')
