import os
import re
import csv
import math
import torch
import argparse
from Bio import PDB
import pickle
import pandas as pd

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from joblib import Parallel, delayed


def get_chain_atom_coordinates(pdb_file, chain1, chain2):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms1 = []
    atoms_coord1 = []
    atoms2 = []
    atoms_coord2 = []
    for model in structure:
        for chain in model:
            if chain.id in chain1:  # 只提取指定链的原子
                for residue in chain:
                    for atom in residue:
                        atoms_coord1.append(atom.coord)
                        atoms1.append(atom)

            if chain.id in chain2:  # 只提取指定链的原子
                for residue in chain:
                    for atom in residue:
                        atoms_coord2.append(atom.coord)
                        atoms2.append(atom)

    return atoms1, atoms2, np.array(atoms_coord1), np.array(atoms_coord2)


# 蛋白质相互作用位点的筛选函数
def identify_interaction_sites(atoms1, atoms2, atom_coords1, atom_coords2, threshold=5.0):
    # 计算两条链之间的距离矩阵
    print("all atoms length: ", len(atom_coords1) + len(atom_coords2))
    print("atom_coords1 shape:", np.shape(atom_coords1))
    print("atom_coords2 shape:", np.shape(atom_coords2))
    distances = cdist(atom_coords1, atom_coords2)

    # 找到距离小于阈值的原子对
    interaction_pairs = np.where(distances < threshold)

    # 返回链1和链2中参与相互作用的原子索引
    interaction_atoms1 = np.unique(interaction_pairs[0])
    interaction_atoms2 = np.unique(interaction_pairs[1])
    atoms = []
    for i in interaction_atoms1:
        atoms.append(atoms1[i])
    for i in interaction_atoms2:
        atoms.append(atoms2[i])

    print("all interaction atoms length: ", len(atoms))

    return atoms


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def match_feature(x, all_for_assign):
    x_p = np.zeros((len(x), 7))

    for j in range(len(x)):
        if x[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif x[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif x[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif x[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif x[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif x[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif x[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif x[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif x[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif x[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif x[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif x[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif x[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif x[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif x[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif x[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif x[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif x[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif x[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif x[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]

    return x_p


def read_atoms(file, chain1, chain2):
    atoms1, atoms2, atom_coords1, atom_coords2 = get_chain_atom_coordinates(file, chain1, chain2)

    interaction_atoms = identify_interaction_sites(atoms1, atoms2, atom_coords1, atom_coords2, threshold=4.0)
    atoms = []
    ajs = []
    for atom in interaction_atoms:
        position = atom.get_coord()
        # residue_name = atom.get_parent().get_resname()
        atoms.append((float(position[0]), float(position[1]), float(position[2])))
        ajs.append(atom.get_name())

    return atoms, ajs  # list[(x,y,z)] list["atom_name"]


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms) - 2):
        for j in range(i + 2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts


def knn(atoms, k=5):
    x = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            x[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(x, axis=-1)

    contacts = []
    for i in range(len(atoms)):
        num = 0
        for j in range(len(atoms)):
            if index[i, j] != i and index[i, j] != i - 1 and index[i, j] != i + 1:
                contacts.append((i, index[i, j]))
                num += 1
            if num == k:
                break

    return contacts


def pdb_to_cm(file, chain1, chain2, threshold):
    atoms, x = read_atoms(file, chain1, chain2)
    r_contacts = compute_contacts(atoms, threshold)
    k_contacts = knn(atoms)
    return r_contacts, k_contacts, x

node_list = []
r_edge_list = []
k_edge_list = []


def data_processing(pdb_file, chain1, chain2):
    # Generate Adjacency Matrix
    distance = 10
    all_for_assign = np.loadtxt("./all_assign.txt")
    pdb_file_dir = ("./results/") + pdb_file

    if os.path.exists(pdb_file_dir):
        r_contacts, k_contacts, x = pdb_to_cm(open(pdb_file_dir, "r"), chain1, chain2, distance)
        # x = match_feature(x, all_for_assign)

        node_list.append(x)
        r_edge_list.append(r_contacts)
        k_edge_list.append(k_contacts)

    else:
        print(pdb_file_dir, "not found")


# 读取CSV文件
file_path = "./processed_skempi.csv"
df = pd.read_csv(file_path)

for i in tqdm(range(len(df))):
    if df.loc[i, "pdb_id"] == '1KBH':
        continue

    pdb_file = "WT_" + df.loc[i, "pdb_id"] + '_Repair_' + str(df.loc[i, "file_name"]) + '.pdb'
    chain1 = df.loc[i, "chain_a"]
    chain2 = df.loc[i, "chain_b"]
    data_processing(pdb_file, chain1, chain2)

np.save("./rball.edges_WT.npy", np.asarray(r_edge_list, dtype=object))
np.save("./knn.edges_WT.npy", np.array(k_edge_list, dtype=object))
torch.save(node_list, "./nodes_WT.pt")

