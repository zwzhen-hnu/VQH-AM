from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
import csv
import pickle
import torch
import dgl
import json
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
num_atom_types = len(atom_types)
atom_to_index = {atom: i for i, atom in enumerate(atom_types)}


def load_graph(prot_r_edge_path, prot_k_edge_path, prot_node_path, outdir):
    if "WT" in prot_r_edge_path:
        outdir = outdir + "sub5.wild_graphs.pkl"
    else:
        outdir = outdir + "sub5.mutation_graphs.pkl"

    if os.path.exists(outdir):
        with open(outdir, "rb") as tf:
            prot_graph_list = pickle.load(tf)
    else:
        prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
        prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
        prot_node = torch.load(prot_node_path)
        prot_graph_list = []
        for i in tqdm(range(len(prot_r_edge))):
            prot_seq = []
            n = len(prot_node[i])
            for j in range(n):  # 顺序边（threshold = 3）
                for k in range(max(0, j - 3), min(n, j + 4)):
                    if k != j:
                        prot_seq.append((j, k))

            # prot_g = dgl.graph(prot_edge[i]).to(device)
            prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                                      ('amino_acid', 'STR_KNN', 'amino_acid'): prot_k_edge[i],
                                      ('amino_acid', 'STR_DIS', 'amino_acid'): prot_r_edge[i]}).to(device)
            # try:
            atom_indices = torch.tensor([atom_to_index[atom] for atom in prot_node[i]])
            # except:
            #     exit(prot_node[i])

            one_hot_encoded_atoms = F.one_hot(atom_indices, num_classes=num_atom_types).float()
            prot_g.ndata['x'] = one_hot_encoded_atoms.to(device)

            prot_graph_list.append(prot_g)

        with open(outdir, "wb") as tf:
            pickle.dump(prot_graph_list, tf)

    return prot_graph_list


class skempi2_dataset(Dataset):
    def __init__(self, processed_path, data_type='all'):
        # mutation_structure
        self.mutation_graph_list = load_graph(processed_path + "/sub/sub.mutation.rball.edges.npy",
                                              processed_path + "/sub/sub.mutation.knn.edges.npy",
                                              processed_path + "/sub/sub.mutation.nodes.pt", processed_path)

        # wild_structure
        self.wild_graph_list = load_graph(processed_path + "/sub/sub.wild.rball.edges.npy",
                                          processed_path + "/sub/sub.wild.knn.edges.npy",
                                          processed_path + "/sub/sub.wild.nodes.pt", processed_path)

        #         # mutation_structure
        #         self.mutation_graph_list = load_graph(processed_path + "mutation.rball.edges.npy",
        #                                               processed_path + "mutation.knn.edges.npy",
        #                                               processed_path + "mutation.nodes.pt", processed_path)

        #         # wild_structure
        #         self.wild_graph_list = load_graph(processed_path + "wild.rball.edges.npy",
        #                                           processed_path + "wild.knn.edges.npy",
        #                                           processed_path + "wild.nodes.pt", processed_path)

        assert len(self.mutation_graph_list) == len(self.wild_graph_list)

        # ddG
        # pdb_name
        # mutation
        self.ddG = []
        self.pdb_name = []
        self.mutation = []

        with open(processed_path + "/processed_skempi.csv") as f:
            reader = csv.reader(f)
            skip = True
            for row in reader:
                if skip:
                    skip = False
                    continue
                if row[1] != "1KBH_A_B" and row[1] != "3VR6_ABCDEF_GH":
                    self.ddG.append(float(row[34]))
                    self.pdb_name.append(row[1])
                    self.mutation.append(row[40])

        print(len(self.ddG))
        assert len(self.mutation_graph_list) == len(self.ddG)
        if data_type == 'single' or data_type == 'mult':
            ddG = []
            pdb_name = []
            mutation = []

            if data_type == 'single':
                for i in range(len(self.ddG)):
                    if ',' not in self.mutation[i]:
                        ddG.append(self.ddG[i])
                        pdb_name.append(self.pdb_name[i])
                        mutation.append(self.mutation[i])
            if data_type == 'mult':
                for i in range(len(self.ddG)):
                    if ',' in self.mutation[i]:
                        ddG.append(self.ddG[i])
                        pdb_name.append(self.pdb_name[i])
                        mutation.append(self.mutation[i])

            self.ddG = ddG
            self.pdb_name = pdb_name
            self.mutation = mutation

        print(data_type)
        print(self.mutation)

    def __len__(self):
        return len(self.ddG)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict["mutation_graph"] = self.mutation_graph_list[idx]
        data_dict["wild_graph"] = self.wild_graph_list[idx]
        data_dict["mutation"] = self.mutation[idx]
        data_dict["pdb_name"] = self.pdb_name[idx]
        data_dict["ddG"] = self.ddG[idx]

        return data_dict


def collate(data):
    collated_data = {}
    collated_data["mutation_graph"] = []
    collated_data["wild_graph"] = []
    collated_data["mutation"] = []
    collated_data["pdb_name"] = []
    collated_data["ddG"] = []

    for data_dict in data:
        collated_data["mutation_graph"].append(data_dict["mutation_graph"])
        collated_data["wild_graph"].append(data_dict["wild_graph"])
        collated_data["mutation"].append(data_dict["mutation"])
        collated_data["pdb_name"].append(data_dict["pdb_name"])
        collated_data["ddG"].append(data_dict["ddG"])

    collated_data["ddG"] = torch.FloatTensor(np.array(collated_data["ddG"]))
    collated_data["mutation_graph"] = dgl.batch_hetero(collated_data["mutation_graph"])
    collated_data["wild_graph"] = dgl.batch_hetero(collated_data["wild_graph"])
    return collated_data
