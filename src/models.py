import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, HeteroGraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, param):
        super(Encoder, self).__init__()
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
        self.layers.append(HeteroGraphConv({'SEQ': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                                            'STR_KNN': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                                            'STR_DIS': GraphConv(param['input_dim'], param['prot_hidden_dim'])},
                                           aggregate='sum'))

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(HeteroGraphConv({'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                                                'STR_KNN': GraphConv(param['prot_hidden_dim'],
                                                                     param['prot_hidden_dim']),
                                                'STR_DIS': GraphConv(param['prot_hidden_dim'],
                                                                     param['prot_hidden_dim'])}, aggregate='sum'))

    def forward(self, batch_graph):

        x = batch_graph.ndata['x']
        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
            if l != self.num_layers - 1:
                x = self.dropout(x)

        return x

class CodeBook(nn.Module):
    def __init__(self, param):
        super(CodeBook, self).__init__()

        self.param = param

        self.Protein_Encoder = Encoder(param)

        self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

    def forward(self, batch_graph):
        h = self.Protein_Encoder(batch_graph)
        z, z_q_loss, encoding_indices = self.vq_layer(h)
        batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
        # batch_graph.ndata['h'] = h
        prot_embed = dgl.mean_nodes(batch_graph, 'h')

        return h, z, prot_embed, z_q_loss
        # return h, h, prot_embed, prot_embed


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices

    def get_code_indices(self, x):
        distances = (
                torch.sum(x ** 2, dim=-1, keepdim=True) +
                torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
                2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )

        encoding_indices = torch.argmin(distances, dim=1)

        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)


class ddG(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codebook = CodeBook(param)

        self.codebook = self.codebook.to(self.device)

        self.loss_fn = nn.MSELoss().to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(param["prot_hidden_dim"] * 4, param["prot_hidden_dim"] * 2), nn.ReLU(),
            nn.Linear(param["prot_hidden_dim"] * 2, param["prot_hidden_dim"] * 2), nn.ReLU(),
            nn.Linear(param["prot_hidden_dim"] * 2, param["prot_hidden_dim"] * 2), nn.ReLU(),
            nn.Linear(param["prot_hidden_dim"] * 2, param["prot_hidden_dim"] * 2)
        )

        self.project = nn.Linear(param["prot_hidden_dim"] * 2, 1, bias=False)

    def forward(self, mutation_graph, wild_graph, labels):
        _,_,mutation_features,z_q_loss1 = self.codebook(mutation_graph)
        _,_,wild_features,z_q_loss2 = self.codebook(wild_graph)
        vq_loss = z_q_loss1 + z_q_loss2

        x = torch.cat([mutation_features, wild_features], dim=-1)
        x = self.mlp(x)
        x = self.project(x)

        labels = labels.unsqueeze(1)

        loss = self.loss_fn(x.float(), labels.float()) + vq_loss


        return x, loss