import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from BioMolX.model import GNN_graphpred
from BioMolX.loader import MoleculeDataset



class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class Meta_model(nn.Module):
    def __init__(self, args):
        super(Meta_model, self).__init__()

        self.device = args['device']
        self.batch_size = args['batch_size']

        self.graph_model = GNN_graphpred(args['num_layer'], args['emb_dim'], 1, JK=args['JK'], drop_ratio=args['dropout_ratio'],
                                         graph_pooling=args['graph_pooling'], gnn_type=args['gnn_type'])
        if not args['input_model_file'] == "":
            self.graph_model.from_pretrained(args['input_model_file'])

        model_param_group = []
        model_param_group.append({"params": self.graph_model.gnn.parameters()})
        if args['graph_pooling'] == "attention":
            model_param_group.append({"params": self.graph_model.pool.parameters(), "lr": args['lr'] * args['lr_scale']})
        model_param_group.append(
            {"params": self.graph_model.graph_pred_linear.parameters(), "lr": args['lr'] * args['lr_scale']})

        self.optimizer = optim.Adam(model_param_group, lr=args['meta_lr'], weight_decay=args['decay'])

    def get_prediction(self, smiles_list):
        self.graph_model.eval()

        dataset = MoleculeDataset(smiles_list=smiles_list)
        query_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        pred_scores = []
        for step, batch in enumerate(query_loader):
            batch = batch.to(self.device)

            pred, _ = self.graph_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, 0)
            pred = torch.sigmoid(pred)
            pred_list = pred.cpu().detach().numpy().tolist()
            flat_list = [item for sublist in pred_list for item in sublist]
            pred_scores = pred_scores + flat_list

        return pred_scores

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return self

