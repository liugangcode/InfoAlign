import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .conv import GNN_node, GNN_node_Virtualnode

from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

class GNN(torch.nn.Module):
    def __init__(
        self,
        num_tasks=None, # to remove
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        graph_pooling="max",
        norm_layer="batch_norm",
        decoder_dims=[1024, 1111, 862, 1783, 966, 978],
        # mol, gene (gc, go), cell (bray, jump), express
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim ,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        )

        self.decoder_list = nn.ModuleList()
        for out_dim in decoder_dims:
            self.decoder_list.append(MLP(emb_dim, hidden_features=emb_dim * 4, out_features=out_dim))


    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        mu, sigma = self.dist_net(h_graph).chunk(2, dim=1)
        sigma = softplus(sigma) + 1e-7 
        p_z_given_x = Independent(Normal(loc=mu, scale=sigma), 1)
        
        out = []
        out.append(p_z_given_x)
        # p, mol, gene (gc, go), cell (bray, jump), express
        for decoder in self.decoder_list:
            out.append(decoder(mu))
        out_gene = torch.cat((out[2], out[3]), dim=1)
        out_cell = torch.cat((out[4], out[5]), dim=1)
        return [out[0], out[1], out_gene, out_cell, out[6]]

# define a new finetune model with the same architecture of GNN with a new MLP

class FineTuneGNN(nn.Module):
    def __init__(
        self,
        num_tasks=None,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        graph_pooling="max",
        norm_layer="batch_norm",
    ):
        super(FineTuneGNN, self).__init__()

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        )

        # self.task_decoder = nn.Linear(emb_dim, num_tasks)
        self.task_decoder = MLP(emb_dim, hidden_features=4 * emb_dim, out_features=num_tasks)
    
    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        mu, _ = self.dist_net(h_graph).chunk(2, dim=1)
        task_out = self.task_decoder(mu)
        return task_out

    def load_pretrained_graph_encoder(self, model_path):
        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        graph_encoder_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('graph_encoder.')}
        graph_encoder_state_dict = {key.replace('graph_encoder.', ''): value for key, value in graph_encoder_state_dict.items()}
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)
        # Load dist_net state dictionary
        dist_net_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('dist_net.')}
        dist_net_state_dict = {key.replace('dist_net.', ''): value for key, value in dist_net_state_dict.items()}
        self.dist_net.load_state_dict(dist_net_state_dict)
        self.freeze_graph_encoder()

    def freeze_graph_encoder(self):
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.dist_net.parameters():
            param.requires_grad = False

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        return x