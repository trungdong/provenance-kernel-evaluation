import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GraphConv, GATv2Conv


class GCN(torch.nn.Module):
    def __init__(self, num_classes: int, hidden_channels: int):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCNWrapper:
    def __init__(self, num_classes: int, hidden_channels: int):
        self.model = GCN(num_classes, hidden_channels)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_loop(self, data_loader):
        self.model.train()
        for data in data_loader:
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train(self, data_loader, epochs: int):
        for epoch in range(epochs):
            self.train_loop(data_loader)

    def test(self, data_loader):
        self.model.eval()
        correct = 0
        for data in data_loader:
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(data_loader.dataset)


class GATv2(torch.nn.Module):
    def __init__(self, num_classes: int, edge_dim: int, hidden_channels: int, num_heads: int = 1):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(-1, hidden_channels, heads=num_heads, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(num_heads * hidden_channels, hidden_channels, heads=num_heads, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(num_heads * hidden_channels, hidden_channels, heads=num_heads, edge_dim=edge_dim)
        self.lin = Linear(num_heads * hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GATv2Wrapper:
    def __init__(self, num_classes: int, edge_dim: int, hidden_channels: int, num_heads: int = 1):
        self.model = GATv2(num_classes, edge_dim, hidden_channels, num_heads)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_loop(self, loader):
        self.model.train()
        for data in loader:
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train(self, data_loader, epochs: int):
        for epoch in range(epochs):
            self.train_loop(data_loader)

    def test(self, loader):
        self.model.eval()
        correct = 0
        for data in loader:
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)
