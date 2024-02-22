import torch
import torch.nn as nn
import torch.nn.functional as F

#position-wise fully connected feed-forward network
#position-wise means that the network is applied to each position separately and identically
#position-wise: FC is applied along the last dimentions of the input tensor
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU() #Gaussian Error Linear Unit
        # self.act = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# MoE: mixture of experts of FFNs
class SwitchFFN(nn.Module):
    def __init__(self, num_experts, dim, hidden_dim):
        self.num_experts = num_experts
        self.softmax = nn.Softmax(dim=-1)
        # self.experts = nn.ModuleList([FFN(dim, hidden_dim) for _ in range(num_experts)])
        self.experts = nn.ModuleList([])
        for _ in range(self.num_experts):
            self.experts.append(FFN(dim, hidden_dim))
        
        #Routing layer
        self.to_switch = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        # Capture the shape to change shapes later
        batch_size, seq_len, dim = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, dim) # (batch_size*seq_len, dim)
        
        switch_logits = self.to_switch(x)
        rout_prob = self.softmax(switch_logits)
        rout_prob_max, expert_id = torch.max(rout_prob, dim=-1)
        
        num_token_all = x.shape[0]*x.shape[1]
        for t in range(num_token_all):
            

class SwitchTransformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1000, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                ),
                nn.LayerNorm(dim)
            ]))
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        x = self.token_embedding(x) + self.pos_embedding[:, :x.shape[1], :]
        for attn, norm1, mlp, norm2 in self.layers:
            x = x + attn(x, x, x)[0]
            x = norm1(x)
            x = x + mlp(x)
            x = norm2(x)
        return self.to_logits(x)


class SwitchTransformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.heads = heads

        self.token_embedding = nn.Embedding(num_tokens, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim)
            ]))

        self.to_switch = nn.Linear(dim, num_tokens)

    def forward(self, x):
        tokens = torch.arange(self.num_tokens, device=x.device)
        x = self.token_embedding(x) + self.pos_embedding[:, :x.shape[1], :]
        switch_logits = self.to_switch(x.mean(dim=1))

        for attn, norm1, lin1, gelu, lin2, norm2 in self.layers:
            x, _ = attn(x, x, x)
            x = norm1(x)
            x = lin2(gelu(lin1(x)))
            x = norm2(x)

            switch_weights = F.softmax(self.to_switch(x), dim=-1)
            x = torch.einsum('b n d, b d t -> b n t', switch_weights, x)

        return x, switch_logits
