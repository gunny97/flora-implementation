import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, num_adapters=32):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.num_adapters = num_adapters
        self.adapter_b_list = nn.ParameterList([nn.Parameter(torch.randn(in_features, rank)) for _ in range(num_adapters)])  # [num_adapters, d_model, rank]
        self.adapter_a_list = nn.ParameterList([nn.Parameter(torch.randn(rank, out_features)) for _ in range(num_adapters)])  # [num_adapters, rank, d_model]

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.size()

        # inputs_expanded: [batch_size * num_adapters, seq_length, d_model]
        inputs_expanded = x.unsqueeze(1).expand(batch_size, self.num_adapters, seq_length, d_model).reshape(batch_size * self.num_adapters, seq_length, d_model)  
        
        # Stack adapter_b and adapter_a for bmm
        adapter_b_expanded = torch.stack(list(self.adapter_b_list), dim=0).unsqueeze(1).expand(self.num_adapters, batch_size, -1, -1).reshape(batch_size * self.num_adapters, d_model, self.rank)  # adapter_b_expanded: [batch_size * num_adapters, d_model, rank]
        adapter_a_expanded = torch.stack(list(self.adapter_a_list), dim=0).unsqueeze(1).expand(self.num_adapters, batch_size, -1, -1).reshape(batch_size * self.num_adapters, self.rank, d_model)  # adapter_a_expanded: [batch_size * num_adapters, rank, d_model]
        
        # Compute hidden and adapter_out using bmm
        hidden = torch.bmm(inputs_expanded, adapter_b_expanded)  # hidden: [batch_size * num_adapters, seq_length, rank]
        adapter_out = torch.bmm(hidden, adapter_a_expanded).reshape(batch_size, self.num_adapters, seq_length, d_model).mean(dim=1)  # adapter_out: [batch_size, seq_length, d_model]
        output = F.relu(adapter_out)
        return output

class LoRATransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', rank=4, num_adapters=32, vocab_size=10000):
        super(LoRATransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lora_q = LoRALayer(d_model, d_model, rank=rank, num_adapters=num_adapters)
        self.lora_k = LoRALayer(d_model, d_model, rank=rank, num_adapters=num_adapters)
        self.lora_v = LoRALayer(d_model, d_model, rank=rank, num_adapters=num_adapters)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        # Ensuring tgt is in correct shape and type for embedding layer
        if tgt.dim() == 2:  # First time it passes through
            tgt = self.embedding(tgt.long())  # tgt shape: [batch_size, seq_length, d_model]
        q = self.lora_q(tgt)
        k = self.lora_k(tgt)
        v = tgt
        
        tgt2, _ = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class FLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, batch_size=32):
        super(FLoRALayer, self).__init__()
        self.rank = rank
        self.batch_size = batch_size
        self.adapter_b = nn.Parameter(torch.randn(batch_size, in_features, rank))  # [batch_size, d_model, rank]
        self.adapter_a = nn.Parameter(torch.randn(batch_size, rank, out_features))  # [batch_size, rank, d_model]
        self.W0 = nn.Parameter(torch.randn(in_features, out_features))  # [d_model, d_model]

    def forward(self, x):
        x = x.to(torch.float16)
        batch_size, seq_length, d_model = x.size()  # x: [batch_size, seq_length, d_model]
        
        if self.rank == 1:
            adapter_b = self.adapter_b.squeeze(-1)  # [batch_size, d_model]
            adapter_a = self.adapter_a.squeeze(-2)  # [batch_size, d_model]

            Bx = adapter_b.unsqueeze(1) * x  # [batch_size, seq_length, d_model]
            BxW0 = torch.matmul(Bx, self.W0)  # [batch_size, seq_length, d_model]
            ABxW0 = adapter_a.unsqueeze(1) * BxW0  # [batch_size, seq_length, d_model]

            output = F.relu(ABxW0)  # Apply activation function
        else:
            x_expanded = x.unsqueeze(-1)  # x_expanded: [batch_size, seq_length, d_model, 1]
            adapter_b_expanded = self.adapter_b.unsqueeze(1).expand(batch_size, seq_length, -1, -1)  # [batch_size, seq_length, d_model, rank]
            Bx = adapter_b_expanded * x_expanded  # Bx: [batch_size, seq_length, d_model, rank]

            Bx = Bx.view(batch_size * seq_length, d_model, self.rank)  # Bx: [batch_size * seq_length, d_model, rank]
            W0_expanded = self.W0.unsqueeze(0).expand(batch_size * seq_length, -1, -1)  # W0_expanded: [batch_size * seq_length, d_model, d_model]
            BxW0 = torch.matmul(Bx.transpose(1, 2), W0_expanded)  # [batch_size * seq_length, rank, d_model]

            BxW0 = BxW0.view(batch_size, seq_length, d_model, self.rank)  # BxW0: [batch_size, seq_length, d_model, rank]
            ABxW0 = self.adapter_a.unsqueeze(1).expand(batch_size, seq_length, -1, -1).permute(0, 1, 3, 2) * BxW0  # ABxW0: [batch_size, seq_length, d_model, rank]
            reduced_ABxW0 = torch.mean(ABxW0, dim=-1)  # reduced_ABxW0: [batch_size, seq_length, d_model]

            output = F.relu(reduced_ABxW0)
            
        return output  # [batch_size, seq_length, d_model]
class FLoRATransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', rank=4, batch_size=32, vocab_size=10000):
        super(FLoRATransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.flora_q = FLoRALayer(d_model, d_model, rank=rank, batch_size=batch_size)
        self.flora_k = FLoRALayer(d_model, d_model, rank=rank, batch_size=batch_size)
        self.flora_v = FLoRALayer(d_model, d_model, rank=rank, batch_size=batch_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        if tgt.dim() == 2:  # First time it passes through
            tgt = self.embedding(tgt.long())  # tgt shape: [batch_size, seq_length, d_model]

        q = self.flora_q(tgt)
        k = self.flora_k(tgt)
        v = tgt
        
        tgt2, _ = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt
