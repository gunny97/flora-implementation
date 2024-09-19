import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, num_adapters=32):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.num_adapters = num_adapters
        self.adapter_b_list = nn.ParameterList([nn.Parameter(torch.randn(in_features, rank)) for _ in range(num_adapters)])
        self.adapter_a_list = nn.ParameterList([nn.Parameter(torch.randn(rank, out_features)) for _ in range(num_adapters)])

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()

        inputs = x.unsqueeze(1).expand(batch_size, self.num_adapters, seq_length, d_model).reshape(batch_size * self.num_adapters, seq_length, d_model)  
        
        adapter_bs = torch.stack(list(self.adapter_b_list), dim=0).unsqueeze(1).expand(self.num_adapters, batch_size, -1, -1).reshape(batch_size * self.num_adapters, d_model, self.rank)
        adapters_as = torch.stack(list(self.adapter_a_list), dim=0).unsqueeze(1).expand(self.num_adapters, batch_size, -1, -1).reshape(batch_size * self.num_adapters, self.rank, d_model)
        
        hidden = torch.bmm(inputs, adapter_bs)
        output = torch.bmm(hidden, adapters_as).reshape(batch_size, self.num_adapters, seq_length, d_model).mean(dim=1)
        output = F.relu(output)
        return output

class LoRATransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', rank=4, num_adapters=32, vocab_size=10000):
        super(LoRATransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lora_q = LoRALayer(d_model, d_model, rank=rank, num_adapters=num_adapters)
        self.lora_k = LoRALayer(d_model, d_model, rank=rank, num_adapters=num_adapters)
        # self.lora_v = LoRALayer(d_model, d_model, rank=rank, num_adapters=num_adapters)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        if tgt.dim() == 2:
            tgt = self.embedding(tgt.long())
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
        self.adapter_b = nn.Parameter(torch.randn(batch_size, in_features, rank))
        self.adapter_a = nn.Parameter(torch.randn(batch_size, rank, out_features))
        self.W0 = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        x = x.to(torch.float16)
        batch_size, seq_length, d_model = x.size()
        
        if self.rank == 1:
            adapter_b = self.adapter_b.squeeze(-1)
            adapter_a = self.adapter_a.squeeze(-2)

            Bx = adapter_b.unsqueeze(1) * x
            BxW0 = torch.matmul(Bx, self.W0)
            ABxW0 = adapter_a.unsqueeze(1) * BxW0

            output = F.relu(ABxW0)
        else:
            x_expanded = x.unsqueeze(-1)
            adapter_bs = self.adapter_b.unsqueeze(1).expand(batch_size, seq_length, -1, -1)
            bx = adapter_bs * x_expanded

            bx = bx.view(batch_size * seq_length, d_model, self.rank)
            wo_expanded = self.W0.unsqueeze(0).expand(batch_size * seq_length, -1, -1)
            bxwo = torch.matmul(bx.transpose(1, 2), wo_expanded)

            bxwo = bxwo.view(batch_size, seq_length, d_model, self.rank)
            abxwo = self.adapter_a.unsqueeze(1).expand(batch_size, seq_length, -1, -1).permute(0, 1, 3, 2) * bxwo
            output = torch.mean(abxwo, dim=-1)

            output = F.relu(output)
            
        return output
    
class FLoRATransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', rank=4, batch_size=32, vocab_size=10000):
        super(FLoRATransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.flora_q = FLoRALayer(d_model, d_model, rank=rank, batch_size=batch_size)
        self.flora_k = FLoRALayer(d_model, d_model, rank=rank, batch_size=batch_size)
        # self.flora_v = FLoRALayer(d_model, d_model, rank=rank, batch_size=batch_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        if tgt.dim() == 2:
            tgt = self.embedding(tgt.long())

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
