import torch 
import torch.nn as nn 
import torch.nn.functional as F
from Layers.Linear import Linear 
from Layers.Dropout import Dropout 
from Layers.RMSprop import RMSnorm
from Layers.SwiGLU import SwiGLU
import math 

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = 4 * cfg["emb_dim"]
        self.layers = nn.Sequential(
            SwiGLU(cfg["emb_dim"], hidden_dim),
            Linear(hidden_dim, cfg["emb_dim"]),
        )
    
    def forward(self, x, return_aux_loss=False):
        output = self.layers(x)
        if return_aux_loss:
            return output, None
        return output

class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"]

        self.load_balance_alpha = cfg.get("load_balance_alpha", 0.01)
        self.capacity_factor = cfg.get("capacity_factor", 1.5)

        self.norm = RMSnorm(cfg["emb_dim"])
        self.gate = Linear(cfg["emb_dim"], cfg["num_experts"], bias=False)
        
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.emb_dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.emb_dim, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.emb_dim))

        self._init_weights()
        
     
        self.register_buffer("expert_counts", torch.zeros(self.num_experts))
        self.register_buffer("total_tokens", torch.tensor(0))
    
    def _init_weights(self):
        """Initialize expert weights using Kaiming initialization"""
        for param in [self.w1, self.w2, self.w3]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    def compute_load_balance_loss(self, router_probs, expert_indices):
        num_tokens = router_probs.shape[0]
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0,1])
        fraction_per_expert = tokens_per_expert / (num_tokens * self.num_experts_per_tok + 1e-8)  #

        
        mean_router_prob = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(fraction_per_expert * mean_router_prob)
        if self.training:
            with torch.no_grad():
                self.expert_counts += tokens_per_expert
                self.total_tokens += num_tokens * self.num_experts_per_tok
        
        return load_balance_loss

    def forward(self, x, return_aux_loss=True):
        batch, seq_len, _ = x.shape 
        x_norm = self.norm(x)
        

        router_logits = self.gate(x_norm)
        router_logits_flat = router_logits.reshape(-1, self.num_experts)
        router_probs = F.softmax(router_logits_flat, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
        x_flat = x_norm.reshape(-1, self.emb_dim)
        num_tokens = x_flat.shape[0]

        capacity = int(self.capacity_factor * num_tokens * self.num_experts_per_tok / self.num_experts) 

        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2,1,0)

        position_in_expert = torch.cumsum(expert_mask, dim=2) * expert_mask
        position_in_expert = position_in_expert - 1

        dispatch_mask = (position_in_expert.unsqueeze(2) == torch.arange(capacity, device=x.device).view(1,1,capacity, 1))
        dispatch_mask = dispatch_mask.any(dim=1)

        expert_inputs = torch.einsum("ect,td->ecd", dispatch_mask.float(), x_flat)

        h1 = torch.einsum("ecd,edh->ech", expert_inputs, self.w1)
        h2 = torch.einsum("ecd,edh->ech", expert_inputs, self.w2)

        hidden = F.silu(h1) * h2
        
        expert_outputs = torch.einsum("ech,ehd->ecd", hidden, self.w3)
        routing_weights = torch.zeros(self.num_experts, self.num_experts_per_tok, num_tokens, device=x.device)
        expert_ids = topk_indices.permute(1, 0)  # [k, num_tokens]
        token_ids = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(self.num_experts_per_tok, -1)  
        k_ids = torch.arange(self.num_experts_per_tok, device=x.device).unsqueeze(1).expand(-1, num_tokens) 
        
        routing_weights[expert_ids.reshape(-1), k_ids.reshape(-1), token_ids.reshape(-1)] = topk_probs.permute(1, 0).reshape(-1)
        combine_weights = torch.einsum("ekt,ect->ect", routing_weights, dispatch_mask.float())
        output = torch.einsum("ect,ecd->td", combine_weights, expert_outputs)
        
        aux_loss = None 
        if return_aux_loss and self.training:
            aux_loss = self.compute_load_balance_loss(router_probs, topk_indices)
        
        return output.reshape(batch, seq_len, self.emb_dim), aux_loss
    
    def get_expert_usage_stats(self):
        """Return statistics about expert usage for monitoring"""
        if self.total_tokens == 0:
            return None
        
        usage_fraction = self.expert_counts / self.total_tokens
        return {
            'expert_counts': self.expert_counts.cpu().numpy(),
            'usage_fraction': usage_fraction.cpu().numpy(),
            'total_tokens': self.total_tokens.item(),
            'std_usage': usage_fraction.std().item(),
        }
    
    def reset_expert_stats(self):
        """Reset expert usage statistics"""
        self.expert_counts.zero_()
        self.total_tokens.fill_(0)
        

        