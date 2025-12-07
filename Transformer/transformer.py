import torch 
import torch.nn as nn 
from Layers.Linear import Linear 
from Layers.Dropout import Dropout 
from Layers.RMSprop import RMSnorm
from Layers.MoEFeedForward import FeedForward
from Layers.MoEFeedForward import MoEFeedForward
from Attention.MultiHeadAttention import MultiHeadAttention
import time 
from typing import Union, Tuple, Optional

class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["num_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
        )
        self.ff = MoEFeedForward(cfg) if cfg["num_experts"] > 0 else FeedForward(cfg)
        self.norm1 = RMSnorm(cfg["emb_dim"])
        self.norm2 = RMSnorm(cfg["emb_dim"])
        self.drop_shortcut = Dropout(cfg["drop_rate"])

        self.enable_profiling = cfg.get("enable_profiling", False)
        if self.enable_profiling:
            self.MOE_FF_MEM_BYTES = []
            self.MOE_FF_TIME_MS = []

    def forward(self, x: torch.Tensor, use_cache: bool=False, return_aux_loss: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)

        x = self.drop_shortcut(x)
        x = x + shortcut 

        shortcut = x
        x = self.norm2(x)

        if self.enable_profiling:
            x, aux_loss = self._forward_ff_with_profiling(x, return_aux_loss=return_aux_loss)
        else: 
            ff_output = self.ff(x, return_aux_loss=return_aux_loss)
            if isinstance(ff_output, tuple):
                x, aux_loss = ff_output
            else: 
                x = ff_output
                aux_loss = None 
            

        x = self.drop_shortcut(x)
        x = x + shortcut 

        if return_aux_loss and aux_loss is not None:
            return x, aux_loss
        return x 
    
    def _forward_ff_with_profiling(
        self,
        x: torch.Tensor, 
        return_aux_loss: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        use_cuda = torch.cuda.is_available()
        if use_cuda: 
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()

        start = time.perf_counter()
        ff_output = self.ff(x, return_aux_loss=return_aux_loss)
        
        if use_cuda: 
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() 
            self.MOE_FF_MEM_BYTES.append(peak_mem - base_mem)

        self.MOE_FF_TIME_MS.append((time.perf_counter() - start) * 1000)
        
        if isinstance(ff_output, tuple):
            return ff_output
        return ff_output, None 
    
    def get_profiling_stats(self) -> dict: 
        if not self.enable_profiling: 
            return {}
        
        if not self.MOE_FF_MEM_BYTES or not self.MOE_FF_TIME_MS:
            return {
                "memory_bytes": [],
                "time_ms": [],
                "avg_memory_mb": 0,
                "avg_time_ms": 0,
                "max_memory_mb": 0,
                "max_time_ms": 0,
                "min_memory_mb": 0,
                "min_time_ms": 0,
            }
        
        return {
            "memory_bytes": self.MOE_FF_MEM_BYTES,
            "time_ms": self.MOE_FF_TIME_MS,
            "avg_memory_mb": sum(self.MOE_FF_MEM_BYTES) / len(self.MOE_FF_MEM_BYTES) / 1e6,
            "avg_time_ms": sum(self.MOE_FF_TIME_MS) / len(self.MOE_FF_TIME_MS),
            "max_memory_mb": max(self.MOE_FF_MEM_BYTES) / 1e6,
            "max_time_ms": max(self.MOE_FF_TIME_MS),
            "min_memory_mb": min(self.MOE_FF_MEM_BYTES) / 1e6,
            "min_time_ms": min(self.MOE_FF_TIME_MS),
        }

    
    def reset_profiling_stats(self):
        """Reset profiling statistics."""
        if self.enable_profiling:
            self.MOE_FF_MEM_BYTES.clear()
            self.MOE_FF_TIME_MS.clear()
