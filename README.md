#### Unlike metas rotary this ties into your decoder block so it takes fewer resources and almost no effort to install
#### Usage : 

put this inside your textdecoder blovk before the forward pass:
self.rotary = rotary(dims=dims, heads=heads)

This in the forward:
x = self.rotary(x)

    
    class TextDecoder(nn.Module):
        def __init__(
            self, vocab: int, ctx: int, dims: int, heads: int, layer: int):
            super().__init__()
    ***     self.rotary = rotary(dims=dims, heads=heads)
            self.token_embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dims)
            self.positional_embedding = nn.Parameter(torch.empty(ctx, dims))
            self.block = nn.ModuleList(modules=[Residual(dims=dims, heads=heads) for _ in range(layer)]) if layer > 0 else None
            self.ln = LayerNorm(normalized_shape=dims)
      
        def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):      
            offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
            x = (self.token_embedding(x)
                + self.positional_embedding[offset : offset + x.shape[-1]])
            
            self.start_pos = offset
            mask = None
            if self.ctx > 1:
                mask = torch.full(size=(self.ctx, self.ctx), fill_value=float("-inf"), device=x.device)
                mask = torch.triu(input=mask, diagonal=1)
                mask = torch.hstack(tensors=[torch.zeros(size=(self.ctx, self.start_pos), device=x.device), mask]).type_as(other=x)
    
      ***   x = self.rotary(x)
       
            x = x.to(xa.dtype)
    
            if self.block is not None:
                for block in self.block:
                    x = block(x, xa, mask=mask, kv_cache=kv_cache)
            print(f"TextDecoder block output shape: {x.shape}") if Decoderdebug else None
    
            x = self.ln(x)
            logits = ( x @ torch.transpose(self.token_embedding.weight.to(dtype=x.dtype), dim0=0, dim1=1)).float()
            print(f"TextDecoder logits shape: {logits.shape}") if Decoderdebug else None
    
            return logits
    
    import torch
    import torch.nn as nn
            
    class rotary(nn.Module):
        def __init__(self, dims, heads):
            super(rotary, self).__init__()
            self.base = 10000
            self.dims = dims
            self.heads = heads
    
            self.h_dim = self.dims // self.heads
            self.rot = (self.dims // self.heads) // 2
    
            self.theta = nn.Parameter(torch.zeros(1), requires_grad=True)
            
            freq_data = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
            self.register_buffer('inv_freq', freq_data)
    
            self.rotation_function = self.apply_rotation
            self.rotation_matrices = None  # Initialize rotation_matrices
    
        def q_rotation(self, x, theta, u, v):
            u = u / torch.norm(u)
            v = v / torch.norm(v)
            cos_theta = torch.cos(theta / 2)
            sin_theta = torch.sin(theta / 2)
    
            q = torch.empty(4, device=x.device)
            q[0] = cos_theta
            q[1] = sin_theta * u[0]
            q[2] = sin_theta * u[1]
            q[3] = sin_theta * u[2]
    
            q_conjugate = torch.empty(4, device=x.device)
            q_conjugate[0] = cos_theta
            q_conjugate[1] = -sin_theta * u[0]
            q_conjugate[2] = -sin_theta * u[1]
            q_conjugate[3] = -sin_theta * u[2]
    
            x_shape = x.shape
            x = x.view(-1, 3)
    
            uv_cross = torch.cross(u.unsqueeze(0), x)
            uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
            x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)
            
            x_rot = x_rot.view(*x_shape)
            return x_rot
    
        def rotation_matrix(self, dims, i, j, theta):
            G = torch.eye(dims).to(theta.device)
            G[i, i] = torch.cos(theta)
            G[i, j] = -torch.sin(theta)
            G[j, i] = torch.sin(theta)
            G[j, j] = torch.cos(theta)
    
            u = torch.eye(dims).to(theta.device)[i]
            v = torch.eye(dims).to(theta.device)[j]
    
            if dims == 3:
                Q = self.q_rotation(x=torch.eye(dims).to(theta.device), theta=theta, u=u, v=v)
                return (G + Q) / 2
            return G
    
        def _precompute_rotation_matrices(self):
    
            self.rotation_matrices = []
            for _ in range(self.rot):
                i, j = torch.randint(0, self.h_dim, (2,)).long()
                theta = self.theta
                B = self.rotation_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
                self.rotation_matrices.append(B)
    
        def apply_rotation(self, x):
    
            if self.rotation_matrices is None:
                self._precompute_rotation_matrices()
    
            for B in self.rotation_matrices: # type: ignore
                x = torch.matmul(x, B)
            return x
    
        def forward(self, x):
            if x.dim() not in [3, 4]:
                raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
    
            batch_size, seq_len, *rest = x.size()
    
            if x.dim() == 3:
                dims = rest[0]
                if dims != self.heads * self.h_dim:
                    raise ValueError(
                        f"Expected d.dims ({dims}) to be compatible with heads ({self.heads}) * h_dim ({self.h_dim}={self.heads * self.h_dim})")
            else:
                heads, h_dim = rest
                if heads != self.heads or h_dim != self.h_dim:
                    raise ValueError(
                        f"For 4D input, expected heads {self.heads} and h_dim {self.h_dim}, but got heads {heads} and h_dim {h_dim}")
    
            x = x.view(batch_size, seq_len, self.heads, self.h_dim)
            x = x.reshape(-1, self.h_dim)
            x = self.rotation_function(x)
            x = x.view(batch_size, seq_len, self.heads, self.h_dim)
            
            sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(seq_len, device=x.device), self.inv_freq)
            sin = sinusoid_inp.sin().unsqueeze(0).unsqueeze(2)
            cos = sinusoid_inp.cos().unsqueeze(0).unsqueeze(2)
            
            x1, x2 = x[..., ::2], x[..., 1::2]
            x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            x = x.view(batch_size, seq_len, self.dims)
            
            return x
