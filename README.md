    import torch
    import torch.nn as nn
    
    class rotary(nn.Module):
        def __init__(self, base, n_state, n_head, rotation_type='mixed_tape', theta_learnable=False):
            super(rotary, self).__init__()
            self.base = base
            self.n_state = n_state
            self.n_head = n_head
            self.rotation_type = rotation_type
    
            self.h_dim = self.n_state // self.n_head
            self.rot = (self.n_state // self.n_head) // 2
    
            self.theta = nn.Parameter(torch.zeros(1), requires_grad=theta_learnable)
            
            freq_data = 1.0 / (self.base ** (torch.arange(0, self.h_dim, 2).float() / self.h_dim))
            self.register_buffer('inv_freq', freq_data)
    
            self.rotation_function = self.apply_blended_rotation
            self.rotation_matrices = None  # Initialize rotation_matrices
    
        def quaternion_rotation(self, x, theta, u, v):
            u = u / torch.norm(u)
            v = v / torch.norm(v)
            cos_theta = torch.cos(theta / 2)
            sin_theta = torch.sin(theta / 2)
    
            # Use existing tensor instead of creating new ones
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
    
        def blended_rotation_matrix(self, dims, i, j, theta):
            G = torch.eye(dims).to(theta.device)
            G[i, i] = torch.cos(theta)
            G[i, j] = -torch.sin(theta)
            G[j, i] = torch.sin(theta)
            G[j, j] = torch.cos(theta)
    
            u = torch.eye(dims).to(theta.device)[i]
            v = torch.eye(dims).to(theta.device)[j]
    
            if dims == 3:
                Q = self.quaternion_rotation(x=torch.eye(dims).to(theta.device), theta=theta, u=u, v=v)
                return (G + Q) / 2
            return G
    
        def _precompute_rotation_matrices(self):
            # Precompute rotation matrices
            self.rotation_matrices = []
            for _ in range(self.rot):
                i, j = torch.randint(0, self.h_dim, (2,)).long()
                theta = self.theta
                B = self.blended_rotation_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
                self.rotation_matrices.append(B)
    
        def apply_blended_rotation(self, x):
            # Apply precomputed rotation matrices
            if self.rotation_matrices is None:
                self._precompute_rotation_matrices()
    
            for B in self.rotation_matrices:
                x = torch.matmul(x, B)
            return x
    
        def forward(self, x, global_step=None):
            if x.dim() not in [3, 4]:
                raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
    
            batch_size, seq_len, *rest = x.size()
    
            if x.dim() == 3:
                n_state = rest[0]
                if n_state != self.n_head * self.h_dim:
                    raise ValueError(
                        f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim}={self.n_head * self.h_dim})")
            else:
                n_head, h_dim = rest
                if n_head != self.n_head or h_dim != self.h_dim:
                    raise ValueError(
                        f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
    
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
            x = x.reshape(-1, self.h_dim)
            x = self.rotation_function(x)
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
            
            sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(seq_len, device=x.device), self.inv_freq)
            sin = sinusoid_inp.sin().unsqueeze(0).unsqueeze(2)
            cos = sinusoid_inp.cos().unsqueeze(0).unsqueeze(2)
            
            x1, x2 = x[..., ::2], x[..., 1::2]
            x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            x = x.view(batch_size, seq_len, self.n_state)
            
            return x
