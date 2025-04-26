import torch

class VPSDE:
    def __init__(self, bmin=0.1, bmax=20.0):
        self.bmin = bmin
        self.bmax = bmax

    def beta(self, t):
        return self.bmin + t * (self.bmax - self.bmin)  # [B] or [B, pred_len]

    def alpha(self, t):
        int_beta = self.bmin * t + 0.5 * (self.bmax - self.bmin) * t ** 2
        return torch.exp(-0.5 * int_beta)  # [B, pred_len]

    def f(self, x, t):
        beta_t = self.beta(t).view(t.shape[0], -1, 1)  # [B, pred_len, 1]
        return -0.5 * beta_t * x

    def g(self, t):
        return torch.sqrt(self.beta(t)).view(t.shape[0], -1, 1)  # [B, pred_len, 1]

    def p(self, x, t):
        alpha_t = self.alpha(t).view(t.shape[0], -1, 1)          # [B, pred_len, 1]
        mu = alpha_t * x
        std = torch.sqrt(torch.clamp(1.0 - alpha_t ** 2, min=1e-5))
        return mu, std
