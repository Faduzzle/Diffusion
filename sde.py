import torch

class VPSDE():
    def __init__(self, bmin=0.1, bmax=20):
        self.bmin = bmin
        self.bmax = bmax

    def beta(self, t):
        return self.bmin + t * (self.bmax - self.bmin)

    def alpha(self, t):
        x = self.bmin * t + ((self.bmax - self.bmin) * t**2) / 2
        return torch.exp(-x / 2)

    def p(self, x, t):
        a = self.alpha(t).view(-1, 1, 1)
        mu = x * a
        std = torch.sqrt(1 - a**2)
        return mu, std

    def f(self, x, t):
        return -0.5 * self.beta(t).view(-1, 1, 1) * x

    def g(self, t):
        return torch.sqrt(self.beta(t)).view(-1, 1, 1)