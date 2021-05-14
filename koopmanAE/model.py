from torch import nn
import torch

FORWARD = 'forward'
BACKWARD = 'backward'


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class EncoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(EncoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))        
        x = self.fc3(x)
        
        return x


class DecoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(DecoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.m, self.n)
        return x



class Dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(Dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class DynamicsBack(nn.Module):
    def __init__(self, b, omega):
        super(DynamicsBack, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.Dynamics.weight.data.t())

    def forward(self, x):
        x = self.dynamics(x)
        return x




class KoopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(KoopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = EncoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = Dynamics(b, init_scale)
        self.backdynamics = DynamicsBack(b, self.dynamics)
        self.decoder = DecoderNet(m, n, b, ALPHA = alpha)


    def forward(self, x, mode=FORWARD):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        if mode == FORWARD:
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == BACKWARD:
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
