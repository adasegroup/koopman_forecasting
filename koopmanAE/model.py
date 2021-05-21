import pytorch_lightning as pl
import torch
from torch import nn


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA=1):
        super(encoderNet, self).__init__()
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


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

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



class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x


class koopmanAE(pl.LightningModule):
    def __init__(self, m, n, b, steps, steps_back, alpha=1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.encoder = encoderNet(m, n, b, ALPHA=alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA=alpha)

    # def training_step(self, train_loader):
    #     out, out_back = model(data_list[0].to(device), mode='forward')
    #
    #     for k in range(steps):
    #         if k == 0:
    #             loss_fwd = criterion(out[k], data_list[k + 1].to(device))
    #         else:
    #             loss_fwd += criterion(out[k], data_list[k + 1].to(device))
    #
    #     loss_identity = criterion(out[-1], data_list[0].to(device)) * steps
    #
    #     loss_bwd = 0.0
    #     loss_consist = 0.0
    #
    #     if backward == 1:
    #         out, out_back = model(data_list[-1].to(device), mode='backward')
    #
    #         for k in range(steps_back):
    #
    #             if k == 0:
    #                 loss_bwd = criterion(out_back[k], data_list[::-1][k + 1].to(device))
    #             else:
    #                 loss_bwd += criterion(out_back[k], data_list[::-1][k + 1].to(device))
    #
    #         A = model.dynamics.dynamics.weight
    #         B = model.backdynamics.dynamics.weight
    #
    #         K = A.shape[-1]
    #
    #         for k in range(1, K + 1):
    #             As1 = A[:, :k]
    #             Bs1 = B[:k, :]
    #             As2 = A[:k, :]
    #             Bs2 = B[:, :k]
    #
    #             Ik = torch.eye(k).float().to(device)
    #
    #             if k == 1:
    #                 loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik) ** 2) + \
    #                                 torch.sum((torch.mm(As2, Bs2) - Ik) ** 2)) / (2.0 * k)
    #             else:
    #                 loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik) ** 2) + \
    #                                  torch.sum((torch.mm(As2, Bs2) - Ik) ** 2)) / (2.0 * k)
    #
    #     #                Ik = torch.eye(K).float().to(device)
    #     #                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
    #     #                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
    #     #
    #     loss = loss_fwd + lamb * loss_identity + nu * loss_bwd + eta * loss_consist
    #     return loss
    # def configure_optimizers(self):
    #      optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay)
    #
    #      def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
    #          """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
    #          if epoch in decayEpoch:
    #              for param_group in optimizer.param_groups:
    #                  param_group['lr'] *= lr_decay_rate
    #              return optimizer
    #          else:
    #              return optimizer
    #      return optimizer

    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
