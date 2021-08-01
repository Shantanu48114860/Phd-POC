import torch
import torch.nn as nn
import torch.nn.functional as F


# from .gather import GatherLayer

class NCELoss(nn.Module):
    def __init__(self, device, temperature=0.5, batch_size=64):
        super(NCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self.register_buffer(
            "mask",
            (1 - torch.eye(2 * batch_size, 2 * batch_size))
        )

    def forward(self, z_i, z_j):
        emb1 = F.normalize(z_i, dim=1)
        emb2 = F.normalize(z_j, dim=1)
        emb = torch.cat((emb1, emb2), dim=0)
        emb_T = torch.transpose(emb, 0, 1)
        sim_matrix = torch.exp(
            torch.mm(emb, emb_T) / self.temperature
        )
        # print(sim_matrix.shape)
        # print(self.mask.shape)
        mask = 1 - torch.eye(sim_matrix.shape[0], sim_matrix.shape[1])
        sim_matrix = sim_matrix * mask.to(self.device)
        # sim_matrix = sim_matrix.masked_fill_(self.mask, 0.)

        pos = torch.mul(emb1, emb2).sum(dim=1)
        pos = torch.exp(torch.cat((pos, pos), dim=0))
        neg = torch.sum(sim_matrix, dim=1)
        loss_val = -(
            torch.sum(torch.log(
                torch.div(pos, neg)
            ), dim=-1)
        ) / (2 * self.batch_size)

        return loss_val

# loss = NCELoss()

# emb1 = torch.tensor([[1.0, 2.0], [3.0, -2.0], [1.0, 5.0]])
# emb2 =torch.tensor([[1.0, 0.75], [2.8, -1.75], [1.0, 4.7]])
# print(loss(emb1, emb2))
