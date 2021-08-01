import torch
import torch.nn.functional as F


def clr_loss(emb1, emb2, temperature=1):
    batch_size = emb1.shape[0]
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    emb = torch.cat((emb1, emb2), dim=0)
    sim = torch.exp(torch.mm(emb, torch.transpose(emb, 0, 1)) / temperature)

    # sim = sim.masked_fill_(torch.eye(sim.shape[0]).bool(), 0.)
    sim = sim * (1 - torch.eye(2 * batch_size, 2 * batch_size))
    print(sim)
    print(sim.shape)

    pos = torch.mul(emb1, emb2).sum(dim=1)
    pos = torch.exp(torch.cat((pos, pos), dim=0))
    print("pos: ")
    print(pos)

    neg = torch.sum(sim, dim=1)
    print("neg: ")
    print(neg)

    print(torch.div(pos, neg))
    loss_val = -(torch.sum(torch.log(torch.div(pos, neg)), dim=-1)) / (2 * batch_size)
    print(loss_val)


def nt_xnet_loss(out1, out2, temperature=1):
    out1 = F.normalize(out1, dim=1)
    out2 = F.normalize(out2, dim=1)
    out = torch.cat((out1, out2), dim=0)
    batch_size = 2 * out1.shape[0]
    print(len(out))
    cov = torch.mm(out, out.t()).contiguous()
    sim = torch.exp(cov / temperature)
    # sim = (cov / temperature)
    print(sim)
    mask = ~torch.eye(batch_size).bool()
    neg = sim.masked_select(mask)
    print("neg:==>")
    print(neg)
    print("neg_final==>")
    neg = neg.view(batch_size, -1)
    print(neg)
    neg = neg.sum(dim=-1)

    pos = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    # pos = (torch.sum(out1 * out2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    print("pos: ")
    print(pos)

    print("neg: ")
    print(neg)

    loss = -torch.log(pos / neg).mean()

    print(loss)


emb1 = torch.tensor([[1.0, 2.0], [3.0, -2.0], [1.0, 5.0]])
emb2 = torch.tensor([[1.0, 0.75], [2.8, -1.75], [1.0, 4.7]])

clr_loss(emb1, emb2)
print("---------------")
nt_xnet_loss(emb1, emb2)
