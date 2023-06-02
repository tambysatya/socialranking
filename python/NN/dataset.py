from torch.utils.data import DataLoader, Dataset
import torch


class KPDataset (Dataset):
    def __init__(self, n, m, l, datalen, realP):
        self.adjmat = torch.load(f"adjmat-{n}_{m}_{l}_{datalen}.pt")
        if realP:
            print ("loading optimal targets")
            self.tgts = torch.load(f"real_tgts-{n}_{m}_{l}_{datalen}.pt")
        else:
            print ("loading lexcell targets")
            self.tgts = torch.load(f"lex_tgts-{n}_{m}_{l}_{datalen}.pt")
    def __len__(self):
        return self.adjmat.size(0)
    def __getitem__(self,idx):
        return self.adjmat[idx], self.tgts[idx]
    
    def append(self, n, m, l, datalen, realP, i):
        adjmat2 = torch.load(f"adjmat-{n}_{m}_{l}_{datalen}_{i}.pt")
        if realP:
            print ("loading optimal targets")
            tgts2 = torch.load(f"real_tgts-{n}_{m}_{l}_{datalen}_{i}.pt")
        else:
            print ("loading lexcell targets")
            tgts2 = torch.load(f"lex_tgts-{n}_{m}_{l}_{datalen}_{i}.pt")

        adjmat = torch.cat((self.adjmat, adjmat2),dim=0)
        tgts = torch.cat((self.tgts, tgts2),dim=0)

        self.adjmat= adjmat
        self.tgts=tgts





        

