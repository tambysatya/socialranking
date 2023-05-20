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




        

