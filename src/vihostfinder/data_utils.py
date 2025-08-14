import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor



class HostDataset(Dataset):
    allhosts = ["Bacteria","Plant","Protist","Fungi","Spiralia","Ecdysozoa",
                "Fish","Reptile","Amphibia","Bird","Mammal","Cnidaria", "Human", 
                "Protostomia","Vertebrate"]
    mainhosts = ["Bacteria","Plant","Protist","Fungi", "Cnidaria", "Protostomia",
                "Vertebrate"]

    def __init__(self, annotations_file, file_dir, partition=False, transform=False,
                    labels="main"):

        self.data = pd.read_csv(annotations_file, sep="\t")
        self.data["Human"] = self.data["HumanHost"].astype(int)

        if partition:
            self.data = self.data[self.data["partition"].isin(partition)]
        self.file_dir = file_dir
        self.transform = transform
        if labels == "main":
            self.labels = HostDataset.mainhosts
        else:
            self.labels = HostDataset.allhosts
        self.weights = self.calculate_weights()


    def calculate_weights(self):
        labels = self.data[self.labels].to_numpy().astype(int)
        pos_num = labels.sum(axis=0)
        neg_num = len(labels) - pos_num
        weights = neg_num/pos_num
        return torch.from_numpy(weights)

    def __len__(self):
        return len(self.data)
    
    def load_vec(self,path):
        o = np.load(path)
        return o["data"]

    def __getitem__(self, idx):
        embed_path = os.path.join(self.file_dir, self.data.iloc[idx]["Files"])
        embedding = self.load_vec("{}.npz".format(embed_path))
        labels = torch.from_numpy(self.data.iloc[idx][self.labels].to_numpy().astype(int))
        if len(embedding.shape) == 2:
            embedding = np.squeeze(embedding)
        elif len(embedding.shape) == 1:
#            print(embedding.shape)
            pass
            #embedding = np.expand_dims(embedding, axis=0)
        else:
            if embedding.shape[1] == 1:
                embedding = np.squeeze(embedding,dim=0)
            else:
                embedding = np.mean(embedding, dim=1)
        if self.transform:
            embedding = self.transform(embedding)
        embedding = torch.from_numpy(embedding)
        return embedding, labels

if __name__ == "__main__":
    training_data = HostDataset(annotations_file="../../RNA_db/data/metadata/repr_dbredux_part.tsv",
                                file_dir="../../RNA_db/data/embeddingvector/embeddingvector_caduceus/",
                                partition=[1,2,3])

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    for t in train_dataloader:
        print(t[0].shape)
        print(t[1].shape)
        exit()

