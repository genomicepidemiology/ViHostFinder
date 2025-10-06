import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Sampler
import random




class HostDataset(Dataset):

    orig_hier1_hosts = {"Bacteria":0, "Plant":1, "Animalia":2, "Protist":3, "Cnidaria":4, "Fungi":5,
                        "Vertebrate":6, "Invertebrate":7,
                        "Amphibia": 8, "Fish":9, "Mammal":10, "Bird":11, "Reptile":12,
                        "Ecdysozoa": 13, "Spiralia": 14, "Human":15}
    
    orig_hier1_hier = [{"Leave":"Human", "Parent":"Mammal"}, {"Leave":"Vertebrate", "Parent":"Mammal"},
                 {"Leave":"Vertebrate", "Parent":"Amphibia"}, {"Leave":"Vertebrate", "Parent":"Fish"},
                 {"Leave":"Vertebrate", "Parent":"Bird"}, {"Leave":"Vertebrate", "Parent":"Reptile"},
                 {"Leave":"Invertebrate", "Parent":"Ecdysozoa"}, {"Leave":"Invertebrate", "Parent":"Spiralia"}]

    orig_all_hosts = {"Bacteria":0, "Plant":1, "Protist":3, "Cnidaria":4, "Fungi":5,
                        "Amphibia": 8, "Fish":9, "Mammal":10, "Bird":11, "Reptile":12,
                        "Ecdysozoa": 13, "Spiralia": 14, "Human":15}


    def __init__(self, annotations_file, file_dir, labels, partition=False, transform=False):

        self.data = pd.read_csv(annotations_file, sep="\t")
        self.data["Human"] = self.data["HumanHost"].astype(int)
        if partition:
            self.data = self.data[self.data["partition"].isin(partition)]
        self.file_dir = file_dir
        self.transform = transform
        self.labels = labels
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
            pass
        else:
            if embedding.shape[1] == 1:
                embedding = np.squeeze(embedding,dim=0)
            else:
                embedding = np.mean(embedding, dim=1)
        if self.transform:
            embedding = self.transform(embedding)
        embedding = torch.from_numpy(embedding)
        return embedding, labels

    @staticmethod
    def create_tree(labels="orig_flat"):
        parent_nums = []
        if labels == "orig_flat":
            pass
        else:
            if labels == "orig_hier1":
                hier_cols = HostDataset.orig_hier1_hier
                label_rows = HostDataset.orig_hier1_hosts
            elif labels == "orig_hier2":
                pass
            else:
                raise ValueError("MUUU")
            parent_nums = []
            for col in hier_cols:
                parent_nums.append({"Leave": label_rows[col["Leave"]], 
                                    "Parent": label_rows[col["Parent"]]})
        return parent_nums    




class ClusterSampler(Sampler):
    def __init__(self, dataset, cluster_column, f=1.0):
        """
        Args:
            dataset (HostDataset): The dataset object with a .data DataFrame.
            cluster_column (str): Column name in dataset.data to use for clustering.
            f (float): Frequency parameter between 0 and 1.
        """
        assert 0 <= f <= 1, "f must be between 0 and 1"
        self.dataset = dataset
        self.cluster_column = cluster_column
        self.f = f
        self.clusters = self._build_clusters()

    def _build_clusters(self):
        clusters = {}
        for idx in range(len(self.dataset.data)):
            cluster_id = self.dataset.data.iloc[idx][self.cluster_column]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(idx)
        return clusters

    def _generate_indices(self):
        selected = []
        for indices in self.clusters.values():
            n = len(indices)
            if self.f == 0:
                selected.append(random.choice(indices))
            elif self.f == 1:
                selected.extend(indices)
            else:
                k = max(1, int(self.f * n))
                selected.extend(random.sample(indices, k))  # no replacement
        random.shuffle(selected)
        return selected

    def __iter__(self):
        return iter(self._generate_indices())

    def __len__(self):
        return sum(
            1 if self.f == 0 else len(indices) if self.f == 1 else max(1, int(self.f * len(indices)))
            for indices in self.clusters.values()
        )




if __name__ == "__main__":
    training_data = HostDataset(annotations_file="../../RNA_db/data/metadata/repr_dbredux_part.tsv",
                                file_dir="../../RNA_db/data/embeddingvector/embeddingvector_caduceus/",
                                partition=[1,2,3])

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    for t in train_dataloader:
        print(t[0].shape)
        print(t[1].shape)
        exit()

