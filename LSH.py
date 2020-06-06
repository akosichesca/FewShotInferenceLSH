import torch
import numpy as np

class LSH:
    def __init__(self,size,qbits=32,memsize=256,ttf=1):
        self.dim = size
        self.lsh_nHashes = 1
        self.lsh_nBuckets = 2**self.lsh_nHashes
        self.lsh_multiplier = torch.from_numpy(np.asarray(self.binlist2int(2,list(range(self.lsh_nHashes-1,-1,-1)))))
        self.lsh_nLibraries = size*qbits
        self.ttf = ttf
        self.memsize = memsize
        self.random_projections = [torch.FloatTensor(self.dim, self.lsh_nHashes).normal_(mean=0, std=2) for i in range(self.lsh_nLibraries)]
        self.lsh_bucket_loc = []
        self.key = [] #torch.zeros([64], dtype=torch.int)
        self.age = [] #torch.zeros([64], dtype=torch.int)
    def signature(self,x):
        return [torch.le(torch.matmul(x,self.random_projections[l]),0).long() for l in range(self.lsh_nLibraries)]
    def binlist2int(self,exp,mylist):
        return [ exp**x for x in mylist ]
    def append(self,x,y):
        signature = self.signature(x)
        if len(self.lsh_bucket_loc) < self.memsize:
            self.lsh_bucket_loc.append(torch.stack([torch.matmul(signature[l],self.lsh_multiplier.long()) for l in range(self.lsh_nLibraries)]))
            self.key.append(y)
            self.age.append(0)
        else:
            maxage = np.argmax(self.age)
            self.lsh_bucket_loc[maxage] = torch.stack([torch.matmul(signature[l],self.lsh_multiplier.long()) for l in range(self.lsh_nLibraries)])
            self.key[maxage] = y
            self.age[maxage] = 0

    def search(self,x):
        n = len(self.lsh_bucket_loc)
        signature = self.signature(x)
        sig_locs = torch.stack([torch.matmul(signature[l],self.lsh_multiplier.long()) for l in range(self.lsh_nLibraries)])
        dist = torch.zeros([n,1])
        for j in range(n):
            if self.age[j] < self.ttf: # time to forget (adjust depending on the application)
                sig_train = self.lsh_bucket_loc[j]
                dist[j] = (sig_locs-sig_train == 0).sum()
            else:
                dist[j] = 0
        maxdist = torch.argmax(dist)
        self.age[maxdist] -= 1
        k = self.key[maxdist]
        self.age = [x + 1 for x in self.age]
        #print(self.age)
        return k
