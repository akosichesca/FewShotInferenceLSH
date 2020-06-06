import torch
import torchvision
import tqdm
import random
import numpy
import LSH

def binlist2int(exp,mylist):
    return [ exp**x for x in mylist ]

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def get_values(values, k, s):
    v = [i for i,x in enumerate(values) if x == k]
    return random.sample(v,s)

def main(opt):

    dataset = torchvision.datasets.Omniglot(
        root="./data", download=True, background=False,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([28,28]),
                torchvision.transforms.ToTensor(),
            ])
    )


    model = torch.load(opt['model.model_path'], encoding='utf-8')
    model.eval()

    nlen = [int(0.8*len(dataset)),int(0.2*len(dataset))]
    trainset, testset = torch.utils.data.random_split(dataset, nlen)

    dataset = 0

    train_values = []
    for d in tqdm.tqdm(trainset):
        train_values.append(d[1])
    test_values = []
    for d in tqdm.tqdm(testset):
        test_values.append(d[1])
    n_way = opt['data.test_way'] #50
    n_shot = opt['data.test_shot']

    acc = 0
    itr = 10000
    lsh = LSH.LSH(64,opt['dist.qbits'],opt['memsize'])
    for it in tqdm.tqdm(range(itr)):
        k = random.sample(train_values, n_way)
        q = random.sample(k, 1)
        while not (q[0] in test_values):
            q = random.sample(k, 1)
        support = []
        support_val = []
        for i in k:
            s = get_values(train_values, i, n_shot)
            for j in s:
                x = model.encoder.forward(1-trainset[j][0][-1,:,:].reshape([1,1,28,28]))
                lsh.append(x,i)
        s = get_values(test_values, q[0], 1)
        x = model.encoder.forward(1-testset[s[0]][0][-1,:,:].reshape([1,1,28,28]))

        y_s = lsh.search(x)
        if y_s == q[0]:
            acc = acc + 1

    print("Accuracy : ",  acc*100/(it+1))
