import pandas as pd
import numpy as np
import sys

#
# train = pd.read_csv("/home/zaraza/bigass2/big_data_hw_2/data/train_epin.csv", sep=",").values
# test = pd.read_csv("/home/zaraza/bigass2/big_data_hw_2/data/test_epin.csv", sep=",").values

# emb_in = pd.read_csv("/home/zaraza/bigass2/big_data_hw_2/emb_in.csv", sep=";", header=None).values
# emb_out = pd.read_csv("/home/zaraza/bigass2/big_data_hw_2/emb_out.csv", sep=";", header=None).values
def softmax(x):
    e_x = np.exp(x - np.max(x, axis = 0))
    return e_x / e_x.sum(axis=0)

def compute_map(train, test, emb_in, emb_out):
    uniq_train = np.unique(train[:, 0])
    result = []
    for i in uniq_train:
        index = test[test[:, 0] == i, 1]
        answer = []

        if index.shape[0] > 0:
            vector = softmax(np.dot(emb_out.T, emb_in[:, i]))
            #vector = np.dot(emb_out.T, emb_in[:, i])
            
            vector[train[train[:, 0] == i, 1]] = -np.inf

            for j in range(10):
                ind = np.argmax(vector)
                vector[ind] = -np.inf
                answer.append(ind)

            answer = np.array(answer)
            cnt = 0
            res = 0.
            for z, ans in enumerate(answer):
                if index[index == ans].shape[0] > 0:
                    cnt += 1
                    res += (cnt / (z + 1.))
            res /= min(10, index.shape[0])
            result.append(res)

        if len(result) % 100 == 0:
            print(np.array(result).mean())

    # print(np.array(result).mean())
    return np.array(result).mean()

"""
to run the function we need to specify 4 arguments:
1) path to train dataset
2) path to test dataset
3) path to embedding IN file
4) path to embedding OUT file
"""
def main():

    if len(sys.argv) == 5 :
        train = pd.read_csv(sys.argv[1], sep=",").values
        test = pd.read_csv(sys.argv[2], sep=",").values

        emb_in = pd.read_csv(sys.argv[3], sep=";", header=None).values
        emb_out = pd.read_csv(sys.argv[4], sep=";", header=None).values

        print(compute_map(train, test, emb_in, emb_out))
    else:
        print("Need more arguments! (train, test, embedding IN, embedding OUT)")

main()