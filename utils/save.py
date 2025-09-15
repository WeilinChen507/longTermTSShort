import pickle as pkl



def save_pkl(fileNmae, data):
    with open(fileNmae, "wb") as f:
        pkl.dump(data, f)


def read_pkl(filename):
    f = open(filename, 'rb')
    data = pkl.load(f)
    return data