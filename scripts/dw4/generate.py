import numpy as onp
import os

def run(args):
    from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean
    data_train, batch_iter_train = get_data_dw4(args.n_data, 'train', 100)
    data_val, batch_iter_val = get_data_dw4(args.n_data, 'val', 100)
    data_test, batch_iter_test = get_data_dw4(args.n_data, 'test', 100)

    data_train = data_train.reshape(-1, 4, 2)
    data_val = data_val.reshape(-1, 4, 2)
    data_test = data_test.reshape(-1, 4, 2)

    data_train = data_train - data_train.mean(dim=-2, keepdim=True)
    data_val = data_val - data_val.mean(dim=-2, keepdim=True)
    data_test = data_test - data_test.mean(dim=-2, keepdim=True)

    data_train = onp.array(data_train)
    data_val = onp.array(data_val)
    data_test = onp.array(data_test)

    onp.save("data_train.npy", data_train)
    onp.save("data_val.npy", data_val)
    onp.save("data_test.npy", data_test)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", type=int, default=100)
    args = parser.parse_args()
    run(args)