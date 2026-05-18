import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from model import ProtmRNA
from utils import dict_res_to_idx, dict_idx_to_res


def extract_feature_and_decode_from_sequence(my_model, sequence):
    seq = sequence.split()

    x = [dict_res_to_idx['<cls>']]
    for i in seq:
        if i in dict_res_to_idx:
            x.append(dict_res_to_idx[i])
        elif len(i) == 3:
            x.append(dict_res_to_idx['XXX'])
        else:
            assert len(i) == 1
            x.append(dict_res_to_idx['X'])
    x.append(dict_res_to_idx['<eos>'])

    x = np.array([x])
    out, decode_out = my_model(x)

    assert out.shape == (1, len(seq) + 2)
    assert decode_out.shape == (1, len(seq) + 2, 1280)

    out_np = out.numpy()[0][1:-1]
    decode_result = [dict_idx_to_res[int(i)] for i in out_np]

    decode_out = decode_out[:, 1:-1, :]

    return decode_out, decode_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="feature.npy"
    )
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    my_model = ProtmRNA(n_layers=33,
                        d_model=1280,
                        n_heads=20,
                        d_ffn=1280*4,
                        vocab_size=78)

    my_model(inputs=np.zeros((1, 1024)))
    my_model.load_model(name="./models/ProtmRNA_weights.h5")

    feature, decode_result = extract_feature_and_decode_from_sequence(
        my_model,
        args.sequence
    )

    feature_np = feature.numpy()
    output_path = os.path.abspath(args.output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, feature_np)

    print("input sequence:")
    print(args.sequence)
    print()
    print("decoded sequence:")
    print(" ".join(decode_result))
    print()
    print("feature shape:", feature_np.shape)
    print()
    print("feature saved to:")
    print(output_path)


if __name__ == "__main__":
    main()