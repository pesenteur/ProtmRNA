# ProtmRNA

ProtmRNA is a codon-level language model for mRNA sequence representation learning.

## Usage

The input sequence should be pre-tokenized by spaces. Each token should be a codon.

```bash
cd ProtmRNA

python infer.py \
    --sequence "ATG GCT TTT CGA GGA" \
    --output_path outputs/feature.npy
```

Example output:

```text
input sequence:
ATG GCT TTT CGA GGA

decoded sequence:
ATG GCT TTT CGA GGA

feature shape: (1, 5, 1280)

feature saved to:
/path/to/ProtmRNA/outputs/feature.npy
```

The saved feature is a NumPy array with shape:

```text
1 × L × 1280
```

where `L` is the number of codon tokens in the input sequence.

## Feature Extraction

The core logic for extracting features from a codon sequence is shown below:

```python
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

feature = decode_out[:, 1:-1, :]
```

## Dependency

    Python 3.7
    TensorFlow 2.4

The pre-trained model of ProtmRNA is hosted on Google Drive.

Please place the weight file in the following path before running feature extraction:

```text
ProtmRNA/models/ProtmRNA_weights.h5
```

