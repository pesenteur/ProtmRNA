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

input_sequence = "ATG GCT TTT CGA GGA"

my_model = ProtmRNA(n_layers=33,
                    d_model=1280,
                    n_heads=20,
                    d_ffn=1280*4,
                    vocab_size=78)

my_model(inputs=np.zeros((1, 1024)))
my_model.load_model(name="./models/ProtmRNA_weights.h5")

feature, decode_result = extract_feature_and_decode_from_sequence(
    my_model,
    input_sequence
)
```

## Dependency

    Python 3.7
    TensorFlow 2.4

The pre-trained model of ProtmRNA is hosted on Google Drive.

Please place the weight file in the following path before running feature extraction:

```text
ProtmRNA/models/ProtmRNA_weights.h5
```

