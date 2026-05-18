import itertools

DNA_BASES = ['A', 'T', 'C', 'G']
STANDARD_DNA_CODONS = [''.join(c) for c in itertools.product(DNA_BASES, repeat=3)]

alphabet = [
    '<cls>', '<pad>', '<eos>', '<unk>', '<mask>'
] + STANDARD_DNA_CODONS + ['XXX', 'A', 'T', 'C', 'G', 'X', '0', '1', '2']

dict_res_to_idx = {}
dict_idx_to_res = {}
for idx, res in enumerate(alphabet):
    dict_res_to_idx[res] = idx
    dict_idx_to_res[idx] = res

