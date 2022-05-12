import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from typing import List, Dict

# Seq2Seq 모델이 기존에 없던 class를 예측하면 사용할 index
UNK_INDEX = 99999


def get_ordinal_encoder(full_data):
    df = pd.read_csv(full_data)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value',
                        unknown_value=UNK_INDEX, dtype=np.int_)
    oe.fit(df['진단코드'].to_numpy().reshape(-1, 1))
    return oe


def get_label_encoder(full_data):
    df = pd.read_csv(full_data)
    le = LabelEncoder()
    return le.fit(df['진단코드'])


def tuple_to_dict(scores):
    return {
        'precision': scores[0],
        'recall': scores[1],
        'f1': scores[2]
    }


def calc_metrics(labels, preds):
    macro = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return tuple_to_dict(macro), acc


def get_vocab_restrict_fn(tokenizer):
    a_to_z = ' '.join([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    zero_to_nine = ' '.join([str(i) for i in range(10)])
    zero_to_nine += '</s>'

    # 생성 첫 번째 Step에 가능한 Token
    a_to_z = tokenizer.encode(a_to_z)
    # 생성 두 번째 Step에 가능한 Token
    zero_to_nine = tokenizer.encode(zero_to_nine)

    def restrict_vocab(batch_idx, prefix_beam):
        if len(prefix_beam) < 2:
            return a_to_z
        return zero_to_nine

    return restrict_vocab


def get_trie_restrict_fn(tokenizer, train_path):
    labels = pd.read_csv(train_path)['진단코드'].unique()
    for i, label in enumerate(labels):
        full_label = '<usr>' + label + '</s>'
        labels[i] = full_label
    label_ids = tokenizer(labels.tolist())['input_ids']
    trie = Trie(label_ids)
    print(f"{trie.get([2])=}")

    def restrict_vocab(batch_idx, prefix_beam):
        # print(f'{prefix_beam=}')
        # print(f'{trie.get(prefix_beam)=}')
        return trie.get(prefix_beam.tolist())

    return restrict_vocab


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)
