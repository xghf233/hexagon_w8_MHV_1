"""Synthetic schema fixtures need no WXF; optional real-data audit runs on server."""

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from projects.heptagon_symbol.alphabet import ALPHABET_VERSION, LETTERS
from projects.heptagon_symbol.dataset import HeptagonDataset, HeptagonDataLoader
from projects.heptagon_symbol.tokenizer import decode_word, encode_prompt


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def refresh_hashes(root):
    meta = json.loads((root / 'metadata.json').read_text())
    meta['files'] = {
        name: {'bytes': (root / name).stat().st_size,
               'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()}
        for name in ('words.npy', 'coefficients.npy', 'splits.npz', 'audit.json')
    }
    write_json(root / 'metadata.json', meta)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    words = np.zeros((10, 6), dtype=np.uint8)
    words[:, -1] = np.arange(10)
    np.save(root / 'words.npy', words, allow_pickle=False)
    np.save(root / 'coefficients.npy', np.array([-48, 48, -1, 1, -2, 2, 3, -3, 4, -4], dtype='<i2'))
    order = np.random.Generator(np.random.PCG64(42)).permutation(10).astype('<i4')
    np.savez(root / 'splits.npz', train=order[:8], val=order[8:9], test=order[9:])
    write_json(root / 'audit.json', {'status': 'passed', 'n_samples': 10, 'unique_words': 10})
    write_json(root / 'metadata.json', {
        'schema_version': 1, 'dataset_id': 'synthetic-schema-fixture', 'validation_status': 'passed',
        'weight': 6, 'loop_order': 3, 'sector': 'MHV', 'nonzero_only': True,
        'source': {'path': '/nonexistent/mac/source.wxf'},
        'alphabet': {'version': ALPHABET_VERSION, 'id_to_letter': list(LETTERS)},
        'arrays': {'words': {'file': 'words.npy', 'dtype': 'uint8', 'shape': [10, 6]},
                   'coefficients': {'file': 'coefficients.npy', 'dtype': '<i2', 'shape': [10]}},
        'split': {'file': 'splits.npz', 'dtype': '<i4', 'split_type': 'random_row',
                  'orbit_grouped': False, 'seed': 42, 'slice_order': ['train', 'val', 'test'],
                  'generator': 'numpy.random.Generator(PCG64)',
                  'counts': {'train': 8, 'val': 1, 'test': 1}},
    })
    refresh_hashes(root)
    return root


def test_saved_split_and_source_independence(data_dir):
    train = HeptagonDataset(data_dir)
    val = HeptagonDataset(data_dir, 'val')
    test = HeptagonDataset(data_dir, 'test')
    assert [len(train), len(val), len(test)] == [8, 1, 1]
    assert not train.words.flags.writeable
    assert not train.indices.flags.writeable
    with np.load(data_dir / 'splits.npz') as stored:
        assert np.array_equal(train.indices, stored['train'])
    ids, coeff = train[0]
    row = train.indices[0]
    assert ids.tolist() == train.words[row].tolist()
    assert coeff == train.coefficients[row]
    ids[:] = 41  # returned samples cannot mutate mmap
    assert train[0][0].tolist() != ids.tolist()
    assert set(train.indices).isdisjoint(val.indices)
    with pytest.raises(IndexError):
        train[-1]
    with pytest.raises(ValueError):
        HeptagonDataLoader(val, batch_size=1)


def test_relocation_and_manifest_anchor(data_dir, tmp_path):
    original = HeptagonDataset(data_dir)
    moved = tmp_path / 'server-copy'
    shutil.copytree(data_dir, moved)
    copy = HeptagonDataset(moved, expected_metadata_sha256=original.metadata_sha256)
    assert copy.identity() == original.identity()
    with pytest.raises(ValueError, match='Metadata SHA'):
        HeptagonDataset(moved, expected_metadata_sha256='0' * 64)


def test_corrupt_file_hash(data_dir):
    path = data_dir / 'words.npy'
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match='hash mismatch'):
        HeptagonDataset(data_dir)


@pytest.mark.parametrize('mutation', ['duplicate', 'letter', 'zero', 'dtype', 'overlap', 'out_of_range', 'shape', 'alphabet', 'orbit', 'encoding'])
def test_semantic_rejection_even_with_updated_hashes(data_dir, mutation):
    if mutation in ('duplicate', 'letter', 'shape'):
        words = np.load(data_dir / 'words.npy')
        if mutation == 'duplicate':
            words[1] = words[0]
        elif mutation == 'letter':
            words[0, 0] = 42
        else:
            words = words[:, :5]
        np.save(data_dir / 'words.npy', words)
    elif mutation in ('zero', 'dtype'):
        coeff = np.load(data_dir / 'coefficients.npy')
        if mutation == 'zero':
            coeff[0] = 0
        else:
            coeff = coeff.astype(np.float32)
        np.save(data_dir / 'coefficients.npy', coeff)
    elif mutation in ('overlap', 'out_of_range'):
        with np.load(data_dir / 'splits.npz') as archive:
            splits = {key: archive[key] for key in archive.files}
        splits['test'][0] = splits['train'][0] if mutation == 'overlap' else 10
        np.savez(data_dir / 'splits.npz', **splits)
    else:
        meta = json.loads((data_dir / 'metadata.json').read_text())
        if mutation == 'alphabet':
            meta['alphabet']['id_to_letter'].reverse()
        elif mutation == 'orbit':
            meta['split']['orbit_grouped'] = True
        else:
            meta['training_encoding_proposal'] = {'vocab_size': 1012}
        write_json(data_dir / 'metadata.json', meta)
    refresh_hashes(data_dir)
    with pytest.raises(ValueError):
        HeptagonDataset(data_dir)


def test_batches_cross_epoch_without_dropping_tail(data_dir):
    dataset = HeptagonDataset(data_dir)
    loader = HeptagonDataLoader(dataset, batch_size=6, shuffle=False)
    seen = []
    for _ in range(3):
        batch = next(loader)
        assert batch['idx'].shape == (6, 15)
        assert (batch['targets'] != -1).sum().item() == 18
        seen.extend(batch['idx'][:, 6].tolist())
    expected = [dataset[i % len(dataset)][0][-1] for i in range(18)]
    assert seen == expected
    assert batch['state_dict']['rows_consumed'] == 18
    assert loader.get_state() == batch['state_dict']


def test_shuffle_once_per_epoch_with_no_split_changes(data_dir):
    dataset = HeptagonDataset(data_dir)
    loader = HeptagonDataLoader(dataset, batch_size=len(dataset), seed=123)
    for epoch in range(3):
        batch = next(loader)
        permutation = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([123, epoch]))).permutation(len(dataset))
        assert batch['idx'][:, 6].tolist() == [int(dataset[i][0][-1]) for i in permutation]
        assert sorted(batch['idx'][:, 6].tolist()) == sorted(dataset.words[dataset.indices, -1].tolist())


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('steps', [0, 1, 4, 5])
def test_resume_exactly_from_next_unread_row(data_dir, shuffle, steps):
    dataset = HeptagonDataset(data_dir)
    original = HeptagonDataLoader(dataset, batch_size=6, seed=77, shuffle=shuffle)
    for _ in range(steps):
        next(original)
    # Checkpoint metadata is JSON, not a Python RNG object or tensor.
    state = json.loads(json.dumps(original.state_dict()))
    resumed = HeptagonDataLoader(dataset, batch_size=6, seed=77, shuffle=shuffle)
    resumed.set_state(state)
    for _ in range(5):
        left, right = next(original), next(resumed)
        for key in left:
            if key == 'state_dict':
                assert left[key] == right[key]
            else:
                assert torch.equal(left[key], right[key])


def test_resume_contract_rejection(data_dir):
    dataset = HeptagonDataset(data_dir)
    state = HeptagonDataLoader(dataset, batch_size=3).state_dict()
    for kwargs in ({'batch_size': 4}, {'batch_size': 3, 'seed': 99},
                   {'batch_size': 3, 'sequence_len': 17}, {'batch_size': 3, 'shuffle': False}):
        with pytest.raises(ValueError, match='contract mismatch'):
            HeptagonDataLoader(dataset, **kwargs).set_state(state)
    state['rows_consumed'] = 2
    with pytest.raises(ValueError, match='batch-boundary'):
        HeptagonDataLoader(dataset, batch_size=3).set_state(state)


def test_reject_distributed_launcher(data_dir, monkeypatch):
    dataset = HeptagonDataset(data_dir)
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='distributed'):
        HeptagonDataLoader(dataset, batch_size=2)


def test_optional_real_converted_dataset():
    location = os.environ.get('HEPTAGON_W6_DATA_DIR')
    if not location:
        pytest.skip('Set HEPTAGON_W6_DATA_DIR on server to audit the converted dataset')
    dataset = HeptagonDataset(Path(location))
    assert dataset.words.shape == (467250, 6)
    assert len(dataset) == 373800
    assert len(np.unique(dataset.coefficients)) == 33
    assert len(np.unique(dataset.coefficients[dataset.indices])) == 33
    assert decode_word(dataset.words[0]) == ('a11',) * 5 + ('a21',)
    assert int(dataset.coefficients[0]) == -48
    batch = next(HeptagonDataLoader(dataset, batch_size=4, shuffle=False))
    for i in range(4):
        ids, coeff = dataset[i]
        assert batch['idx'][i, :8].tolist() == encode_prompt(ids)
        assert batch['targets'][i, 8].item() == 48 + abs(coeff)
