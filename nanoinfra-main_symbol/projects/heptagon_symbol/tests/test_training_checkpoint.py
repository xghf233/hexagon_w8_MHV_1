from copy import deepcopy
import json
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from projects.heptagon_symbol.checkpoint import capture_rng, restore_rng, run_contract, save, resume
from projects.heptagon_symbol.dataset import HeptagonDataset, HeptagonDataLoader
from projects.heptagon_symbol.evaluator import select_rows
from projects.heptagon_symbol.model import model_config
from projects.heptagon_symbol.train import validate_config, fresh_output
from .test_dataset import data_dir  # register the small synthetic fixture
from .test_model_evaluator import small_system


def config_named(name):
    directory = Path(__file__).resolve().parents[1] / 'configs'
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        cfg = compose(config_name=name, overrides=['data_dir=/server/data', 'output_dir=/server/new-run'])
        return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize('name', ['smoke', 'tiny_overfit', 'w6_random'])
def test_hydra_recipes(name):
    config = config_named(name)
    validate_config(config)
    assert model_config(config).vocab_size == 1048
    assert config['model'] == {'n_layer': 4, 'n_embd': 512, 'n_head': 8, 'n_kv_head': 8}


def test_exact_parameter_count_on_meta_device():
    from core.model.gpt import GPT
    from core.model.heads import LMHead
    from core.model.system import LMSystem
    config = model_config(config_named('w6_random'))
    with torch.device('meta'):
        system = LMSystem(GPT(config), LMHead(config.n_embd, config.vocab_size))
    assert sum(p.numel() for p in system.parameters()) == 13657600


def test_reject_incompatible_config():
    for key, value in [('max_steps', -1), ('total_batch_size', 8192), ('sequence_len', 32)]:
        config = config_named('smoke')
        config[key] = value
        with pytest.raises(ValueError):
            validate_config(config)
    config = config_named('smoke')
    config['model']['vocab_size'] = 1012
    with pytest.raises(ValueError):
        validate_config(config)


def test_artifacts_never_in_repo_or_existing_directory(data_dir, tmp_path):
    repo = Path(__file__).resolve().parents[4]
    for path in [repo / 'outputs/new-run', data_dir / 'run', tmp_path]:
        with pytest.raises((ValueError, FileExistsError)):
            fresh_output(str(path), data_dir)


def test_subset_identity_and_validation_selection(data_dir):
    dataset = HeptagonDataset(data_dir)
    subset = dataset.training_subset(4, 52)
    assert len(subset) == 4 and len(dataset) == 8
    assert set(subset.indices) <= set(dataset.indices)
    assert subset.identity() != dataset.identity()
    assert np.array_equal(subset.indices, dataset.training_subset(4, 52).indices)
    assert np.array_equal(select_rows(dataset, 4, 10), select_rows(dataset, 4, 10))
    assert np.array_equal(select_rows(dataset, None, 10), dataset.indices)
    with pytest.raises(ValueError):
        HeptagonDataset(data_dir, 'test').training_subset(1)


def test_rng_json_roundtrip():
    before = capture_rng()
    try:
        state = json.loads(json.dumps(capture_rng()))
        expected = (random.random(), np.random.random(), torch.rand(4))
        restore_rng(state)
        assert random.random() == expected[0]
        assert np.random.random() == expected[1]
        assert torch.equal(torch.rand(4), expected[2])
    finally:
        restore_rng(before)


def test_cpu_checkpoint_optimizer_loader_and_rng_roundtrip(data_dir, tmp_path):
    dataset = HeptagonDataset(data_dir)
    system = small_system().train()
    config = config_named('smoke')
    config['model'] = {'n_layer': 2, 'n_embd': 32, 'n_head': 2, 'n_kv_head': 2}
    config['device_batch_size'], config['total_batch_size'] = 2, 32
    loader = HeptagonDataLoader(dataset, 2)
    optimizer = torch.optim.AdamW(system.parameters(), lr=0.001, fused=False)
    loss = system.loss(next(loader))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    contract = run_contract(config, dataset, head_type=type(system.head).__name__)
    directory = tmp_path / 'checkpoint'
    save(directory, system, [optimizer], loader, contract, 1, None)
    expected_rng = (random.random(), np.random.random(), torch.rand(4))
    expected_batch = next(loader)
    expected_loss = system.loss(expected_batch)
    expected_loss.backward()
    optimizer.step()

    restored = small_system().train()
    restored_loader = HeptagonDataLoader(dataset, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, fused=False)
    meta = resume(directory, restored, [restored_optimizer], restored_loader, contract)
    assert meta['completed_steps'] == 1
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(4), expected_rng[2])
    restored_batch = next(restored_loader)
    assert torch.equal(expected_batch['idx'], restored_batch['idx'])
    loss = restored.loss(restored_batch)
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    restored_optimizer.step()
    for name, parameter in system.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])
    with pytest.raises(FileExistsError):
        save(directory, system, [optimizer], loader, contract, 2, None)
    changed = deepcopy(contract)
    changed['recipe']['max_steps'] += 1
    with pytest.raises(ValueError, match='contract mismatch'):
        resume(directory, restored, [restored_optimizer], restored_loader, changed)
