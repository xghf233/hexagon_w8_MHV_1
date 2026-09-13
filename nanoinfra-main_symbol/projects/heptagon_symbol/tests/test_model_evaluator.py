"""CPU-only tests, to be executed on the server. No local execution implied."""

import numpy as np
import pytest
import torch

from core.model.gpt import GPT, GPTConfig
from core.model.heads import LMHead
from core.model.inference import autoregressive_generate
from core.model.system import LMSystem
from projects.amplitude_symbol.blocks.attention import attach_word_bidirectional_attention
from projects.heptagon_symbol import evaluator as ev
from projects.heptagon_symbol.model import attach_attention, validate_attention
from projects.heptagon_symbol.tokenizer import assemble_batch, build_layout, encode_prompt


def small_system():
    torch.manual_seed(721)
    config = GPTConfig(sequence_len=16, vocab_size=1048, n_layer=2, n_embd=32,
                       n_head=2, n_kv_head=2, n_token_types=3)
    # Default nonzero projections are deliberate: zero init would hide mask bugs.
    with torch.device('cpu'):
        system = LMSystem(GPT(config), LMHead(32, 1048))
    system.arch = 'GPT'
    attach_attention(system)
    return system.eval()


@torch.no_grad()
def test_no_answer_leakage_with_negative_control():
    system = small_system()
    encoded = assemble_batch([[0, 1, 2, 3, 4, 5]] * 2, [48, -24])
    hidden = system.trunk(encoded.tokens[:, :-1], token_types=encoded.token_types[:, :-1])
    torch.testing.assert_close(hidden[0, :8], hidden[1, :8], atol=1e-6, rtol=1e-6)
    mask = system.trunk.blocks[0].attn._word_bidi_mask
    assert torch.isneginf(mask[:8, 8:]).all()
    assert (mask[1:7, 1:7] == 0).all()
    assert torch.isneginf(mask[8, 9:]).all()
    # The old ten-letter defaults must really leak on the same nontrivial weights.
    attach_word_bidirectional_attention(system, 16)
    wrong = system.trunk(encoded.tokens[:, :-1], token_types=encoded.token_types[:, :-1])
    assert not torch.allclose(wrong[0, 1:7], wrong[1, 1:7])
    with pytest.raises(ValueError, match='attention'):
        validate_attention(system)


@torch.no_grad()
def test_cached_greedy_matches_full_reforward():
    system = small_system()
    ids = torch.tensor([encode_prompt([0, 1, 2, 3, 4, 5]), encode_prompt([6] * 6)])
    types = build_layout().classify_token_types(ids)
    cached = autoregressive_generate(system, ids, types, max_new_tokens=6,
                                     gen_token_type=2, temperature=0, stop_token=None)
    expected = []
    for _ in range(6):
        hidden = system.trunk(ids, token_types=types)
        token = system.head(hidden[:, -1:]).argmax(-1)
        expected.append(token)
        ids = torch.cat([ids, token], dim=1)
        types = torch.cat([types, torch.full_like(token, 2)], dim=1)
    assert torch.equal(cached, torch.cat(expected, dim=1))


@pytest.mark.parametrize('tokens, expected', [
    ([46, 49, 44], 1), ([47, 96, 44, 44, 44], -48),
    ([46, 60, 382, 44], 12334), ([46, 48, 44], 0),
    ([47, 48, 44], None), ([46, 48, 49, 44], None),
    ([46, 49], None), ([46, 49, 44, 49], None), ([0, 49, 44], None),
])
def test_engine_eos_padding(tokens, expected):
    assert ev.decode_engine_output(tokens) == expected


def test_metrics_invalid_and_zero_are_not_sign_success():
    result = ev.metrics([1, -2, None, 0, -4], [1, 2, -3, 1, -4])
    assert result['exact_accuracy'] == 2 / 5
    assert result['magnitude_accuracy'] == 3 / 5
    assert result['sign_accuracy'] == 2 / 5
    assert result['invalid_rate'] == 1 / 5


def test_generator_receives_only_word_prompts_and_restores_mode(monkeypatch):
    system = small_system().train()
    received = []
    def fake_engine(system, prompt_ids, prompt_types, **kwargs):
        received.append(prompt_ids.tolist())
        assert prompt_ids.shape[1] == 8
        assert torch.equal(prompt_types, build_layout().classify_token_types(prompt_ids))
        assert kwargs['temperature'] == 0 and kwargs['top_k'] is None
        assert kwargs['max_new_tokens'] == 8
        return torch.tensor([[46, 49, 44, 44]] * len(prompt_ids))
    monkeypatch.setattr(ev, 'autoregressive_generate', fake_engine)
    words = np.array([[0] * 6, [1] * 6, [2] * 6], dtype=np.uint8)
    assert ev.generate_coefficients(system, words, batch_size=2) == [1, 1, 1]
    assert received == [[encode_prompt(words[0]), encode_prompt(words[1])], [encode_prompt(words[2])]]
    assert system.training
    def fail(*args, **kwargs):
        raise RuntimeError('engine failure')
    monkeypatch.setattr(ev, 'autoregressive_generate', fail)
    with pytest.raises(RuntimeError):
        ev.generate_coefficients(system, words)
    assert system.training


def test_corrupt_runtime_mask_is_rejected():
    system = small_system()
    system.trunk.blocks[0].attn._word_bidi_mask[1, 8] = 0
    with pytest.raises(ValueError, match='content'):
        validate_attention(system)
