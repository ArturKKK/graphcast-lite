"""Пересчёт активаций процессора (grad_checkpoint, 29.09.2026).

Двойное сгущение меша не влезло в 80 ГБ: процессор хранит активации всех
12 шагов на каждом шаге развёртки. С grad_checkpoint они пересчитываются на
обратном проходе. Опыт ref2 сравнивается с ref, который учился без пересчёта,
поэтому здесь проверяется главное: выход и градиенты те же самые.
"""
import pytest
from conftest import needs_torch


@needs_torch
@pytest.mark.parametrize("train_mode", [True, False])
def test_checkpointed_processor_matches_plain(train_mode):
    import torch

    from src.models import InteractionNetProcessor

    def make(ckpt):
        torch.manual_seed(0)
        return InteractionNetProcessor(node_dim=8, raw_edge_dim=4, edge_latent_dim=8,
                                       hidden_dim=8, num_steps=3, grad_checkpoint=ckpt)

    plain, ckpt = make(False), make(True)
    plain.train(train_mode)
    ckpt.train(train_mode)
    torch.manual_seed(1)
    n, e = 20, 60
    x = torch.randn(n, 8)
    ei = torch.randint(0, n, (2, e))
    ea = torch.randn(e, 4)

    outs = []
    for m in (plain, ckpt):
        xi = x.clone().requires_grad_(True)
        y = m(xi, ei, ea)
        y.square().sum().backward()
        # у обновления рёбер последнего шага градиента нет в обоих режимах
        outs.append((y.detach(), xi.grad, [p.grad for p in m.parameters()]))

    torch.testing.assert_close(outs[0][0], outs[1][0])
    torch.testing.assert_close(outs[0][1], outs[1][1])
    for a, b in zip(outs[0][2], outs[1][2]):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


def test_config_accepts_grad_checkpoint():
    from src.config import GraphBlock
    g = GraphBlock(layer_type="interaction_net", output_dim=8,
                   num_message_passing_steps=2, grad_checkpoint=True)
    assert g.grad_checkpoint is True
