"""EMA, накопление и обрезка градиента в train_epoch.

Добавлены 23.09.2026. По умолчанию выключены, и главная проверка — что при
выключенных приёмах обучение идёт ровно как раньше: иначе все посчитанные
таблицы статьи перестали бы воспроизводиться.
"""
from conftest import needs_torch

pytestmark = needs_torch

G, C, OBS = 5, 3, 2


def make_model(torch, nn):
    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_window = OBS
            self.w = nn.Parameter(torch.zeros(C))

        def forward(self, X, attention_threshold=0.0, **kw):
            N, Gg, _ = X.shape
            return X.view(N, Gg, OBS, C)[:, :, -1, :] * 0 + self.w
    return Lin()


def batches(torch, n):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(1, G, OBS * C, generator=g),
             torch.randn(1, G, 2 * C, generator=g)) for _ in range(n)]


class CountingSGD:
    """Обёртка, считающая шаги оптимизатора."""
    def __init__(self, opt):
        self.opt, self.steps = opt, 0

    def step(self):
        self.steps += 1
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

    def __getattr__(self, k):
        return getattr(self.opt, k)


def run(torch, model, data, **kw):
    from src.train import train_epoch
    opt = CountingSGD(torch.optim.SGD(model.parameters(), lr=0.1))
    train_epoch(model, data, opt, None, torch.device("cpu"), threshold=0.0, epoch=0,
                current_ar_steps=2, use_residual=True, **kw)
    return opt


def test_defaults_reproduce_old_behaviour():
    """Без новых приёмов: шаг оптимизатора на каждый пример, как было всегда."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    m = make_model(torch, nn)
    opt = run(torch, m, batches(torch, 6))
    assert opt.steps == 6


def test_accumulation_steps_less_often_and_matches_mean_gradient():
    """accum=3 на 6 примерах: два шага, и градиент — среднее, а не сумма."""
    import torch
    import torch.nn as nn
    data = batches(torch, 6)
    m1 = make_model(torch, nn)
    opt = run(torch, m1, data, grad_accum_steps=3)
    assert opt.steps == 2

    # Эталон: те же два шага руками по среднему лоссу трёх примеров
    m2 = make_model(torch, nn)
    sgd = torch.optim.SGD(m2.parameters(), lr=0.1)
    for k in (0, 3):
        sgd.zero_grad()
        tot = 0
        for X, y in data[k:k + 3]:
            yv = y.view(1, G, 2, C)
            cur = X.view(1, G, OBS, C)
            ls = 0
            for s in range(2):
                out = cur[:, :, -1, :] + m2.w
                ls = ls + ((out - yv[:, :, s, :]) ** 2).mean()
                cur = torch.cat([cur[:, :, 1:, :], out.unsqueeze(2)], dim=2)
            tot = tot + ls / 2
        (tot / 3).backward()
        sgd.step()
    assert torch.allclose(m1.w, m2.w, atol=1e-5), (m1.w, m2.w)


def test_tail_of_epoch_is_not_lost():
    """7 примеров при accum=3: третий шаг на хвосте, градиент хвоста не пропадает."""
    import torch
    import torch.nn as nn
    opt = run(torch, make_model(torch, nn), batches(torch, 7), grad_accum_steps=3)
    assert opt.steps == 3


def test_grad_clip_limits_update():
    import torch
    import torch.nn as nn
    data = [(torch.zeros(1, G, OBS * C), torch.full((1, G, 2 * C), 100.0))]
    free = make_model(torch, nn)
    run(torch, free, data)
    clip = make_model(torch, nn)
    run(torch, clip, data, grad_clip_norm=1.0)
    assert clip.w.norm() < free.w.norm()
    assert clip.w.norm() <= 0.1 * 1.0 + 1e-5      # lr × норма


def test_ema_tracks_weights_and_swap_roundtrips():
    import torch
    import torch.nn as nn

    from src.train import WeightEMA
    m = make_model(torch, nn)
    ema = WeightEMA(m, 0.5)
    run(torch, m, batches(torch, 4), ema=ema)
    raw = m.w.detach().clone()
    avg = ema.shadow["w"].clone()
    assert not torch.allclose(raw, avg), "EMA не отличается от сырых весов"
    ema.swap(m)
    assert torch.allclose(m.w, avg)
    ema.swap(m)
    assert torch.allclose(m.w, raw), "swap не вернул сырые веса"
