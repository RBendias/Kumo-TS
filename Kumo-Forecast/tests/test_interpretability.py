# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import torch

from channel_flow import (
    ChannelFlowConfig,
    _ChannelFlowEmbedDP,
    _embed_windows_batched,
    compute_per_channel_flow,
)
from channel_stability import compute_per_channel_embedding_stability
from interpretability import (
    _embed_tensor,
    _forecast_autoregressive,
    _identity_recipe,
    compute_target_channel_input_jacobian_importance,
    explain_forecast,
)

forecast_module = pytest.importorskip("sdm.models.kumo.timeseries.forecast.model")
KumoForecast = forecast_module.KumoForecast


@pytest.fixture(params=[False, True], ids=["temporal", "cross_channel"])
def model(request, monkeypatch):
    backbone = forecast_module._KumoForecast
    monkeypatch.setattr(
        forecast_module,
        "_KumoForecast",
        lambda **kwargs: backbone(
            **kwargs,
            context_size=16,
            prediction_size=4,
            patch_size=4,
            channels=8,
            hidden_channels=16,
            num_layers=1,
            num_heads=2,
            cross_channel_heads=2,
        ),
    )
    return KumoForecast(cross_channel=request.param, pretrained=False)


def test_embeddings_preserve_batch_and_ignore_masked_values(model):
    x = torch.randn(3, 2, 16)
    mask = torch.ones(3, 16, dtype=torch.long)
    mask[:, :4] = 0
    embeddings = _embed_tensor(model, x, mask)
    assert embeddings.shape == (3, 8)
    assert embeddings.isfinite().all()
    individual = torch.cat(
        [_embed_tensor(model, row.unsqueeze(0), m.unsqueeze(0)) for row, m in zip(x, mask, strict=True)]
    )
    torch.testing.assert_close(embeddings, individual)
    changed = x.clone()
    changed[:, :, :4] = 10000
    torch.testing.assert_close(embeddings, _embed_tensor(model, changed, mask))


def test_embedding_hook_is_removed_on_failure(model, monkeypatch):
    forward = model._forward

    def fail_after_encoding(*args: object, **kwargs):
        forward(*args, **kwargs)
        raise RuntimeError("forecast failed")

    encoder = model.model.icl if model.model.cross_channel is None else model.model.cross_channel
    hooks_before = dict(encoder._forward_hooks)
    monkeypatch.setattr(model, "_forward", fail_after_encoding)
    with pytest.raises(RuntimeError, match="forecast failed"):
        _embed_tensor(model, torch.randn(2, 2, 16), torch.ones(2, 16))
    assert encoder._forward_hooks == hooks_before


def test_forecast_matches_sdm_multitarget_rollout(model):
    x = torch.randn(3, 2, 16)
    mask = torch.ones(3, 16, dtype=torch.long)
    actual = _forecast_autoregressive(model, x, mask, model_horizon=4, target_horizon=9)
    expected = model(
        x_context=torch.empty(3, 16, 0),
        y_context=x.transpose(-1, -2),
        x_query=torch.empty(3, 9, 0),
        recipe=_identity_recipe(),
        num_estimators=1,
    ).numerical.transpose(-1, -2)
    assert actual.shape == (3, 2, 9)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_explanation_and_channel_stability_accept_sdm(model):
    series = np.random.default_rng().normal(size=(2, 16)).astype(np.float32)
    cfg = ChannelFlowConfig(time_indices=(14,), batch_size=8)
    explanation = explain_forecast(
        model,
        x_context_ct=series,
        input_mask_l=np.ones(16, dtype=np.int64),
        model_horizon=4,
        forecast_horizon=3,
        device=torch.device("cpu"),
        n_lags=1,
        surrogate=False,
        channel_axis=True,
        chan_cfg=cfg,
    )
    assert explanation.baseline_forecast.shape == (2, 3)
    assert explanation.latent_trajectory.shape == (19, 8)
    assert explanation.channel_horizon_attributions.shape == (2, 3)
    assert np.isfinite(explanation.lag_horizon_attributions).all()
    report = compute_per_channel_flow(
        model, series, seq_len=16, device=torch.device("cpu"), cfg=cfg
    )
    assert np.isfinite(report.per_channel_flow[14]).all()
    stability = compute_per_channel_embedding_stability(
        model,
        series,
        seq_len=16,
        device=torch.device("cpu"),
        time_indices=(15,),
        n_trials=2,
    )
    assert stability.lip_ratio_mean.shape == (2,)
    assert np.isfinite(stability.lip_ratio_mean).all()


def test_output_aware_attribution_uses_sdm_forecasts(model):
    importance = compute_target_channel_input_jacobian_importance(
        model,
        x_context_ct=np.random.default_rng().normal(size=(2, 16)).astype(np.float32),
        input_mask_l=np.ones(16, dtype=np.int64),
        device=torch.device("cpu"),
        target_channel=0,
        model_horizon=4,
        forecast_horizon=3,
    )
    assert importance.shape == (2, 3)
    assert np.isfinite(importance).all()
    np.testing.assert_array_equal(importance[0], 0)


def _cpu_replica(module: torch.nn.Module) -> torch.nn.Module:
    """Use PyTorch's replica semantics with CPU parameters shared read-only."""
    replica = module._replicate_for_data_parallel()
    replica._parameters = module._parameters.copy()
    for name, child in module._modules.items():
        replica._modules[name] = None if child is None else _cpu_replica(child)
    return replica


@pytest.mark.parametrize("fail_one_replica", [False, True])
def test_parallel_replica_hooks_are_isolated_and_removed(model, monkeypatch, fail_one_replica):
    windows = [torch.randn(2, 2, 16), torch.randn(3, 2, 16)]
    masks = [torch.ones(x.shape[0], 16) for x in windows]
    expected = [_embed_tensor(model, x, mask) for x, mask in zip(windows, masks, strict=True)]
    wrapper = _ChannelFlowEmbedDP(model)
    replicas = [_cpu_replica(wrapper) for _ in range(2)]
    encoder = model.model.icl if model.model.cross_channel is None else model.model.cross_channel
    hooks_before = dict(encoder._forward_hooks)
    barrier = Barrier(2)

    def synchronize(module, inputs):
        # Both replicas must have registered their capture hook before encoding.
        barrier.wait(timeout=20)

    if fail_one_replica:
        forward = replicas[0]._m._forward

        def fail_after_encoding(*args: object, **kwargs):
            forward(*args, **kwargs)
            raise RuntimeError("replica failed")

        monkeypatch.setattr(replicas[0]._m, "_forward", fail_after_encoding)

    handle = encoder.register_forward_pre_hook(synchronize)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(replica, x, mask)
                for replica, x, mask in zip(replicas, windows, masks, strict=True)
            ]
            for i, future in enumerate(futures):
                if fail_one_replica and i == 0:
                    with pytest.raises(RuntimeError, match="replica failed"):
                        future.result(timeout=30)
                else:
                    torch.testing.assert_close(future.result(timeout=30), expected[i])
    finally:
        handle.remove()
    assert encoder._forward_hooks == hooks_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dataparallel_embedding_batches_match_single_device(model):
    device = torch.device("cuda:0")
    model.to(device)
    windows = np.random.default_rng().normal(size=(5, 2, 16)).astype(np.float32)
    masks = np.ones((5, 16), dtype=np.int64)
    expected = _embed_windows_batched(model, windows, masks, device=device, batch_size=4)
    parallel = torch.nn.DataParallel(
        _ChannelFlowEmbedDP(model), device_ids=list(range(min(2, torch.cuda.device_count())))
    )
    actual = _embed_windows_batched(parallel, windows, masks, device=device, batch_size=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_multigpu_channel_flow_matches_single_device(model):
    device = torch.device("cuda:0")
    model.to(device)
    series = np.random.default_rng().normal(size=(2, 16)).astype(np.float32)
    cfg = ChannelFlowConfig(time_indices=(14,), batch_size=4)
    expected = compute_per_channel_flow(
        model, series, seq_len=16, device=device, cfg=cfg, value_fn=_embed_windows_batched
    )
    actual = compute_per_channel_flow(model, series, seq_len=16, device=device, cfg=cfg)
    np.testing.assert_allclose(
        actual.per_channel_flow, expected.per_channel_flow, rtol=1e-4, atol=1e-6, equal_nan=True
    )
