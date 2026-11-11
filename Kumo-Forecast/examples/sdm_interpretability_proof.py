# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pretrained Kumo-TS and SDM, then visualize the SDM XAI pipeline."""

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sdm.models import KumoForecast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_flow import ChannelFlowConfig
from channel_stability import compute_per_channel_embedding_stability
from integrated_gradients import integrated_gradients_embedding
from interpretability import _embed_tensor, _forecast_tensor, _identity_recipe, explain_forecast
from sdk.forecasting import _load_cached_model, _normalization_statistics_gradient, clear_model_cache

CHECKPOINT_REVISION = "abff20a58834638b28227ff4ab934f26206e4b09"
CHECKPOINT = "run8_best_model_cr.pt"
SDM_REVISION = "083995260db6762bb8cb4f111098bd1e16bfd643"
CHANNELS = ["Demand", "Temperature", "Promotion"]


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.detach().float(), expected.detach().float()
    error = actual - expected
    return {
        "max_absolute_error": error.abs().max().item(),
        "relative_l2_error": (error.norm() / expected.norm().clamp_min(1e-12)).item(),
        "close": torch.allclose(actual, expected, rtol=1e-3, atol=1e-4),
        "rtol": 1e-3,
        "atol": 1e-4,
    }


def save_figure(output: Path, series: np.ndarray, arrays: dict, checks: dict, pipeline_precision: str) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), layout="constrained")
    figure.suptitle("Kumo-TS explainability using pretrained SDM KumoForecast", fontsize=17)
    horizon = arrays["sdm_forecast"].shape[-1]
    for channel, name in enumerate(CHANNELS):
        line, = axes[0, 0].plot(np.arange(416, 512), series[channel, 416:512], label=name)
        axes[0, 0].plot(np.arange(512, 512 + horizon), arrays["sdm_forecast"][channel], color=line.get_color())
        axes[0, 0].plot(
            np.arange(512, 512 + horizon), arrays["reference_forecast"][channel],
            color=line.get_color(), linestyle=":", linewidth=3,
        )
    axes[0, 0].axvline(511.5, color="black", alpha=0.4)
    axes[0, 0].set(title="Full history: SDM (solid), original Kumo-TS (dotted)", ylabel="Standardized signal")
    axes[0, 0].legend(ncol=3)
    for channel, name in enumerate(CHANNELS):
        line, = axes[0, 1].plot(arrays["padded_sdm_forecast"][channel], label=f"SDM: {name}")
        axes[0, 1].plot(arrays["padded_reference_forecast"][channel], color=line.get_color(), linestyle=":")
    axes[0, 1].set(title="Open discrepancy: forecasts after 384 padded steps", xlabel="Forecast horizon")
    axes[0, 1].legend(fontsize=8, ncol=3)
    for axis, values, title, labels in [
        (axes[1, 0], arrays["lag_attribution"], "Temporal attribution (latent-flow heuristic)", None),
        (axes[1, 1], arrays["channel_attribution"], "Channel attribution by forecast horizon", CHANNELS),
    ]:
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        axis.set(title=title, xlabel="Forecast horizon")
        if labels is None:
            axis.set_ylabel("Lag index (0 = most recent transition)")
        else:
            axis.set_yticks(range(len(labels)), labels)
        figure.colorbar(image, ax=axis, label="Normalized attribution")
    saliency = arrays["embedding_ig"]
    limit = max(float(np.abs(saliency).max()), 1e-12)
    image = axes[2, 0].imshow(saliency, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2, 0].set(title="Embedding Integrated Gradients (not forecast-error attribution)", xlabel="Context time step")
    axes[2, 0].set_yticks(range(3), CHANNELS)
    figure.colorbar(image, ax=axes[2, 0], label="Signed embedding attribution")
    axes[2, 1].axis("off")
    rows = [
        "Same pretrained checkpoint, evaluation mode",
        "Reference comparisons: float32",
        f"XAI baseline: {pipeline_precision}",
        "Agreement threshold: rtol=1e-3, atol=1e-4",
        "",
    ]
    for name, result in checks.items():
        if isinstance(result, dict) and "max_absolute_error" in result:
            status = "PASS" if result["close"] else "OPEN"
            rows.append(f"{status:4s} {name}: {result['max_absolute_error']:.3g}")
    rows.extend([
        "", "PASS finite, nonzero gradients to context inputs",
        "PASS finite XAI arrays and attribution normalization",
        "", "Synthetic signals; this is an integration check,",
        "not a forecast-quality or causal-explanation benchmark.",
    ])
    axes[2, 1].text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=10)
    figure.savefig(output / "overview.png", dpi=150)
    svg = output / "overview.svg"
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sdm_proof_output"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offline", action="store_true", help="Use cached Hugging Face files only.")
    parser.add_argument("--ig-steps", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    time = np.arange(536, dtype=np.float32)
    series = np.stack([
        np.sin(time / 13) + 0.3 * np.cos(time / 5),
        0.8 * np.cos(time / 23) + time / 1500,
        np.sin(time / 31) + 0.7 * np.exp(-((time - 480) / 12) ** 2),
    ]).astype(np.float32)
    x = torch.from_numpy(series[:, :512].copy()).unsqueeze(0).to(device)
    mask = torch.ones((1, 512), device=device, dtype=torch.long)
    padded_mask = mask.clone()
    padded_mask[:, :384] = 0
    padded_x = x.masked_fill(~padded_mask.bool().unsqueeze(1), 0)
    print("Loading original Kumo-TS checkpoint", flush=True)
    checkpoint = hf_hub_download(
        "nvidia/Kumo-Forecast", CHECKPOINT, revision=CHECKPOINT_REVISION, local_files_only=args.offline
    )
    reference = _load_cached_model(
        "AutonLab/MOMENT-1-large", checkpoint, 512, 72, device, local_files_only=args.offline
    ).requires_grad_(False)
    with torch.no_grad():
        reference_forecast = reference(x_enc=x, input_mask=mask).forecast.float()
        reference_embedding = reference.embed(x_enc=x, input_mask=mask).embeddings.float()
        padded_reference_forecast = reference(x_enc=padded_x, input_mask=padded_mask).forecast.float()
        padded_reference_embedding = reference.embed(x_enc=padded_x, input_mask=padded_mask).embeddings.float()
    clear_model_cache()
    del reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Loading SDM KumoForecast", flush=True)
    model = KumoForecast(device=device).eval().requires_grad_(False)
    with torch.no_grad():
        native = model(
            x_context=torch.empty((1, 512, 0), device=device),
            y_context=x.transpose(-1, -2),
            x_query=torch.empty((1, 72, 0), device=device),
            recipe=_identity_recipe(), num_estimators=1,
        ).numerical.transpose(-1, -2)
        forecast = _forecast_tensor(model, x, mask, horizon=72)
        embedding = _embed_tensor(model, x, mask)
        padded_forecast = _forecast_tensor(model, padded_x, padded_mask, horizon=72)
        padded_embedding = _embed_tensor(model, padded_x, padded_mask)
        # Kumo-TS's rollout helper enables AMP on CUDA. Compare like precision.
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            native_pipeline_precision = model(
                x_context=torch.empty((1, 512, 0), device=device),
                y_context=x.transpose(-1, -2),
                x_query=torch.empty((1, 72, 0), device=device),
                recipe=_identity_recipe(), num_estimators=1,
            ).numerical.transpose(-1, -2)
    checks = {
        "direct_sdm_forecast": comparison(forecast, native),
        "reference_forecast": comparison(forecast, reference_forecast),
        "reference_embedding": comparison(embedding, reference_embedding),
        "padded_reference_forecast": comparison(padded_forecast, padded_reference_forecast),
        "padded_reference_embedding": comparison(padded_embedding, padded_reference_embedding),
    }
    for key in ("direct_sdm_forecast", "reference_forecast", "reference_embedding", "padded_reference_embedding"):
        assert checks[key]["close"], (key, checks[key])
    print(json.dumps(checks, indent=2), flush=True)
    print("Running Kumo-TS explain_forecast with the SDM model", flush=True)
    explanation = explain_forecast(
        model, x_context_ct=series[:, :512], input_mask_l=np.ones(512, dtype=np.int64),
        model_horizon=72, forecast_horizon=24, device=device, n_lags=16,
        surrogate=False, latent_batch_size=8, channel_axis=True,
        chan_cfg=ChannelFlowConfig(time_indices=tuple(range(495, 511)), batch_size=8),
    )
    checks["pipeline_native_forecast"] = comparison(
        torch.from_numpy(explanation.baseline_forecast).to(device), native_pipeline_precision[0, :, :24]
    )
    assert checks["pipeline_native_forecast"]["close"], checks["pipeline_native_forecast"]
    np.testing.assert_allclose(explanation.lag_horizon_attributions.sum(axis=0), 1, atol=1e-5)
    np.testing.assert_allclose(explanation.channel_horizon_attributions.sum(axis=0), 1, atol=1e-5)
    print("Checking context gradients and embedding Integrated Gradients", flush=True)

    def embed_fn(values: torch.Tensor) -> torch.Tensor:
        observed = mask.expand(values.shape[0], -1)
        with _normalization_statistics_gradient(model, enabled=True):
            return _embed_tensor(model, values, observed)

    context = x.clone().requires_grad_(True)
    gradient = torch.autograd.grad(embed_fn(context).square().sum(), context)[0]
    assert gradient.isfinite().all() and gradient.norm() > 0
    ig = integrated_gradients_embedding(
        embed_fn, x[0], baseline="noise", n_steps=args.ig_steps,
        internal_batch_size=1, reduce="l2", seed=0, device=device,
    )
    print("Checking per-channel embedding stability", flush=True)
    stability = compute_per_channel_embedding_stability(
        model, series[:, :512], seq_len=512, device=device,
        time_indices=np.array([511]), n_trials=3, batch_size=4, trajectory_batch_size=8,
    )
    arrays = {
        "sdm_forecast": forecast[0, :, :24].cpu().numpy(),
        "reference_forecast": reference_forecast[0, :, :24].cpu().numpy(),
        "padded_sdm_forecast": padded_forecast[0, :, :24].cpu().numpy(),
        "padded_reference_forecast": padded_reference_forecast[0, :, :24].cpu().numpy(),
        "sdm_embedding": embedding.cpu().numpy(),
        "reference_embedding": reference_embedding.cpu().numpy(),
        "lag_attribution": explanation.lag_horizon_attributions,
        "channel_attribution": explanation.channel_horizon_attributions,
        "latent_trajectory": explanation.latent_trajectory,
        "embedding_ig": ig.attribution,
        "channel_stability": stability.lip_ratio_mean,
    }
    for name, values in arrays.items():
        assert np.isfinite(values).all(), name
    import sdm

    source = Path(sdm.__file__).resolve().parents[1]
    revision = subprocess.check_output([shutil.which("git"), "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "sdm_revision": revision, "expected_sdm_revision": SDM_REVISION,
        "checkpoint": CHECKPOINT, "checkpoint_revision": CHECKPOINT_REVISION,
        "torch_version": torch.__version__, "device": str(device),
        "reference_forecast_horizon": 72, "xai_forecast_horizon": 24,
        "reference_comparison_precision": "float32",
        "pipeline_precision": "CUDA autocast float16" if device.type == "cuda" else "float32",
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("interpretability.py", "channel_flow.py", "sdk/forecasting.py", "examples/sdm_interpretability_proof.py")
        },
        "checks": checks, "context_gradient_norm": gradient.norm().item(),
        "array_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "ig_steps": ig.n_steps, "ig_convergence_delta": ig.convergence_delta,
        "channel_stability": stability.lip_ratio_mean.tolist(),
        "limits": [
            "Padded forecasts currently differ from the original Kumo-TS model; this is reported, not hidden.",
            "Surrogate fitting, retrieval/DARR, and multi-GPU execution are not exercised by this example.",
            "Embedding IG explains the embedding norm; latent-flow attribution is a heuristic, not a causal claim.",
            "Rolling windows with no fully observed patch have undefined embeddings; the existing pipeline replaces these with zero.",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    save_figure(args.output, series, arrays, checks, metadata["pipeline_precision"])
    print(f"Saved proof artifacts to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
