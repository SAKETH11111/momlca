# Pretrained Backbone Artifacts

This directory holds DVC-tracked pretrained checkpoints that are safe to reference from Hydra experiment configs.

## Available Artifact

- `isdpainn_random_split_painn_stage_backbone.pt`
  - Format: plain PyTorch state dict
  - Loading contract:
    - `model.pretrained_backbone.checkpoint_path=artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.pt`
    - `model.pretrained_backbone.checkpoint_format=state_dict`
    - `model.pretrained_backbone.backbone_key_prefix=backbone.`
  - Compatible backbone: `gnn.models.backbones.PaiNNStageBackbone`

## Materialize The Artifact

The repo now points at a public read-only DVC cache mirror, so a fresh clone can materialize the artifact with the standard pull command:

```bash
poetry install
poetry run dvc pull
```

If you only need the pretrained checkpoint, pull the specific sidecar:

```bash
poetry run dvc pull artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.pt.dvc
```

## Canonical Fine-Tune Commands

PaiNN-stage fine-tune preset:

```bash
poetry run python scripts/train.py experiment=pfasbench_finetune
```

MoMLCA-shell fine-tune preset with the same concrete backbone override:

```bash
poetry run python scripts/train.py experiment=pfasbench_finetune_momlca
```

## Provenance Summary

- Public upstream checkpoint: ISD-PaiNN random-split weights from Zenodo DOI `10.5281/zenodo.10547719`
- Upstream code/license: `nmdl-mizo/isdpainn` (`v1.0.0`, MIT)
- Upstream weights license: CC-BY-4.0
- Upstream retrieval timestamps (UTC): checkpoint `2026-04-13T17:25:45Z`, config `2026-04-13T17:25:46Z`
- Immutable upstream checkpoint digests: `sha256=9473a3d62485297825e43dab47d9fd6be9956487e212c9143e1663f5538cf3ec`, `md5=5265a7d187000628c2b71bc26e9bd46f`
- Immutable upstream config digests: `sha256=0cac15db2931da1cdf86fea638a22d7507b160db4a14e8dc6c899542a36f186f`, `md5=de5c3250ca57531e7a43acdd49687ddf`
- Conversion script: `scripts/prepare_pretrained_backbone.py`
- Conversion metadata: `artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.metadata.yaml`
- Repo-side schema pinning: metadata records the conversion script SHA-256, feature-schema module SHA-256, repo HEAD commit, and a schema digest for exact regeneration.

## Regenerate The Normalized Artifact

```bash
poetry run python scripts/prepare_pretrained_backbone.py
poetry run dvc add artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.pt
poetry run dvc push -r <your-writable-remote>
```

The script now caches the upstream Zenodo files under your system temporary directory by default, so the documented regeneration command no longer leaves `.tmp/upstream/` repo junk behind. If you want to reuse a pinned local download instead, pass `--source-checkpoint` and `--source-config` explicitly.

The conversion is deterministic: it verifies the pinned upstream SHA-256 and MD5 digests before proceeding, loads the public checkpoint via `torch.load(..., weights_only=True)` on CPU, reduces the upstream invariant atom embeddings from 512 to 128 channels, maps the repo's C/N/O/F one-hot atom columns to the converted upstream embeddings, uses the upstream hydrogen embedding for the repo's `other` atom bucket, and falls back to the mean H/C/N/O/F embedding for P/S/Cl/Br/I because those elements are not present in the QM9-trained source checkpoint.

Publishing still requires an explicit writable DVC remote; the checked-in `githubraw` remote remains read-only for `dvc pull`.

The metadata also pins the repo-side conversion contract that can change the normalized output over time: the exact script digest, the atom-feature constants module digest, the current repo HEAD commit, and a derived schema fingerprint covering the feature-layout assumptions used during conversion.

## Limitations

- The artifact is intentionally scoped to the current placeholder `PaiNNStageBackbone` delivered before the full PaiNN implementation lands.
- The fine-tune presets pin `model.backbone.use_positions=false`, so this artifact targets the 22-feature atom-input seam of the current placeholder backbone rather than the optional position-augmented 26-feature variant.
- The `pfasbench_finetune_momlca` experiment still overrides its backbone to `PaiNNStageBackbone`; the artifact is not compatible with the default MoMLCA fallback backbone.
