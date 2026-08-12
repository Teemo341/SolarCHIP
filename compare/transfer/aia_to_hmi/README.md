# AIA-to-HMI comparison baselines

This directory contains SolarCHIP training adapters for two peer-reviewed
AIA-to-magnetogram methods:

| Directory | Paper-aligned method | Optimization |
| --- | --- | --- |
| `dannehl_pix2pixcc` | Dannehl, Delouille & Barra (2024), Pix2PixCC | conditional GAN + feature matching + multiscale concordance-correlation loss |
| `i2iwfilm` | Sayez et al. (2025), I2IwFiLM | deterministic two-stage reconstruction and guidance losses |

Both modules are ordinary `pytorch_lightning.LightningModule` classes and are
launched through the existing `solarchip/main/train.py` entry point. They read
the existing dataloader batch dictionary directly and do not replace or modify
SolarCHIP's dataset code.

## Run

From the repository root, for example:

```bash
python -m solarchip.main.train \
  -b configs/compare/aia_to_hmi/dannehl_pix2pixcc/aia_0304_to_hmi.yaml

python -m solarchip.main.train \
  -b configs/compare/aia_to_hmi/i2iwfilm/aia_0304_to_hmi.yaml
```

To resume, explicitly pass the same comparison config together with the run
directory. This prevents the default VQGAN base config in `train.py` from being
merged after the saved run config:

```bash
python -m solarchip.main.train \
  -r logs/<run-directory> \
  -b configs/compare/aia_to_hmi/i2iwfilm/aia_0304_to_hmi.yaml
```

Each model has ten configs, one for every SolarCHIP AIA key:
`0094`, `0131`, `0171`, `0193`, `0211`, `0304`, `0335`, `1600`, `1700`, and
`4500`.

## Important direction contract

`data.dataset.SolarDataset.multimodal_dataset` requires `hmi` to be the first
entry of `modal_list`, even for the reverse translation direction. Therefore a
correct AIA-to-HMI config deliberately contains:

```yaml
modal_list: ['hmi', '0304']
```

The model reverses the direction by setting `source_modal: '0304'` and
`target_modal: 'hmi'`. Reordering `modal_list` will make the existing dataset
constructor fail.

## Evidence and adaptation boundaries

- The directly evaluated single-band task in both papers is AIA 304 Angstrom
  to HMI. The other eight AIA passbands are algorithm-transfer comparisons;
  `4500` is an additional SolarCHIP project extension and is not EUV.
- The project dataset exposes one `hmi.M_720s`-style LOS channel. Dannehl et
  al. also analyze vector-field targets, while the released code commonly uses
  Bx/By/Bz. These configs intentionally remain one-input/one-output to match the
  SolarCHIP dataloader.
- SolarCHIP's paired samples use signed `log1p` followed by per-modality
  z-scoring, the project's time split, paired augmentation, and 1024-pixel
  resizing. The paper datasets use different clipping, scaling, sampling, and
  spatial resolutions. Network outputs are consequently linear rather than
  `tanh` bounded.
- The implementation READMEs distinguish peer-reviewed statements, author-code
  details, paper/code conflicts, and engineering choices. These are
  core-algorithm reproductions in the SolarCHIP framework, not bit-exact or
  metric-exact reproductions.
- A generated magnetogram is a model estimate and must not be treated as a
  substitute for an HMI measurement. In particular, EUV intensity does not
  uniquely determine the magnetic-field structure or polarity details.

## Dependency-free debug check

The preparation environment does not include PyTorch. Run the static validator
to check Python syntax, config coverage, dotted targets, constructor arguments,
directionality, paper-aligned defaults, logger keys, and checkpoint contracts:

```bash
python compare/transfer/aia_to_hmi/debug_validate.py
```

Tensor execution, autograd, GPU memory use, and distributed training remain for
the real PyTorch environment.
