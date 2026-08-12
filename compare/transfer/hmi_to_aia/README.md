# HMI → AIA paper baselines

This directory contains two paper-aligned comparison implementations that plug
directly into `solarchip/main/train.py` and the existing
`data.dataset.SolarDataset.multimodal_dataset` dataloader.

| Directory | Paper anchor | Implemented comparison |
|---|---|---|
| `dash_pix2pixhd` | Dash et al. (2024) | conditional global ResNet generator + two-scale PatchGAN + LSGAN + discriminator feature matching |
| `sdoml_cnn` | Galvez et al. (2019) | deterministic 11-layer, 128-filter CNN regression baseline + MSE + SGD/Nesterov |

The implementations are deliberately self-contained under the user-authorized
comparison path. No SolarCHIP source, dataloader, trainer, or callback file is
modified.

## Run

From the repository root, select any one of the ten independent configurations:

```bash
python -m solarchip.main.train \
  -b configs/compare/hmi_to_aia/dash_pix2pixhd/hmi_to_0304.yaml

python -m solarchip.main.train \
  -b configs/compare/hmi_to_aia/sdoml_cnn/hmi_to_0304.yaml
```

CLI dotlist overrides still work through the existing trainer, for example:

```bash
python -m solarchip.main.train \
  -b configs/compare/hmi_to_aia/sdoml_cnn/hmi_to_0193.yaml \
  data.params.batch_size=2 lightning.trainer.devices=[0]
```

Resume either run with the existing interface **and repeat the same comparison
config explicitly**. The current parser has a legacy VQGAN default for `--base`,
so omitting `-b` while resuming can merge that unrelated default last:

```bash
python -m solarchip.main.train \
  -r logs/<run-directory> \
  -b configs/compare/hmi_to_aia/dash_pix2pixhd/hmi_to_0304.yaml
```

Each configuration loads only `['hmi', '<target>']`, so it does not require all
eleven modalities to exist at a timestamp. Train augmentation remains paired;
validation deliberately disables random flip/rotation so checkpoint selection is
stable.

## Ten targets and evidence status

The project targets are `0094`, `0131`, `0171`, `0193`, `0211`, `0304`, `0335`,
`1600`, `1700`, and `4500`.

- Dash et al. trained and evaluated only HMI LOS → AIA 304 Å. The other eight
  UV/EUV configurations transfer the same algorithm, which the authors describe
  as compatible with other passbands but do not evaluate. The 4500 Å config is a
  SolarCHIP-only extension.
- Galvez et al. jointly predicted the first nine AIA channels from HMI Bx/By/Bz.
  All supplied configurations adapt that 3→9 vector-field task to the project's
  1→1 LOS task. The 4500 Å config is again outside the paper.

Consequently, this code is an architecture-level comparison in a common data and
training framework. It is not a bit-exact, data-exact, or metric-exact reproduction.

## Static debug without PyTorch

The local environment used to prepare this comparison did not contain PyTorch.
Run the dependency-free structural checks with:

```bash
python3 compare/transfer/hmi_to_aia/debug_validate.py
```

The check compiles every Python file as AST, verifies all twenty config contracts,
checks target/channel consistency, and confirms the expected paper hyperparameters.
It does not execute tensor operations. A real forward/backward smoke test remains
required in the training environment.

## Primary sources

- Dash, A., Ye, J., Wang, G., & Jin, H. (2024). *High Resolution Solar Image
  Generation Using Generative Adversarial Networks*. Annals of Data Science,
  11, 1545–1561. [DOI](https://doi.org/10.1007/s40745-022-00436-2),
  [final paper PDF](https://junyiye.github.io/assets/pdf/solar.pdf),
  [authors' released code](https://github.com/ankan2709/Image-Generation-Using-GANs).
- Galvez, R., et al. (2019). *A Machine-learning Data Set Prepared from the NASA
  Solar Dynamics Observatory Mission*. The Astrophysical Journal Supplement
  Series, 242, 7. [DOI](https://doi.org/10.3847/1538-4365/ab1005),
  [peer-reviewed full text](https://eprints.gla.ac.uk/187270/1/187270.pdf).
