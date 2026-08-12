# Dash Pix2PixHD comparison

`DashPix2PixHD` is an author-code-aligned, SolarCHIP-compatible adaptation of
Dash et al. (2024). The directly supported paper task is full-disk HMI
line-of-sight magnetogram → AIA 304 Å at 1024×1024.

## What is implemented

- A conditional generator with a 7×7 stem, four stride-2 downsampling blocks,
  nine residual blocks, four transposed-convolution upsampling blocks, and a
  7×7 output head.
- Two conditional PatchGAN discriminators. The second discriminator receives a
  2× downsampled source/target pair.
- LSGAN (MSE) adversarial loss, matching the released HD code.
- L1 feature matching over all non-logit discriminator activations, summed over
  layers and averaged over scales, with paper weight `lambda_feature_matching=10`.
- Batch size 1, Adam `lr=2e-4`, betas `(0.5, 0.999)`, 200 epochs, and a
  code-backed 100-epoch constant + 100-epoch linear-decay schedule.
- Manual Lightning optimization with isolated discriminator and generator
  updates, detached fake images in the discriminator update, frozen discriminator
  parameters in the generator update, and detached real features for matching.
- Validation L1/MSE/PCC plus inverse-preprocessing versions of signed/absolute
  flux error and PPE10. The zero-target policy is explicit (`target > 1e-6`).

## Paper/code conflict and chosen variant

The paper's §3.4.1 describes a coarse-to-fine G1+G2 local enhancer generator, but
does not specify its layers consistently. The authors' released `pix2pixHD`
`networks.py` instead contains a single 4-down/9-residual/4-up global ResNet
generator and no G2. The paper equations use a log-cGAN notation, while released
HD code uses LSGAN. This implementation follows the concrete released network and
loss, while retaining the two-scale discriminator and feature matching explicitly
reported in the paper.

It also fixes two apparent released-code defects rather than preserving them:

- feature matching normalizes each layer by its element count and sums it once,
  instead of adding a cumulative feature sum repeatedly;
- the number of discriminators respects configuration instead of being hard-coded
  after construction.

This should be described as a **Dash et al. core-algorithm / author-code-aligned
Pix2PixHD adaptation**, not a strict reproduction.

## SolarCHIP adaptations

| Aspect | Paper / released code | This comparison |
|---|---|---|
| Tensor channels | rendered 3-channel RGB HMI and AIA | raw project tensors, 1 HMI channel → 1 AIA channel |
| Output activation | tanh on range-normalized RGB | identity on log1p + z-score values |
| Data split | 2012–2014; test-month description is internally inconsistent | existing project index blocks `[0,5000)` / `[5000,5400)` |
| Pairing and filtering | JSOC, manual filtering; exact list/tolerance unavailable | existing project co-temporal pairs |
| Checkpoint selection | generators saved every 10 epochs and evaluated on test data | every 10 epochs, top-3 by validation L1, test untouched |
| Targets | only 304 Å evaluated | ten separate project configs |

Set `output_activation: tanh` only if the dataloader is also changed to provide
values in `[-1,1]`. With the supplied z-score configs, `identity` is required.

`batch_size: 1` is per process in the supplied DDP configurations. On multiple
GPUs the global batch therefore exceeds the paper's batch of one; set
`lightning.trainer.devices=[0]` for that strict optimizer-batch condition.

The optional `lambda_l1` exists for controlled ablation but is `0.0` in all
paper-aligned configs because the reported Pix2PixHD objective is GAN + feature
matching, not the Pix2Pix pixel-L1 objective.

## Metric boundary

Dash et al. report RE, PCC, PPE10, and SSIM on rendered RGB images and used the
test set to choose checkpoints. Exact rendering ranges, timestamp pairing,
aggregation, SSIM settings, and zero handling are unavailable; the paper's RE
equation also shows an absolute value while its table reports a negative number.
The project metrics therefore have explicit local definitions and must not be
compared numerically to the paper's reported values.

Primary evidence: [final paper PDF](https://junyiye.github.io/assets/pdf/solar.pdf),
[DOI](https://doi.org/10.1007/s40745-022-00436-2), and
[authors' code](https://github.com/ankan2709/Image-Generation-Using-GANs/tree/main/pix2pixHD).
