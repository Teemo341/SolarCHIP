# Galvez SDOML-CNN comparison

`GalvezSDOMLCNN` is a single-band LOS adaptation of the deterministic CNN
baseline in Galvez et al. (2019), §5.2. The paper is primarily an SDOML dataset
and protocol paper, but it does provide a concrete HMI→AIA trainable baseline.
It is not a GAN and does not define a newly named generative architecture.

## Paper-described network and training

- Input: HMI Bx/By/Bz, three channels; output: nine AIA UV/EUV channels jointly.
- Resolution used by the model: 256×256.
- Stem: 7×7 convolution, stride 2, then 3×3 max-pooling, stride 2.
- Body: stride-1 3×3 convolutions. Every intermediate convolution has 128
  filters and is followed by ReLU and batch normalization.
- Head: 3×3 output convolution followed by 4× bilinear upsampling.
- Compared depths: 3, 7, and 11 convolutions. The supplied default is 11 layers,
  which has the best aggregate result and means nine body convolutions between
  the stem and head.
- Objective: MSE.
- Optimizer: mini-batch SGD with Nesterov momentum 0.99, weight decay `1e-8`,
  initial LR `1e-3`, LR multiplied by 0.1 every five epochs, 15 epochs total.
- Original batch size: 32; checkpoint each epoch and select lowest validation
  loss.

The paper does not report exact padding, bias, initialization, seed, shuffle,
augmentation, BN/ReLU implementation order beyond prose, `align_corners`, model
code, pretrained weights, or channel means. This is therefore a paper-described
rewrite rather than an official-code port.

## SolarCHIP adaptations

| Aspect | Galvez et al. | This comparison |
|---|---|---|
| HMI input | `hmi.B_720s` vector Bx/By/Bz | project `hmi.M_720s`-style LOS key, one channel |
| AIA output | nine channels jointly | one target per independent config |
| Resolution | 256×256 | 1024×1024 for comparison with current SolarCHIP training |
| Normalization | divide each channel by training mean | existing log1p + fixed z-score |
| Batch | 32 at 256px | 2 at 1024px; lower by CLI if memory requires |
| Split | temporal blocks; 2012/2013 test | existing project `[0,5000)` train and `[5000,5400)` validation |
| 4500 Å | excluded | explicit project-only extension |

The core topology, MSE, optimizer, LR schedule, and 15-epoch training length are
preserved. The model keeps generic `input_channels` and `output_channels`, so a
future strict data adapter can instantiate the original 3→9 topology without
changing network code.

Keeping the unchanged convolution topology at 1024 px reduces its receptive
field as a fraction of the solar disk relative to the paper's 256 px experiment.
This is an intentional common-resolution comparison choice, not an exact
reproduction of the paper's spatial scale.

## Metrics

Training and checkpoint selection use MSE in the project's normalized space.
Validation also reverses the project's signed-log1p and AIA z-score transforms and logs
the paper's normalized absolute error and fractions of valid pixels below 10%,
20%, and 50% relative error. Each metric is averaged over valid pixels per image
and then over images, following the paper's aggregation description. Pixels
with raw target `<=1e-6` are excluded to make the paper's unspecified
division-by-zero policy explicit.

Primary evidence: [peer-reviewed full text](https://eprints.gla.ac.uk/187270/1/187270.pdf),
[institutional record](https://eprints.gla.ac.uk/187270/), and
[DOI](https://doi.org/10.3847/1538-4365/ab1005).
