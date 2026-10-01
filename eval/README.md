# Evaluation

Ground truth comes from public salient object detection datasets, where a
pixel mask marks the main subject(s), and from one attention dataset.

```sh
eval/fetch.sh                  # all datasets into eval/data (~900 MB)
eval/fetch.sh ECSSD PASCAL-S   # or a subset
bun eval/run.ts eval/data/ECSSD eval/data/PASCAL-S
```

## Datasets

The official academic download hosts are often unreachable, so `fetch.sh`
pulls third-party copies that are committed to public GitHub repositories.
Counts and file names match the official releases. These datasets are for
research and evaluation only. Don't redistribute them.

| Name | Images | Ground truth | Notes |
|---|---|---|---|
| ECSSD | 1000 | binary masks | One subject, busy backgrounds. Strong centre bias. |
| PASCAL-S | 850 | graded masks (annotator agreement) | Often several objects. Masks are thresholded at 0.5. |
| DUTS-TE | 5019 | masks with anti-aliased edges | Standard test set, smaller subjects (15% of the image on average). |
| MSRA10K | 10000 | binary masks | Easy, single and mostly centred subjects. |
| SALICON | 1000 | mouse-tracking attention maps | Random subset of COCO val2014, cluttered everyday scenes, often no single subject. |

Sources and citations:

- **ECSSD**: [total-black/U2Net-ECSSD-Evaluation](https://github.com/total-black/U2Net-ECSSD-Evaluation).
  Yan et al., *Hierarchical Saliency Detection*, CVPR 2013.
- **PASCAL-S**: [frankLeo123/Saliency_Matlab](https://github.com/frankLeo123/Saliency_Matlab).
  Li et al., *The Secrets of Salient Object Segmentation*, CVPR 2014.
- **DUTS-TE, MSRA10K**: [beqooo09/SOD_Project](https://github.com/beqooo09/SOD_Project).
  Wang et al., *Learning to Detect Salient Objects with Image-level Supervision*, CVPR 2017;
  Cheng et al., *Global Contrast based Salient Region Detection*, TPAMI 2015.
- **SALICON**: maps from [dogsteven/salicon-maps-val](https://github.com/dogsteven/salicon-maps-val),
  images from the COCO S3 bucket. Jiang et al., *SALICON: Saliency in Context*, CVPR 2015.

## Metrics

Every image is decoded at 256 px on its longest side.

- **hit**: the share of images where the point lands on the subject (mask ≥ 0.5).
  For SALICON this means within the region that has at least half of the peak attention.
- **dist**: the distance from the point to the subject's centroid, in normalised image coordinates.
- **square / portrait / banner / zoom**: the fraction of the subject that is
  still visible after cropping around the point. The crop is the largest one of
  that shape that fits, clamped to the image:
  - square: 1:1
  - portrait: 9:16
  - banner: 3:1
  - zoom: the same aspect ratio at half the width and height
- **avg**: the mean of the four crop scores. This is the number that matters most.

Baselines:

- **oracle**: the true centroid of the mask. This is the best a single point can do.
- **center**: always (0.5, 0.5).
- **smartcrop**: [smartcrop.js](https://github.com/jwagner/smartcrop.js). It is
  run separately for each crop shape, so it gets the target aspect ratio, which
  our point does not. For hit and dist its square crop centre is used.

## Tools

- `run.ts`: the benchmark. `--limit N` samples N images evenly, and `--methods` picks which methods to run.
- `tune.ts`: a coordinate descent over the parameters. It tunes on even-numbered
  images and reports on odd-numbered ones. `--hit` sets how much the hit rate
  weighs against crop retention.
- `failures.ts`: renders the worst misses (red is the prediction, green the truth).
- `debug.ts`: renders the saliency map and focus point for any images.

The defaults were tuned on ECSSD and PASCAL-S with `--hit 0.2`. DUTS-TE,
MSRA10K and SALICON were held out.
