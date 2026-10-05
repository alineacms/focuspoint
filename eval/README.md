# Evaluation

Ground truth comes from public salient object detection datasets, where a
pixel mask marks the main subject(s), and from one attention dataset.

```sh
eval/fetch.sh                  # all datasets into eval/data (~1.7 GB)
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
| SALICON | 5000 | mouse-tracking attention maps | COCO val2014, cluttered everyday scenes, often no single subject. |

For training only, `fetch.sh SALICON-TR DUTS-TR` adds the 10,000 SALICON
train images (COCO train2014, maps from
[dogsteven/salicon-maps-train](https://github.com/dogsteven/salicon-maps-train))
and the 10,553 DUTS-TR images. `fetch.sh LLM` fetches the images behind the
LLM labelling experiment, see [llm/README.md](llm/README.md).

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

Every image is decoded at 256 px on its longest side. Ground truth is
"importance" per pixel: a subject mask, or the SALICON attention map scaled so
its peak is 1.

The point is scored by how it crops. For five container shapes (3:1, 16:9,
1:1, 4:5, 9:16), the image is cropped to the largest window of that shape. The
window is centred on the point and clamped to the image. Only one axis gets
cropped, as with `object-fit: cover`.

- **peak** ("top spot", the main number): the share of crops that contain the
  most important spot, which is the peak of the lightly blurred ground truth
  (for example the face most people looked at). A focus point is one point,
  so what counts is that the thing people look at first survives every crop.
  Flat subject masks tie over their whole interior; ties go to the middle of
  the tied area, so on mask sets the spot is the heart of the subject (which
  almost every crop keeps, so peak mostly discriminates on SALICON).
- **head**: the share of crops that keep the whole most important region:
  the area around the peak that stays above half of it. For attention that
  is the face or object people looked at.
- **kept**: the share of total importance inside the window, averaged over
  the five shapes. A guard against regressions: maximising it can put the
  point between two people and cut both.
- **hit**: whether the point lands on the subject (importance ≥ 0.5). This is
  secondary: a good point may deliberately sit off-centre or near an edge.

Two more checks:

- **Faces** (`faces.ts`): on images where YuNet finds faces
  (`train/faces.py` writes `faces.json`), the share of crops that keep the
  main head, or every head, whole. "ideal" centres on the main head, the
  best any point can do. Cutting through a head is the worst crop there is.
- **Hand-marked points** (`marks.ts`): ideal points marked on the website
  (`site/labels.json`) become `eval/data/MARKS`, a disc mask at each point, so
  peak there is how often the crops keep the marked point. It is a test set:
  nothing is tuned on it.

Baselines:

- **oracle**: the best single point for each image, found by exhaustive search
  for maximum kept against the ground truth.
- **center**: always (0.5, 0.5).
- **smartcrop**: the focus point as Alinea derives it today, which is the
  centre of [smartcrop.js](https://github.com/jwagner/smartcrop.js)'s best
  100×100 crop.

`metrics.ts` can also model CSS `object-position` cropping (`mode: 'css'`),
but this isn't scored by default.

## Tools

- `run.ts`: the benchmark. `--limit N` samples N images evenly, `--methods`
  picks which methods to run (`model` is the learned model as shipped), and
  `--maps <dir>` scores precomputed maps (from `train/predict.py`) through
  the library's selection and placement.
- `faces.ts`, `marks.ts`: the face and hand-marked checks above.
- `report.ts`: an HTML progress report with images, for the training work.
- `tune.ts`: a coordinate descent over the parameters. It tunes on even-numbered
  images and reports on odd-numbered ones. `--peak` and `--head` set how much
  peak and head weigh against kept, `--guard` rejects settings whose kept
  drops more than that below the heuristic on any set, and `--maps` tunes
  only placement, for a model's maps.
- `failures.ts`: renders the images that lose the most against the oracle (red is the prediction, green the oracle point).
- `debug.ts`: renders the saliency map and focus point for any images.

The defaults were tuned with `--peak 0.3` on the even-numbered images of SALICON,
ECSSD and PASCAL-S. DUTS-TE and MSRA10K were held out completely.
