# Task: train a tiny learned importance model for focuspoint

Paste this into Claude Code on a machine with a GPU, from the root of this repo,
or say "Follow TRAINING_PROMPT.md".

## Goal

`@alinea/focuspoint` finds the focus point of an image so that crops keep what
matters. Today it uses hand-written signals: minimum barrier distance
saliency, skin tone and a centre prior (`src/focus.ts` → `saliency()`). Those
signals have hit their ceiling. What remains is semantic, like knowing that a
face matters more than a bright sign or a horse's head more than its tail.

Train a **small learned model that predicts the importance map**, and ship it
as an optional entry point (`@alinea/focuspoint/model`). It replaces only the
saliency step. The rest of the pipeline stays: subject selection and the
crop-aware placement (`src/place.ts`), which turns any importance map into a
point.

Ship it only if it beats the current heuristic on the held-out sets within the
budget below. If it doesn't, report the results and stop. Don't ship a
regression.

## Budget

- Weights: **≤ 40 kB** gzipped. The current library is 3.6 kB, and gpu-lexer
  ships 41k weights at 6 bits in about 28 kB, so that's the reference point.
- Inference: **≤ 20 ms** per image in plain JS/TS on one CPU core in Node/Bun
  and browsers. No WebGPU, WASM or ONNX runtime, and no dependencies.
- The model input is a small RGB thumbnail (for example 64×64 or 96×96,
  aspect handled by padding or resizing; your choice, but document it). The
  output is a low-resolution importance map (for example 16×16 to 48×48) with
  values 0..1.

## Current state of the repo (read these first)

- `README.md` covers the algorithm, how points are applied and the current results.
- `eval/README.md` covers the datasets, metrics and tools.
- `src/focus.ts` has `saliency()` and `focusPoint()`. `src/place.ts` has the crop-aware placement.
- `eval/run.ts` is the benchmark (methods: `oracle`, `center`, `smartcrop`,
  `focuspoint`). `eval/metrics.ts` has the scoring. `eval/tune.ts` is the
  parameter search.
- `eval/fetch.sh` downloads the evaluation datasets into `eval/data/`.
- `demo/focuspoint-viewer.html` is a self-contained viewer (`bun run demo` rebuilds it).
- Tooling: Bun, TypeScript 7 (`bun run check`), `bun test`. Commit to `main`
  (repo convention).

### How points are scored

A point is applied by **centre-and-clamp** cropping: the crop window is
centred on the point and pushed back inside the image. For five container
shapes (3:1, 16:9, 1:1, 4:5, 9:16), take the largest window of that shape.

- **peak** (the main number, "top spot") is how often the most important
  spot stays inside. A focus point is one point, so what counts is that the
  thing people look at first survives every crop. When attention is split,
  for example between two people, maximising kept can put the point between
  them and let a tall crop cut off the face most people looked at. On flat
  subject masks the spot is the middle of the subject's solid part, which
  survives almost every crop (97% even for a fixed centre point on DUTS), so
  peak only discriminates on SALICON.
- **kept** is the share of ground-truth importance inside the window,
  averaged over the five shapes. It is a guard against regressions.
- **oracle** is the best single point per image, found by exhaustive search.

### Numbers to beat (kept / peak)

| | SALICON (5000) | DUTS-TE (5019) | ECSSD | PASCAL-S | MSRA10K (2000 sample) |
|---|---|---|---|---|---|
| oracle | 85.7 / 93.8 | 92.2 / 87.0 | 88.7 / 77.2 | 85.6 / 80.6 | 90.8 / 79.3 |
| **focuspoint (current)** | **82.9 / 87.8** | **86.4 / 83.5** | **85.1 / 77.8** | **80.9 / 78.4** | **88.1 / 79.2** |
| centre (0.5, 0.5) | 82.6 / 87.4 | 84.9 / 82.1 | 83.6 / 76.1 | 79.7 / 77.3 | 83.3 / 75.8 |

The biggest gap is on SALICON, which records where people actually looked,
and on DUTS-TE. Those two are the targets.

## Data rules (important)

The evaluation sets must stay unseen during training. **Never train or
early-stop on**:

- SALICON **val** (`eval/data/SALICON`, the 5000 val images)
- DUTS-**TE**
- MSRA10K
- ECSSD and PASCAL-S. These were used to tune the heuristic, so keep them out
  of training too. A single fixed validation split for early stopping is fine
  only if it comes from the training data below.

Training data you may use:

- **DUTS-TR**, 10,553 images with subject masks. The `beqooo09/SOD_Project`
  GitHub repo that `fetch.sh` uses for DUTS-TE also holds it. Add a `DUTS-TR`
  case to `eval/fetch.sh` (sparse checkout of that folder).
- **SALICON train**, 10,000 COCO train2014 images with attention maps, from
  the official SALICON site or mirrors. This is the most important source,
  because attention maps encode "face over tail". COCO images are on
  `s3.amazonaws.com/images.cocodataset.org/train2014/`.
- Optionally **pseudo-labels** from big teachers on unlabelled images (COCO
  train, Open Images): a salient-object model (U²-Net, BiRefNet or InSPyReNet),
  a face detector (YuNet or BlazeFace) to boost faces, and/or a saliency
  predictor (DeepGaze IIE, TranSalNet). This is the distillation recipe from
  Twitter's cropper and appwrite/focalnet. Check each teacher's licence and
  record it.
- Optionally FCDB, CPC or GAICD, human crop datasets, for an extra fine-tuning
  or evaluation signal. They were unreachable from the cloud sandbox but
  should download fine on a PC. If you use them, add a crop-overlap metric to
  `eval/` and keep their test splits held out.

Masks and attention maps measure different things. Train on both, so that
attention data teaches "what people look at" and masks teach "where the whole
subject is", or blend them into one target. Document the choice.

## Suggested approach (adapt as results dictate)

1. **Baseline and plumbing first:**
   - Write `train/` (Python, PyTorch, `requirements.txt`, `README.md`). It
     should load the training data at the chosen thumbnail size, with the
     same downscale as `src/image.ts` where it matters, and augment it with
     flips, crops, colour jitter and aspect changes.
   - Add an `eval` method that runs **PyTorch predictions through the existing
     JS pipeline**: dump predicted maps to `eval/predictions/<name>/*.png`,
     add a `--maps` option in `eval/run.ts` that feeds them to the same
     selection and placement code, and score them. This tells you whether a
     model is worth porting before you write any TS inference.
   - Check: feeding the oracle (the ground-truth maps) through that path
     should land close to the oracle row.
2. **Architecture:**
   - A tiny CNN: a few 3×3 or depthwise-separable conv blocks with stride-2
     downsampling, 8–32 channels, and a small decoder or a direct 1×1 head.
   - Target 20–50k parameters. Inputs with coordinates (CoordConv) can learn
     the centre and edge priors cheaply.
   - Count MACs; for 20 ms in JS stay around ≤ 20M.
3. **Loss:** per-pixel KL or BCE on normalised maps, plus optionally a
   differentiable version of "kept" (soft crop windows around the
   soft-argmax point) so the model optimises what is scored.
4. **Quantise:** int8, or int6 like gpu-lexer, with per-channel scales, after
   quantisation-aware fine-tuning. Re-score after quantising; that is the
   number that counts.
5. **Port:**
   - `src/model/` gets the weights as a compact base64 or `Uint8Array` module
     and plain-TS inference (conv, depthwise, activation, upsample).
   - It exports `modelSaliency(image)` with the same `SaliencyMap` shape as
     `saliency()`, and `focusPoint(image, {saliency: modelSaliency})`, or a
     separate `focusPointModel()`. Pick the cleaner API.
   - Add a unit test that the TS output matches PyTorch within tolerance on a
     few fixed images.
   - Add the package export `./model`.
6. **Evaluate:**
   - Add a `model` method to `eval/run.ts`, then run all five sets with the
     baselines.
   - Re-tune only the placement and selection parameters for the model's map
     with `eval/tune.ts`, on its train split. Don't tune on the held-out sets.
   - Report size (`bun run size`, extended to the model entry) and speed
     (`scripts/bench.ts`).
7. **Ship or stop:**
   - Ship if **peak** improves on SALICON by a clear margin (at least about
     +1 point), does not drop on the mask sets, and **kept** drops by no more
     than 1.5 on any set, within budget. Head-first placement (keeping the
     face people look at rather than the middle of the whole subject) is
     preferred over whole-subject coverage, which is why the kept guard is
     loose.
   - Also report peak on the hand-marked ideal points (`site/labels.json`,
     `bun eval/marks.ts`). They are a test set: never tune on them.
   - When shipping, update both READMEs (method, data sources, licences,
     results table) and add a model toggle to the viewer (`demo/`).
   - Otherwise, write up what was tried and the numbers in
     `train/README.md`, and leave the model out of the package.

## Deliverables

- `train/`: reproducible training (data prep, training, quantisation, export)
  with a README that states data sources, licences and exact commands.
- `src/model/` with its inference, tests and package export (only if shipped).
- Eval results for all five datasets against oracle, centre, smartcrop and the
  current heuristic, as markdown tables in `README.md`.
- Commits on `main` with clear messages. Don't commit datasets, teacher
  weights or training checkpoints (add them to `.gitignore`).
