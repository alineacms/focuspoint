# Training the importance model

A small network that predicts where the important parts of a photo are,
shipped as `@alinea/focuspoint/model`. It replaces only the heuristic's
importance map; subject selection and crop placement (`locate()` in
`src/focus.ts`) are shared.

## Data

| Set | Images | Labels | Licence | Used for |
|---|---|---|---|---|
| SALICON train | 10,000 (COCO train2014) | mouse-tracking attention | research use; COCO images under their Flickr licences | training |
| DUTS-TR | 10,553 | subject masks | research use | training |
| Open Images V7 validation | 29,972 | none, labelled by the teacher | CC BY 2.0 | distillation |

Every 20th image of SALICON train and DUTS-TR is held out (`*-VAL`) for
early stopping and for tuning placement. The evaluation sets (SALICON val,
DUTS-TE, ECSSD, PASCAL-S, MSRA10K) and the hand-marked points
(`site/labels.json`) are never used for training, selection or tuning.

The teacher is a MobileNetV3-Large backbone (torchvision, ImageNet weights,
BSD-3-Clause) with a small top-down decoder, about 3M parameters. Its weights
are not shipped.

## Pipeline

```sh
cd train
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
../eval/fetch.sh SALICON-TR DUTS-TR OPENIMAGES-U   # from the repo root: eval/fetch.sh ...
python split.py                                    # link the *-VAL slices
python prep.py ../eval/data/{SALICON-TR,DUTS-TR,SALICON-TR-VAL,DUTS-TR-VAL,OPENIMAGES-U}
python prep.py --size 256 ../eval/data/{SALICON-TR,DUTS-TR,SALICON-TR-VAL,DUTS-TR-VAL,OPENIMAGES-U}

# Teacher, then its maps as targets for every training image
python train.py teacher --arch teacher --cache 256 --input 192 --epochs 12 --kept 1 --lr 0.001 --batch 64
python label.py teacher OPENIMAGES-U
# SALICON-TR-T and DUTS-TR-T are copies of the 128 px caches (and links to
# the 256 px ones) whose targets the teacher replaces
python label.py teacher SALICON-TR-T DUTS-TR-T

# Heads: detect faces, raise them to full importance in copies of the targets
python faces.py ../eval/data/{SALICON-TR,DUTS-TR,OPENIMAGES-U}
python boost.py SALICON-TR-T:../eval/data/SALICON-TR DUTS-TR-T:../eval/data/DUTS-TR OPENIMAGES-U:../eval/data/OPENIMAGES-U

# Student: distilled, then heads first, then quantisation-aware at 4 bits
python train.py student --epochs 40 --peak 1 --kept 0.5 --widths 16,32,64,96 \
  --sources SALICON-TR-T,DUTS-TR-T,OPENIMAGES-U --val SALICON-TR,DUTS-TR
python train.py student-heads --init student --epochs 15 --lr 0.0015 --peak 1 --kept 0.5 --head 1 \
  --sources SALICON-TR-T-F,DUTS-TR-T-F,OPENIMAGES-U-F --val SALICON-TR,DUTS-TR
python train.py student-q4 --init student-heads --qat 4 --epochs 4 --lr 0.0004 --peak 1 --kept 0.5 --head 1 \
  --sources SALICON-TR-T-F,DUTS-TR-T-F,OPENIMAGES-U-F --val SALICON-TR,DUTS-TR
python export.py student-q4 --bits 4        # writes src/model/weights.ts and the parity fixture
```

Then, from the repo root:

```sh
bun test                                     # includes the PyTorch/TS parity test
python train/predict.py student-q4 SALICON-TR-VAL DUTS-TR-VAL --checkpoint train/runs/student-q4/quantized.pt
bun eval/tune.ts --maps eval/predictions/student-q4 --peak 0.4 --head 0.4 --guard 0.015 \
  eval/data/SALICON-TR-VAL eval/data/DUTS-TR-VAL   # placement defaults for the model
bun eval/run.ts eval/data/SALICON eval/data/DUTS-TE --methods oracle,center,focuspoint,model
```

`predict.py` writes maps for `eval/run.ts --maps`, which scores any map
through the library's own selection and placement. `report.ts` renders a
visual progress report. `sizes.py` compares weight encodings by compressed
size.

## Model

- Input: the image squashed to 64×64 sRGB, averaged in linear light, plus
  two coordinate channels so the centre and edge priors are cheap to learn.
- A stem convolution, then depthwise-separable blocks down to 4×4, a global
  context vector, and a decoder with skip connections up to a 32×32 map.
  Only 3×3 convolutions (full and depthwise), 1×1 convolutions, ReLU, 2×
  bilinear upsampling and a global mean, so the TypeScript port in
  `src/model/net.ts` is a few loops.
- Output: logits; a softmax over positions is the importance map.

## Loss

KL divergence to the normalised target, plus differentiable versions of the
crop metrics: for each container shape, crop positions are soft-selected by
how much of the predicted map they keep (as `src/place.ts` places them), and
scored on the target's most important spot (`--peak`), on all of it
(`--kept`), or on its whole top region (`--head`).

## Weights

Weights are stored as text, one base64url character per value: zigzagged
4-bit integers per weight, a quarter-octave log scale per output channel, and
12-bit biases. That compresses much better than a packed bitstream, because
gzip and brotli can exploit how values cluster around zero
(`python sizes.py <run>`). Quantisation-aware fine-tuning trains against this
exact scheme, so the quantised model scores the same as the float one.

## Results

The shipped model is the 55k-parameter student (widths 16, 32, 64, 96;
64 px input), distilled from the teacher on all three sources with
face-boosted targets (`faces.py`, `boost.py`, `--head 1`), then fine-tuned
quantisation-aware at 4 bits. Its placement default, `protect: 0.5`, was
chosen on the validation slices. The held-out results are in the main
[README](../README.md#evaluation).

What mattered, in order:

1. **Distillation from a pretrained teacher.** Small networks trained on the
   human labels alone plateaued at about +0.5 kept over the heuristic,
   whatever their size or input resolution. The teacher (MobileNetV3
   backbone) sits near the oracle on the validation slices and agreed with
   every hand-marked point; training the student on its maps for 50,000
   images closed much of the gap.
2. **Heads first.** Students placed the point on the middle of a person or
   animal and let wide crops cut the head off. Detected heads raised to full
   importance in the targets, a loss that charges any crop cutting the top
   region, and `protect` in placement together lifted heads kept whole from
   about 61% (heuristic) to about 80% of crops.
3. **Text-encoded 4-bit weights with quantisation-aware training.** Packing
   bits tightly compresses worse than one character per weight; 4-bit
   weights after QAT score as well as 5-bit, and fit the 40 kB budget.

What did not help: a 96 px input (same results at 2.3× the compute), a
208k-parameter student trained from scratch, and asking a cheap vision LLM
for labels (see [eval/llm](../eval/llm/README.md)).
