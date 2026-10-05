# LLM focus labels

An experiment to grow the training set cheaply: a vision LLM marks what
matters in each photo, and the marks become an importance map. Before such
labels are trusted they are scored against real human attention, on images
that have SALICON maps.

## Sets

| Set | Images | Ground truth | Labels |
|---|---|---|---|
| SALICON-LLM | 300 SALICON **train** images (every 33rd) | mouse-tracking attention | Haiku 4.5 (299), Sonnet (40) |
| OPENIMAGES-LLM | 200 Open Images V7 validation photos, 20 per category | none | Haiku 4.5 |

The Open Images categories are portrait, people, food, architecture,
interior, landscape, animal, product, vehicle and fashion, picked from the
human-verified image labels to resemble what a CMS holds. The photos are
CC BY 2.0. Authors and source pages are listed in `openimages.csv`. No image
comes from an evaluation set.

```sh
eval/fetch.sh LLM    # both sets into eval/data
```

## Method

1. `render.ts` draws an 8×8 grid of named cells (A1…H8) on each image.
2. A model reads `PROMPT.md`, looks at each image and writes
   `{"name", "subject", "cells": {"C4": 1, "C5": 0.6}, "point": "C4"}`. The
   labels here came from Claude Code subagents (Haiku 4.5, and Sonnet for
   the comparison), 20 images per agent, with no code run on the images.
3. `labels.ts` turns the weighted cells into a soft map.
4. `score.ts` scores the labels with the usual metrics, next to centre,
   focuspoint and the oracle:
   - `llm-point` is the model's own point.
   - `llm-map` is the label map through the library's crop placement.
   - `llm-blend` is the label map averaged with the heuristic's map.

```sh
bun eval/llm/render.ts eval/out/llm/cells/SALICON-LLM eval/data/SALICON-LLM/images/*.jpg
bun eval/llm/score.ts eval/data/SALICON-LLM eval/llm/labels/SALICON-LLM.haiku.jsonl
bun eval/llm/show.ts eval/out/llm/show eval/data/OPENIMAGES-LLM eval/llm/labels/OPENIMAGES-LLM.haiku.jsonl
bun eval/llm/worst.ts eval/data/SALICON-LLM eval/llm/labels/SALICON-LLM.haiku.jsonl
```

## Results

SALICON-LLM, kept and peak as in [../README.md](../README.md):

| 299 images | peak | kept |
|---|---|---|
| oracle | 92.4% | 85.4% |
| focuspoint | 88.0% | 82.9% |
| centre | 87.5% | 82.7% |
| Haiku, blend | 87.7% | 82.2% |
| Haiku, map | 87.1% | 81.6% |
| *ideal cell labels* | *93.4%* | *84.3%* |

"Ideal cell labels" (`ideal-cells`) are made from the attention maps
themselves: every cell with at least 30% of the peak cell's attention,
weighted by it. They show that the label format can carry most of the
oracle's lead. What falls short is the labelling.

The same 40 images labelled by both models:

| 40 images | peak | kept |
|---|---|---|
| centre | 87.0% | 82.9% |
| focuspoint | 83.0% | 82.7% |
| Haiku, map | 83.5% | 81.7% |
| Sonnet, map | 89.0% | 82.0% |

## Findings

- **Coordinates fail.** A first prompt asked for boxes in 0–100 units, read
  off a ruled grid. Haiku picked sensible subjects but placed the boxes
  10–30 units off. On a 30-image pilot it scored 83.8% kept, against 83.9%
  for centre. Named cells fixed most of that.
- **Haiku is still off by a cell or two.** On the dog-and-bike photo it
  marked B6 for a head in D5, and it sometimes picks a different subject
  than people looked at (fries over a hot dog). It wins against centre on 72
  images and loses on 145, so its labels are not good enough to train on.
- **Sonnet keeps what people look at most.** Its peak is 6 points above
  Haiku and above every baseline, but its kept doesn't beat centre.
  SALICON attention is spread out and centre-biased, so a map that sits on
  one subject loses kept even when it is right about the subject.
- **Agents occasionally write broken JSON** (2 lines in 500) or skip an
  image (1 in 500). `readLabels` names the bad line.
