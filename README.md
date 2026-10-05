# focuspoint

Find the focus point of an image: the (x, y) position that every crop
centred on it should keep, such as the face people look at first.

- **Two ways to find it**: a 4 kB heuristic, or a tiny learned model (35 kB)
  that knows a head matters more than a bright sign.
- **Fast**: about 0.5 ms per image for the heuristic, 4–6 ms for the model.
- **Runs anywhere**: browser, Node, Bun, Deno, workers. No dependencies, no
  WebGPU or WASM. It only needs RGBA pixels.

**Try it**: [alineacms.github.io/focuspoint](https://alineacms.github.io/focuspoint/)
shows 150 photos cropped live, with every setting adjustable.

```ts
import {focusPoint} from '@alinea/focuspoint/model' // or '@alinea/focuspoint'

const {x, y, box, confidence} = focusPoint({data, width, height})
// x, y: focus point, 0..1 from the top left
// box: bounding box of the subject, 0..1
// confidence: how clearly a subject stands out, 0..1
```

## Which one

| | Heuristic `@alinea/focuspoint` | Model `@alinea/focuspoint/model` |
|---|---|---|
| Size (min + gzip) | 3.9 kB | 34.9 kB |
| Time per image | ~0.5 ms | 4–6 ms |
| Keeps the main head whole (photos with faces) | 61% of crops | 79% of crops |
| Keeps the most-looked-at spot (SALICON) | 87.7% | 89.4% |

The model is the better choice whenever people or animals may be in the
picture. The heuristic is smaller and keeps slightly more of large, centred
subjects. Both expose the same API.

## Usage

### Browser

```ts
import {focusPointFromImage} from '@alinea/focuspoint/model/browser' // or '@alinea/focuspoint/browser'

const img = new Image()
img.src = '/photo.jpg'
await img.decode()
const {x, y} = focusPointFromImage(img)
```

`focusPointFromImage` accepts an `HTMLImageElement`, `HTMLCanvasElement`,
`HTMLVideoElement`, `ImageBitmap` or `OffscreenCanvas`. It draws the image onto
a small canvas so the browser does the downscaling.

### Node / Bun with sharp

Pass any `{data, width, height}` RGBA buffer. Decoding and shrinking the image
first keeps things fast. Around 256 px on the longest side is plenty.

```ts
import sharp from 'sharp'
import {focusPoint} from '@alinea/focuspoint/model'

const {data, info} = await sharp(file)
  .rotate()
  .resize(256, 256, {fit: 'inside'})
  .ensureAlpha()
  .raw()
  .toBuffer({resolveWithObject: true})
const point = focusPoint({data, width: info.width, height: info.height})
```

### Importance map

`saliency(image)` returns the importance map behind the point
(`{width, height, data: Float32Array}`, values 0..1), from the heuristic or
the model depending on the import. `locate(map, options)` turns any map into
a focus point, so you can also bring your own.

### Applying the point

The point is meant for **centre-and-clamp** cropping: centre the crop window
on the point, then shift it back inside the image where it would overflow.
This is what image CDNs and most crop tools do:

```ts
function crop(point, imageWidth, imageHeight, cropWidth, cropHeight) {
  const clamp = (v: number, max: number) => Math.min(max, Math.max(0, v))
  return {
    x: clamp(point.x * imageWidth - cropWidth / 2, imageWidth - cropWidth),
    y: clamp(point.y * imageHeight - cropHeight / 2, imageHeight - cropHeight),
    width: cropWidth,
    height: cropHeight
  }
}
```

In CSS, `object-position: x% y%` aligns proportionally rather than centring.
To get the same crop, convert the clamped window: `object-position` is
`cropX / (imageWidth - cropWidth)` (or 50% when nothing is cropped on that
axis).

## How it works

1. **Score how important each pixel is.**
   - The **heuristic** downscales to 48 px in linear light and combines how
     cut off each region is from the image border ([minimum barrier
     distance](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_Minimum_Barrier_Salient_ICCV_2015_paper.pdf)),
     skin tone, a centre prior and the alpha channel.
   - The **model** squashes the image to 64×64 and runs a 55k-parameter
     convolutional network (depthwise-separable encoder, global context,
     decoder with skips, 32×32 output). Its weights are 4-bit, stored as
     text. It was distilled from a larger network trained on where people
     look (SALICON), on subject masks (DUTS) and on 30,000 extra photos, with
     detected heads marked as the most important part. See
     [train/README.md](train/README.md).
2. **Find the subject**: threshold the map, take the strongest region, and
   settle on the centre of the importance around it.
3. **Protect the top spot** (model default): the area around the map's peak,
   usually a face, must stay whole in every crop. Only points that achieve
   that are considered; when none can (a face taller than a 3:1 banner), the
   one that cuts least of it wins. Set `protect: 0` to favour more of the
   whole subject instead.
4. **Place the point for cropping**: search x and y for the position that keeps
   the most importance in view when the image is cropped to common container
   shapes (3:1, 16:9, 1:1, 4:5, 9:16), each crop centred on the point and
   clamped. Positions within a small tolerance of the best count as equal,
   and the one closest to the subject wins.

## Evaluation

Every image is cropped to five container shapes (3:1, 16:9, 1:1, 4:5, 9:16)
around the point, with centre-and-clamp. All numbers are on held-out data:
SALICON val, DUTS-TE, ECSSD, PASCAL-S and MSRA10K were never used for
training or tuning the model. See [eval/README.md](eval/README.md).

**Heads kept whole**: on photos where a face detector finds a face, the share
of crops that keep the main head (forehead to chin, with hair) entirely in
view. Cutting through a head is the worst crop there is, so this comes
first. "Ideal" centres on the head itself.

| Main head kept | SALICON (1704) | DUTS-TE (952) | ECSSD (201) | PASCAL-S (149) |
|---|---|---|---|---|
| ideal | 98.5% | 97.1% | 94.3% | 94.8% |
| **model** | **79.3%** | **83.8%** | **77.7%** | **80.1%** |
| model, `protect: 0` | 77.2% | 81.9% | 75.3% | 78.4% |
| heuristic | 61.3% | 65.5% | 58.8% | 60.7% |
| centre | 59.0% | 63.7% | 57.1% | 58.3% |

**Top spot**: how often the single most important spot survives the crop. On
SALICON that is where most people looked. On the subject-mask sets it is the
middle of the subject, which the head-first model trades for the head.

| Top spot | SALICON | DUTS-TE | ECSSD | PASCAL-S | MSRA10K* |
|---|---|---|---|---|---|
| oracle | 93.8% | 99.4% | 99.3% | 97.9% | 99.7% |
| **model** | **89.4%** | 92.7% | 92.7% | 90.0% | 94.7% |
| model, `protect: 0` | 90.8% | 93.9% | 94.4% | 91.9% | 96.3% |
| heuristic | 87.7% | 95.3% | 97.9% | 95.7% | 98.7% |
| centre | 87.4% | 94.2% | 97.1% | 94.4% | 94.8% |
| smartcrop, as Alinea uses it | 86.6% | 93.1% | 94.0% | 92.8% | 95.5% |

**Kept**: how much of what matters stays in view, averaged over the crops.

| Kept | SALICON | DUTS-TE | ECSSD | PASCAL-S | MSRA10K* |
|---|---|---|---|---|---|
| oracle | 85.7% | 92.2% | 88.7% | 85.6% | 90.8% |
| **model** | 81.1% | 85.8% | 83.1% | 78.6% | 86.6% |
| model, `protect: 0` | 82.5% | 86.4% | 84.0% | 79.6% | 87.6% |
| heuristic | 82.9% | 86.4% | 85.1% | 80.9% | 88.1% |
| centre | 82.6% | 84.9% | 83.6% | 79.7% | 83.3% |
| smartcrop, as Alinea uses it | 81.9% | 84.3% | 82.0% | 78.8% | 85.0% |

\* MSRA10K: an evenly spaced sample of 2000 images.

**Hand-marked points**: on 55 of the website's photos, a person marked where
the focus should be. The model keeps that point in 93.8% of crops, against
84.4% for the heuristic and 83.6% for the centre.

"smartcrop, as Alinea uses it" is the centre of
[smartcrop.js](https://github.com/jwagner/smartcrop.js)'s best 100×100 crop.

**What the model trades**: protecting the head moves the point away from the
middle of large subjects, so it keeps 1–2 points less of the whole subject
than the heuristic. `protect: 0` keeps nearly as much as the heuristic and
still keeps far more heads.

## Viewer and website

- [`demo/focuspoint-viewer.html`](demo/focuspoint-viewer.html): open it in a
  browser to try your own images, with the importance map and crops.
- [`site/`](site): the website, built with `bun scripts/site.ts` and published
  to GitHub Pages on every push to `main`.

## Development

```sh
bun install
bun test          # unit tests, including the PyTorch/TypeScript parity test
bun run check     # typecheck (TypeScript 7)
bun run build     # emit dist/
bun run size      # bundle sizes
bun scripts/bench.ts <images...>  # speed
bun eval/run.ts <dataset-dir> --methods oracle,center,focuspoint,model
bun eval/faces.ts <dataset-dir>   # heads kept whole
bun scripts/site.ts --serve       # build and serve the website, with point marking
bun run demo      # rebuild demo/focuspoint-viewer.html
```

A dataset directory holds `images/*.jpg` and `masks/*.png` (or
`fixations/*.png`) with matching names. See [eval/README.md](eval/README.md)
for how to get the datasets, and [train/README.md](train/README.md) for
training the model.
