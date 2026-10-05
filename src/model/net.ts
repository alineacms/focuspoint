import {resize} from './resize.ts'
import {biasScales, bits, weights, widths} from './weights.ts'

/** Channel-planar feature map */
interface Tensor {
  c: number
  size: number
  data: Float32Array
}

interface Layer {
  cin: number
  cout: number
  k: number
  stride: number
  depthwise: boolean
  relu: boolean
  w: Float32Array
  b: Float32Array
}

type Spec = [cin: number, cout: number, k: number, stride: number, depthwise?: boolean, relu?: boolean]

/** Layers in the order train/export.py writes them, see train/model.py */
function specs(widths: ReadonlyArray<number>): Array<Spec> {
  const [a, b, c, d] = widths as [number, number, number, number]
  const block = (cin: number, cout: number, stride = 1): Array<Spec> => [
    [cin, cin, 3, stride, true],
    [cin, cout, 1, 1]
  ]
  return [
    [5, a, 3, 2],
    ...block(a, b, 2),
    ...block(b, b),
    ...block(b, c, 2),
    ...block(c, c),
    ...block(c, d, 2),
    ...block(d, d),
    [d, c, 1, 1],
    [d + c, c, 1, 1],
    ...block(c, c),
    [c + b, b, 1, 1],
    ...block(b, b),
    [b + a, a, 1, 1],
    ...block(a, a),
    [a, 1, 1, 1, false, false]
  ]
}

const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

/**
 * The weights are text, one base64url character per value, since that
 * compresses much better than a packed bitstream. Per layer: zigzagged
 * `bits`-bit weights, a scale per output channel in quarter-octave steps,
 * then a 12-bit bias per output channel as two characters. Batch norm is
 * already folded in. See train/export.py.
 */
function decode(): Array<Layer> {
  const code = new Uint8Array(128)
  for (let i = 0; i < 64; i++) code[alphabet.charCodeAt(i)] = i
  let at = 0
  const next = () => code[weights.charCodeAt(at++)]!
  const signed = (z: number) => (z & 1 ? -(z + 1) / 2 : z / 2)
  const layers = specs(widths).map(([cin, cout, k, stride, depthwise = false, relu = true], i) => {
    const per = (depthwise ? 1 : cin) * k * k
    const w = new Float32Array(cout * per)
    for (let j = 0; j < w.length; j++) w[j] = signed(next())
    for (let o = 0; o < cout; o++) {
      const scale = 2 ** ((next() - 56) / 4)
      for (let j = 0; j < per; j++) w[o * per + j]! *= scale
    }
    const b = new Float32Array(cout)
    for (let o = 0; o < cout; o++) b[o] = signed((next() << 6) | next()) * biasScales[i]!
    return {cin, cout, k, stride, depthwise, relu, w, b}
  })
  if (at !== weights.length || bits > 6) throw new Error('focuspoint: model weights do not match')
  return layers
}

let model: Array<Layer> | undefined

/** 1x1 convolution over the channels of all inputs, concatenated */
function pointwise(l: Layer, inputs: Array<Tensor>): Tensor {
  const size = inputs[0]!.size
  const hw = size * size
  const out = new Float32Array(l.cout * hw)
  for (let o = 0; o < l.cout; o++) {
    const row = out.subarray(o * hw, (o + 1) * hw)
    row.fill(l.b[o]!)
    let ci = 0
    for (const x of inputs)
      for (let c = 0; c < x.c; c++, ci++) {
        const w = l.w[o * l.cin + ci]!
        const src = x.data.subarray(c * hw, (c + 1) * hw)
        for (let p = 0; p < hw; p++) row[p]! += w * src[p]!
      }
    if (l.relu) for (let p = 0; p < hw; p++) if (row[p]! < 0) row[p] = 0
  }
  return {c: l.cout, size, data: out}
}

/** 3x3 convolution with zero padding, depthwise or over all channels */
function conv3(l: Layer, x: Tensor): Tensor {
  const {size} = x
  const s = l.stride
  const out = Math.floor((size - 1) / s) + 1
  const n = out * out
  // Pad every channel once so the taps need no bounds checks
  const p = size + 2
  const pp = p * p
  const padded = new Float32Array(x.c * pp)
  for (let c = 0; c < x.c; c++)
    for (let y = 0; y < size; y++)
      padded.set(x.data.subarray((c * size + y) * size, (c * size + y + 1) * size), c * pp + (y + 1) * p + 1)
  const data = new Float32Array(l.cout * n)
  const groups = l.depthwise ? 1 : x.c
  for (let o = 0; o < l.cout; o++) {
    const acc = data.subarray(o * n, (o + 1) * n)
    acc.fill(l.b[o]!)
    for (let g = 0; g < groups; g++) {
      const base = (l.depthwise ? o : g) * pp
      const k = (o * groups + g) * 9
      const w = l.w
      const w0 = w[k]!, w1 = w[k + 1]!, w2 = w[k + 2]!
      const w3 = w[k + 3]!, w4 = w[k + 4]!, w5 = w[k + 5]!
      const w6 = w[k + 6]!, w7 = w[k + 7]!, w8 = w[k + 8]!
      for (let oy = 0; oy < out; oy++) {
        const r0 = base + oy * s * p
        const r1 = r0 + p
        const r2 = r1 + p
        for (let ox = 0; ox < out; ox++) {
          const i = ox * s
          acc[oy * out + ox]! +=
            w0 * padded[r0 + i]! + w1 * padded[r0 + i + 1]! + w2 * padded[r0 + i + 2]! +
            w3 * padded[r1 + i]! + w4 * padded[r1 + i + 1]! + w5 * padded[r1 + i + 2]! +
            w6 * padded[r2 + i]! + w7 * padded[r2 + i + 1]! + w8 * padded[r2 + i + 2]!
        }
      }
    }
    if (l.relu) for (let i = 0; i < n; i++) if (acc[i]! < 0) acc[i] = 0
  }
  return {c: l.cout, size: out, data}
}

function up(x: Tensor): Tensor {
  const {c, size} = x
  const hw = size * size
  const data = new Float32Array(c * hw * 4)
  for (let i = 0; i < c; i++)
    data.set(resize(x.data.subarray(i * hw, (i + 1) * hw), size, size, size * 2, size * 2), i * hw * 4)
  return {c, size: size * 2, data}
}

/**
 * Run the network on a square sRGB thumbnail (planar, 0..1) and return the
 * logits of the importance map, at half the thumbnail's size.
 */
export function infer(rgb: Float32Array, size: number): Float32Array {
  const layers = (model ??= decode())
  let next = 0
  const layer = () => layers[next++]!
  const block = (x: Tensor) => {
    const d = conv3(layer(), x)
    return pointwise(layer(), [d])
  }
  const hw = size * size
  // Inputs scaled to -1..1, plus x and y coordinates
  const input = new Float32Array(5 * hw)
  for (let i = 0; i < 3 * hw; i++) input[i] = rgb[i]! * 2 - 1
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      input[3 * hw + y * size + x] = ((x + 0.5) / size) * 2 - 1
      input[4 * hw + y * size + x] = ((y + 0.5) / size) * 2 - 1
    }
  const s = conv3(layer(), {c: 5, size, data: input})
  const f1 = block(block(s))
  const f2 = block(block(f1))
  const f3 = block(block(f2))
  const mean = new Float32Array(f3.c)
  const n3 = f3.size * f3.size
  for (let c = 0; c < f3.c; c++) {
    let sum = 0
    for (let p = 0; p < n3; p++) sum += f3.data[c * n3 + p]!
    mean[c] = sum / n3
  }
  const g = pointwise(layer(), [{c: f3.c, size: 1, data: mean}])
  let y = block(pointwise(layer(), [up(f3), f2]))
  const n2 = y.size * y.size
  for (let c = 0; c < y.c; c++) for (let p = 0; p < n2; p++) y.data[c * n2 + p]! += g.data[c]!
  y = block(pointwise(layer(), [up(y), f1]))
  y = block(pointwise(layer(), [up(y), s]))
  return pointwise(layer(), [y]).data
}
