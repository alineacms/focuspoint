import {focusPoint, saliency, type FocusPoint} from '../src/index.ts'

/** Container shapes previewed, matching the evaluation. */
const shapes = [
  {name: '3:1', label: 'Banner', aspect: 3},
  {name: '16:9', label: 'Wide', aspect: 16 / 9},
  {name: '1:1', label: 'Square', aspect: 1},
  {name: '4:5', label: 'Portrait', aspect: 4 / 5},
  {name: '9:16', label: 'Story', aspect: 9 / 16}
]

interface Rect {
  x: number
  y: number
  width: number
  height: number
}

interface State {
  image: ImageBitmap
  name: string
  point: FocusPoint
  heat: HTMLCanvasElement
  ms: number
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const stage = $<HTMLCanvasElement>('stage')
const fileInput = $<HTMLInputElement>('file')
const drop = $<HTMLElement>('drop')
const showHeat = $<HTMLInputElement>('show-heat')
const showBox = $<HTMLInputElement>('show-box')
const compare = $<HTMLInputElement>('compare')
const crops = $<HTMLElement>('crops')
const readout = $<HTMLElement>('readout')
const message = $<HTMLElement>('message')

let state: State | undefined
let highlight: number | undefined

/** Largest crop of `aspect` centred on the point and clamped to the image. */
function centreClamp(p: {x: number; y: number}, w: number, h: number, aspect: number): Rect {
  let cw = w
  let ch = w / aspect
  if (ch > h) {
    ch = h
    cw = h * aspect
  }
  const x = Math.min(w - cw, Math.max(0, p.x * w - cw / 2))
  const y = Math.min(h - ch, Math.max(0, p.y * h - ch / 2))
  return {x, y, width: cw, height: ch}
}

function cssColor(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/** Turbo-like ramp, transparent at zero so the photo shows through. */
function heatColor(v: number): [number, number, number, number] {
  const stops: Array<[number, number, number, number]> = [
    [0, 30, 60, 160],
    [0.35, 40, 170, 200],
    [0.6, 250, 220, 60],
    [1, 240, 60, 40]
  ]
  let i = 0
  while (i < stops.length - 2 && v > stops[i + 1]![0]) i++
  const [a, b] = [stops[i]!, stops[i + 1]!]
  const t = Math.min(1, Math.max(0, (v - a[0]) / (b[0] - a[0])))
  const mix = (k: 1 | 2 | 3) => Math.round(a[k] + (b[k] - a[k]) * t)
  return [mix(1), mix(2), mix(3), Math.round(40 + 190 * v)]
}

// The model looks at a 64 px thumbnail, so 256 px leaves enough to average
function pixels(image: ImageBitmap): ImageData {
  const scale = Math.min(1, 256 / Math.max(image.width, image.height))
  const w = Math.max(1, Math.round(image.width * scale))
  const h = Math.max(1, Math.round(image.height * scale))
  const src = document.createElement('canvas')
  src.width = w
  src.height = h
  const sctx = src.getContext('2d', {willReadFrequently: true})!
  sctx.drawImage(image, 0, 0, w, h)
  return sctx.getImageData(0, 0, w, h)
}

function heatmap(data: ImageData): HTMLCanvasElement {
  const map = saliency(data)
  const out = document.createElement('canvas')
  out.width = map.width
  out.height = map.height
  const octx = out.getContext('2d')!
  const img = octx.createImageData(map.width, map.height)
  for (let i = 0; i < map.data.length; i++) {
    const [r, g, b, a] = heatColor(map.data[i]!)
    img.data.set([r, g, b, a], i * 4)
  }
  octx.putImageData(img, 0, 0)
  return out
}

function drawMarker(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
  ctx.save()
  ctx.lineWidth = Math.max(2, r / 6)
  ctx.strokeStyle = 'rgba(0,0,0,0.55)'
  ctx.beginPath()
  ctx.arc(x, y, r + 1.5, 0, Math.PI * 2)
  ctx.stroke()
  ctx.strokeStyle = color
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.moveTo(x - r * 1.8, y)
  ctx.lineTo(x - r * 0.45, y)
  ctx.moveTo(x + r * 0.45, y)
  ctx.lineTo(x + r * 1.8, y)
  ctx.moveTo(x, y - r * 1.8)
  ctx.lineTo(x, y - r * 0.45)
  ctx.moveTo(x, y + r * 0.45)
  ctx.lineTo(x, y + r * 1.8)
  ctx.stroke()
  ctx.restore()
}

function renderStage() {
  if (!state) return
  const {image, point} = state
  const dpr = window.devicePixelRatio || 1
  const maxW = stage.parentElement!.clientWidth
  const maxH = Math.max(260, window.innerHeight * 0.62)
  const scale = Math.min(maxW / image.width, maxH / image.height, 1.5)
  const w = Math.round(image.width * scale)
  const h = Math.round(image.height * scale)
  stage.style.width = `${w}px`
  stage.style.height = `${h}px`
  stage.width = Math.round(w * dpr)
  stage.height = Math.round(h * dpr)
  const ctx = stage.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(image, 0, 0, w, h)
  if (showHeat.checked) {
    ctx.save()
    ctx.globalAlpha = 0.7
    ctx.imageSmoothingEnabled = true
    ctx.drawImage(state.heat, 0, 0, w, h)
    ctx.restore()
  }
  if (highlight !== undefined) {
    const r = centreClamp(point, w, h, shapes[highlight]!.aspect)
    ctx.save()
    ctx.fillStyle = 'rgba(10,12,20,0.55)'
    ctx.beginPath()
    ctx.rect(0, 0, w, h)
    ctx.rect(r.x, r.y, r.width, r.height)
    ctx.fill('evenodd')
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1.5
    ctx.strokeRect(r.x + 0.75, r.y + 0.75, r.width - 1.5, r.height - 1.5)
    ctx.restore()
  }
  if (showBox.checked) {
    const b = point.box
    ctx.save()
    ctx.setLineDash([6, 4])
    ctx.lineWidth = 1.5
    ctx.strokeStyle = cssColor('--box')
    ctx.strokeRect(b.x * w, b.y * h, b.width * w, b.height * h)
    ctx.restore()
  }
  if (compare.checked) drawMarker(ctx, w / 2, h / 2, 9, cssColor('--centre'))
  drawMarker(ctx, point.x * w, point.y * h, 11, cssColor('--mark'))
}

function renderCrops() {
  if (!state) return
  const {image, point} = state
  crops.replaceChildren()
  shapes.forEach((shape, index) => {
    const figure = document.createElement('figure')
    figure.className = 'crop'
    figure.tabIndex = 0
    const row = document.createElement('div')
    row.className = 'crop-row'
    const variants = compare.checked
      ? [
          {who: 'Focus point', p: point, color: '--mark'},
          {who: 'Centre', p: {x: 0.5, y: 0.5}, color: '--centre'}
        ]
      : [{who: 'Focus point', p: point, color: '--mark'}]
    for (const v of variants) {
      const r = centreClamp(v.p, image.width, image.height, shape.aspect)
      const c = document.createElement('canvas')
      const dpr = window.devicePixelRatio || 1
      const targetH = 132
      const cw = Math.round(Math.min(targetH * shape.aspect, 300))
      const ch = Math.round(cw / shape.aspect)
      c.style.width = `${cw}px`
      c.style.height = `${ch}px`
      c.width = Math.round(cw * dpr)
      c.height = Math.round(ch * dpr)
      c.setAttribute('aria-label', `${shape.label} crop using the ${v.who.toLowerCase()}`)
      c.className = 'crop-canvas'
      c.style.setProperty('--edge', `var(${v.color})`)
      const ctx = c.getContext('2d')!
      ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(image, r.x, r.y, r.width, r.height, 0, 0, c.width, c.height)
      row.append(c)
    }
    const caption = document.createElement('figcaption')
    caption.innerHTML = `<span>${shape.label}</span><span class="mono">${shape.name}</span>`
    figure.append(row, caption)
    const on = () => {
      highlight = index
      renderStage()
    }
    const off = () => {
      highlight = undefined
      renderStage()
    }
    figure.addEventListener('mouseenter', on)
    figure.addEventListener('mouseleave', off)
    figure.addEventListener('focus', on)
    figure.addEventListener('blur', off)
    crops.append(figure)
  })
}

function renderReadout() {
  if (!state) return
  const {point, image, ms, name} = state
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`
  readout.innerHTML = `
    <div><dt>Image</dt><dd title="${name}">${name}</dd></div>
    <div><dt>Size</dt><dd class="mono">${image.width} × ${image.height}</dd></div>
    <div><dt>Focus x</dt><dd class="mono">${point.x.toFixed(3)} <small>${pct(point.x)}</small></dd></div>
    <div><dt>Focus y</dt><dd class="mono">${point.y.toFixed(3)} <small>${pct(point.y)}</small></dd></div>
    <div><dt>Confidence</dt><dd class="mono">${point.confidence.toFixed(2)}</dd></div>
    <div><dt>Time</dt><dd class="mono">${ms.toFixed(1)} ms</dd></div>`
}

function render() {
  renderStage()
  renderCrops()
  renderReadout()
}

async function analyse(image: ImageBitmap, name: string) {
  const t = performance.now()
  const data = pixels(image)
  const point = focusPoint(data)
  const ms = performance.now() - t
  state = {image, name, point, ms, heat: heatmap(data)}
  message.hidden = true
  render()
}

async function openFile(file: File | undefined) {
  if (!file) return
  if (!file.type.startsWith('image/')) {
    showMessage(`${file.name} isn't an image. Choose a JPEG, PNG, WebP or GIF file.`)
    return
  }
  try {
    const image = await createImageBitmap(file, {imageOrientation: 'from-image'})
    await analyse(image, file.name)
  } catch {
    showMessage(`${file.name} couldn't be decoded by this browser. Try a JPEG or PNG.`)
  }
}

function showMessage(text: string) {
  message.textContent = text
  message.hidden = false
}

/** A generated example scene: a balloon high on the right of a wide landscape. */
async function sample(): Promise<ImageBitmap> {
  const w = 1500
  const h = 1000
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')!
  const sky = ctx.createLinearGradient(0, 0, 0, h * 0.7)
  sky.addColorStop(0, '#7fb2dd')
  sky.addColorStop(1, '#d9e8ef')
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, w, h)
  // Soft clouds
  ctx.fillStyle = 'rgba(255,255,255,0.55)'
  for (const [cx, cy, r] of [
    [260, 180, 70],
    [330, 160, 90],
    [420, 190, 60],
    [820, 300, 50],
    [880, 285, 70]
  ] as const) {
    ctx.beginPath()
    ctx.ellipse(cx, cy, r * 1.6, r * 0.7, 0, 0, Math.PI * 2)
    ctx.fill()
  }
  // Hills
  const hill = (base: number, amp: number, color: string, phase: number) => {
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.moveTo(0, h)
    for (let x = 0; x <= w; x += 10)
      ctx.lineTo(x, base + Math.sin(x / 210 + phase) * amp + Math.sin(x / 77 + phase * 2) * amp * 0.25)
    ctx.lineTo(w, h)
    ctx.fill()
  }
  hill(640, 40, '#8aa58a', 0.3)
  hill(720, 30, '#6f8f5e', 1.7)
  hill(820, 22, '#5b7c47', 2.9)
  // Field rows
  ctx.strokeStyle = 'rgba(40,60,30,0.25)'
  ctx.lineWidth = 3
  for (let i = 0; i < 14; i++) {
    ctx.beginPath()
    ctx.moveTo(-200 + i * 140, h)
    ctx.lineTo(w / 2 + (i - 7) * 30, 840)
    ctx.stroke()
  }
  // Balloon near the top right edge
  const bx = 1180
  const by = 210
  const env = ctx.createRadialGradient(bx - 40, by - 60, 20, bx, by, 170)
  env.addColorStop(0, '#ff8a5c')
  env.addColorStop(1, '#c2311f')
  ctx.fillStyle = env
  ctx.beginPath()
  ctx.moveTo(bx, by + 170)
  ctx.bezierCurveTo(bx - 210, by + 40, bx - 150, by - 160, bx, by - 160)
  ctx.bezierCurveTo(bx + 150, by - 160, bx + 210, by + 40, bx, by + 170)
  ctx.fill()
  ctx.strokeStyle = 'rgba(255,230,180,0.75)'
  ctx.lineWidth = 6
  for (const k of [-0.55, 0, 0.55]) {
    ctx.beginPath()
    ctx.moveTo(bx, by - 160)
    ctx.quadraticCurveTo(bx + k * 230, by, bx + k * 40, by + 160)
    ctx.stroke()
  }
  ctx.strokeStyle = '#4a3020'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(bx - 30, by + 160)
  ctx.lineTo(bx - 18, by + 215)
  ctx.moveTo(bx + 30, by + 160)
  ctx.lineTo(bx + 18, by + 215)
  ctx.stroke()
  ctx.fillStyle = '#7a4a2a'
  ctx.fillRect(bx - 22, by + 212, 44, 32)
  return createImageBitmap(c)
}

fileInput.addEventListener('change', () => openFile(fileInput.files?.[0]))
for (const el of [showHeat, showBox, compare]) el.addEventListener('change', render)
drop.addEventListener('dragover', e => {
  e.preventDefault()
  drop.classList.add('over')
})
drop.addEventListener('dragleave', () => drop.classList.remove('over'))
drop.addEventListener('drop', e => {
  e.preventDefault()
  drop.classList.remove('over')
  openFile(e.dataTransfer?.files[0])
})
window.addEventListener('paste', e => {
  const file = [...(e.clipboardData?.files ?? [])].find(f => f.type.startsWith('image/'))
  if (file) openFile(file)
})
let resize = 0
window.addEventListener('resize', () => {
  cancelAnimationFrame(resize)
  resize = requestAnimationFrame(renderStage)
})
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderStage)
new MutationObserver(renderStage).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']})

sample().then(image => analyse(image, 'Example: generated balloon scene'))
