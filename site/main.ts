import {defaults as heuristicDefaults, locate, saliency, type FocusPoint, type Params, type SaliencyMap} from '../src/index.ts'
import {defaults as modelDefaults, saliency as modelSaliency} from '../src/model/index.ts'

interface Photo {
  id: string
  alt: string
  author: string
  page: string
  src: string
  width: number
  height: number
}

type Method = 'model' | 'heuristic' | 'centre'

interface Item {
  photo: Photo
  index: number
  frame: HTMLElement
  img: HTMLImageElement
  heat: HTMLCanvasElement
  af: HTMLElement
  ideal: HTMLElement
  window: HTMLElement
  box: HTMLElement
  time: HTMLElement
  data?: ImageData
  model?: {key: number; map: SaliencyMap; ms: number}
  heuristic?: {key: string; map: SaliencyMap; ms: number}
  point?: FocusPoint
  drawn?: SaliencyMap
}

const shapes = [
  {id: 'mixed', label: 'Mixed', aspect: 0},
  {id: '3-1', label: '3:1', aspect: 3},
  {id: '16-9', label: '16:9', aspect: 16 / 9},
  {id: '1-1', label: '1:1', aspect: 1},
  {id: '4-5', label: '4:5', aspect: 4 / 5},
  {id: '9-16', label: '9:16', aspect: 9 / 16},
  {id: 'original', label: 'Orig.', aspect: -1}
]
// Shapes cycled through in the mixed layout, so the sheet shows every crop
const mixed = [4 / 5, 16 / 9, 1, 9 / 16, 3, 1, 4 / 5, 16 / 9, 9 / 16]

interface Slider {
  key: keyof Params
  label: string
  min: number
  max: number
  step: number
  help: string
  heuristic?: boolean
}

const sliders: Array<Slider> = [
  {key: 'protect', label: 'Protect the face', min: 0, max: 0.9, step: 0.1, help: 'Keep the most important spot whole in every crop: the area around the peak above this share of it. 0 turns it off.'},
  {key: 'gamma', label: 'Peak focus', min: 1, max: 4, step: 0.5, help: 'Higher values pull the point towards the brightest part of the subject.'},
  {key: 'emphasis', label: 'Peak emphasis', min: 1, max: 4, step: 0.5, help: 'Higher values chase the single strongest spot when placing crops.'},
  {key: 'fit', label: 'Crop fit', min: 0, max: 1, step: 0.05, help: 'How far the point moves from the subject towards the spot that keeps the most in view.'},
  {key: 'tolerance', label: 'Tolerance', min: 0, max: 0.05, step: 0.005, help: 'Placements this close to the best count as equal. The one nearest the subject wins.'},
  {key: 'threshold', label: 'Split threshold', min: 0, max: 0.6, step: 0.05, help: 'Raise it to split touching subjects and keep only the strongest.'},
  {key: 'radius', label: 'Subject radius', min: 0, max: 0.4, step: 0.05, help: 'How far around the strongest region to look for the rest of the subject.'},
  {key: 'intensity', label: 'Small subjects', min: 0, max: 1, step: 0.05, help: 'At 0 the biggest region wins; at 1 the most intense one does.'},
  {key: 'mbd', label: 'Border cut-off', min: 0, max: 3, step: 0.25, help: 'Weight of regions that are cut off from the image border.', heuristic: true},
  {key: 'skin', label: 'Skin tone', min: 0, max: 1.5, step: 0.25, help: 'Weight of skin-coloured areas.', heuristic: true},
  {key: 'center', label: 'Centre bias', min: 0, max: 1, step: 0.05, help: 'How much the middle of the frame is preferred.', heuristic: true},
  {key: 'border', label: 'Border contrast', min: 0, max: 2, step: 0.25, help: 'Weight of colours that differ from the border.', heuristic: true}
]
const signals: Array<keyof Params> = ['size', 'mbd', 'border', 'skin', 'center', 'blur']

const state = {
  method: 'model' as Method,
  shape: 'mixed',
  view: 'crop' as 'crop' | 'full',
  point: true,
  heat: false,
  box: false,
  mark: false,
  params: {model: {...modelDefaults}, heuristic: {...heuristicDefaults}} as Record<'model' | 'heuristic', Params>
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const sheet = $<HTMLElement>('sheet')
const readout = $<HTMLElement>('readout')
const items: Array<Item> = []
// Ideal points marked by hand, by photo id, in 0..1 image coordinates
let labels: Record<string, {x: number; y: number}> = {}

const cross = `<svg viewBox="0 0 26 26" aria-hidden="true"><g stroke="var(--mark)" stroke-width="2.4" stroke-linecap="round"><path d="M13 2v7M13 17v7M2 13h7M17 13h7"/></g></svg>`
const reticle = `<svg viewBox="0 0 26 26" aria-hidden="true"><g fill="none" stroke="var(--pencil)" stroke-width="2.2"><path d="M1 8V1h7M18 1h7v7M25 18v7h-7M8 25H1v-7"/></g><circle cx="13" cy="13" r="2.2" fill="var(--pencil)"/></svg>`

function controls() {
  $('shapes').innerHTML = shapes
    .map(s => `<label><input type="radio" name="shape" id="shape-${s.id}" value="${s.id}"${s.id === state.shape ? ' checked' : ''}><span>${s.label}</span></label>`)
    .join('')
  $('sliders').innerHTML = sliders
    .map(
      s => `<div class="slider" data-key="${s.key}"${s.heuristic ? ' data-heuristic' : ''}>
  <label class="label" for="p-${s.key}">${s.label}<output id="o-${s.key}"></output></label>
  <input type="range" id="p-${s.key}" min="${s.min}" max="${s.max}" step="${s.step}">
  <small>${s.help}</small></div>`
    )
    .join('')
  document.addEventListener('change', e => {
    const t = e.target as HTMLInputElement
    if (t.name === 'method') state.method = t.value as Method
    else if (t.name === 'shape') state.shape = t.value
    else if (t.name === 'view') state.view = t.value as 'crop' | 'full'
    else if (t.id === 'show-point') state.point = t.checked
    else if (t.id === 'show-heat') state.heat = t.checked
    else if (t.id === 'show-box') state.box = t.checked
    else if (t.id === 'mark') {
      state.mark = t.checked
      // Marking is easiest on the whole photo
      if (state.mark) {
        state.view = 'full'
        $<HTMLInputElement>('view-full').checked = true
      }
      document.body.classList.toggle('marking', state.mark)
    } else return
    syncSliders()
    update()
  })
  $('sliders').addEventListener('input', e => {
    const t = e.target as HTMLInputElement
    const key = t.id.slice(2) as keyof Params
    if (state.method === 'centre') return
    state.params[state.method][key] = Number(t.value)
    $(`o-${key}`).textContent = t.value
    update()
  })
  $('copy').addEventListener('click', async () => {
    const text = JSON.stringify(labels, null, 1)
    try {
      await navigator.clipboard.writeText(text)
      $('copy').textContent = 'Copied'
    } catch {
      $('copy').textContent = 'Copy failed'
    }
    setTimeout(() => ($('copy').textContent = 'Copy marks'), 1500)
  })
  $('reset').addEventListener('click', () => {
    if (state.method === 'centre') return
    state.params[state.method] = {...(state.method === 'model' ? modelDefaults : heuristicDefaults)}
    syncSliders()
    update()
  })
  const drop = $<HTMLElement>('drop')
  $<HTMLInputElement>('file').addEventListener('change', e => addFiles((e.target as HTMLInputElement).files))
  drop.addEventListener('dragover', e => {
    e.preventDefault()
    drop.classList.add('over')
  })
  drop.addEventListener('dragleave', () => drop.classList.remove('over'))
  drop.addEventListener('drop', e => {
    e.preventDefault()
    drop.classList.remove('over')
    addFiles(e.dataTransfer?.files ?? null)
  })
  syncSliders()
}

function syncSliders() {
  const tuning = $<HTMLDetailsElement>('tuning')
  tuning.hidden = state.method === 'centre'
  if (state.method === 'centre') return
  const p = state.params[state.method]
  for (const el of document.querySelectorAll<HTMLElement>('.slider')) {
    const key = el.dataset.key as keyof Params
    el.hidden = el.hasAttribute('data-heuristic') && state.method !== 'heuristic'
    $<HTMLInputElement>(`p-${key}`).value = String(p[key])
    $(`o-${key}`).textContent = String(p[key])
  }
}

function tile(photo: Photo, own = false): Item {
  const figure = document.createElement('figure')
  figure.className = own ? 'tile own' : 'tile'
  figure.innerHTML = `<div class="frame"><img alt="${photo.alt.replaceAll('"', '&quot;')}" decoding="async"><canvas hidden></canvas><div class="box" hidden></div><div class="window" hidden></div><div class="af">${reticle}</div><div class="ideal" hidden>${cross}</div></div>
<figcaption>${photo.page ? `<a href="${photo.page}" target="_blank" rel="noopener">${photo.author}</a>` : `<span>${photo.author}</span>`}<span class="t"></span></figcaption>`
  const q = <T extends HTMLElement>(s: string) => figure.querySelector(s) as T
  const item: Item = {
    photo,
    index: items.length,
    frame: q('.frame'),
    img: q('img'),
    heat: q('canvas'),
    af: q('.af'),
    ideal: q('.ideal'),
    window: q('.window'),
    box: q('.box'),
    time: q('.t')
  }
  item.img.addEventListener('load', () => {
    item.img.classList.add('ready')
    analyse(item)
  })
  item.frame.addEventListener('click', e => {
    if (!state.mark) return
    const b = item.frame.getBoundingClientRect()
    const u = (e.clientX - b.left) / b.width
    const v = (e.clientY - b.top) / b.height
    const {width: W, height: H} = item.photo
    const r = crop(item.point ?? {x: 0.5, y: 0.5}, W, H, aspectOf(item))
    const x = state.view === 'full' ? u : (r.x + u * r.width) / W
    const y = state.view === 'full' ? v : (r.y + v * r.height) / H
    labels[item.photo.id] = {x: Math.round(x * 1000) / 1000, y: Math.round(y * 1000) / 1000}
    save()
    render(item)
    summary()
  })
  item.img.src = photo.src
  if (own) sheet.prepend(figure)
  else sheet.append(figure)
  render(item)
  return item
}

/** Pixels for the analysis: the photo drawn small, as the browser helper does. */
function pixels(img: HTMLImageElement): ImageData {
  const scale = Math.min(1, 256 / Math.max(img.naturalWidth, img.naturalHeight))
  const w = Math.max(1, Math.round(img.naturalWidth * scale))
  const h = Math.max(1, Math.round(img.naturalHeight * scale))
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d', {willReadFrequently: true})!
  ctx.drawImage(img, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h)
}

function analyse(item: Item) {
  item.data = pixels(item.img)
  compute(item)
  render(item)
  summary()
}

function timed<T>(f: () => T): [T, number] {
  const t = performance.now()
  const value = f()
  return [value, performance.now() - t]
}

function compute(item: Item) {
  if (!item.data) return
  const method = state.method
  if (method === 'centre') {
    item.point = {x: 0.5, y: 0.5, box: {x: 0, y: 0, width: 1, height: 1}, confidence: 0}
    return
  }
  const p = state.params[method]
  let map: SaliencyMap
  if (method === 'model') {
    if (item.model?.key !== p.size) {
      const [m, ms] = timed(() => modelSaliency(item.data!, p))
      item.model = {key: p.size, map: m, ms}
    }
    map = item.model.map
  } else {
    const key = signals.map(k => p[k]).join()
    if (item.heuristic?.key !== key) {
      const [m, ms] = timed(() => saliency(item.data!, p))
      item.heuristic = {key, map: m, ms}
    }
    map = item.heuristic.map
  }
  item.point = locate(map, p)
}

function aspectOf(item: Item): number {
  const shape = shapes.find(s => s.id === state.shape)!
  if (shape.aspect > 0) return shape.aspect
  if (shape.aspect < 0) return item.photo.width / item.photo.height
  return mixed[item.index % mixed.length]!
}

/** The crop window for a container of `aspect`, centred on the point and clamped (image units). */
function crop(p: {x: number; y: number}, w: number, h: number, aspect: number) {
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

const pct = (v: number) => `${(v * 100).toFixed(2)}%`

function render(item: Item) {
  const {photo, frame, af, window: win, box, heat} = item
  const W = photo.width, H = photo.height
  const p = item.point ?? {x: 0.5, y: 0.5, box: {x: 0, y: 0, width: 1, height: 1}, confidence: 0}
  const aspect = aspectOf(item)
  const r = crop(p, W, H, aspect)
  const full = state.view === 'full'
  frame.style.setProperty('--aspect', String(full ? W / H : aspect))
  // object-position aligns proportionally, so translate the centred and
  // clamped window into the matching percentage
  const ox = W > r.width + 0.5 ? r.x / (W - r.width) : 0.5
  const oy = H > r.height + 0.5 ? r.y / (H - r.height) : 0.5
  frame.style.setProperty('--pos', full ? '50% 50%' : `${pct(ox)} ${pct(oy)}`)
  // Map image coordinates (0..1) into the frame
  const fx = (x: number) => (full ? x : (x * W - r.x) / r.width)
  const fy = (y: number) => (full ? y : (y * H - r.y) / r.height)
  af.hidden = !state.point || !item.point
  af.style.setProperty('--px', pct(fx(p.x)))
  af.style.setProperty('--py', pct(fy(p.y)))
  const mine = labels[photo.id]
  item.ideal.hidden = !mine
  if (mine) {
    item.ideal.style.setProperty('--px', pct(fx(mine.x)))
    item.ideal.style.setProperty('--py', pct(fy(mine.y)))
  }
  win.hidden = !full
  if (full) Object.assign(win.style, {left: pct(r.x / W), top: pct(r.y / H), width: pct(r.width / W), height: pct(r.height / H)})
  box.hidden = !state.box || !item.point || state.method === 'centre'
  if (!box.hidden) {
    const b = p.box
    Object.assign(box.style, {left: pct(fx(b.x)), top: pct(fy(b.y)), width: pct(fx(b.x + b.width) - fx(b.x)), height: pct(fy(b.y + b.height) - fy(b.y))})
  }
  const map = state.method === 'model' ? item.model?.map : state.method === 'heuristic' ? item.heuristic?.map : undefined
  heat.hidden = !state.heat || !map
  if (map && !heat.hidden && item.drawn !== map) drawHeat(heat, map, (item.drawn = map))
  const ms = state.method === 'model' ? item.model?.ms : state.method === 'heuristic' ? item.heuristic?.ms : undefined
  item.time.textContent = ms === undefined ? '' : `${ms.toFixed(1)} ms`
}

function pencil(): [number, number, number] {
  const hex = getComputedStyle(document.documentElement).getPropertyValue('--pencil').trim().replace('#', '')
  return [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number]
}

function drawHeat(canvas: HTMLCanvasElement, map: SaliencyMap, _: SaliencyMap) {
  canvas.width = map.width
  canvas.height = map.height
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(map.width, map.height)
  const [r, g, b] = pencil()
  for (let i = 0; i < map.data.length; i++) {
    const v = map.data[i]!
    img.data[i * 4] = r
    img.data[i * 4 + 1] = g
    img.data[i * 4 + 2] = b
    img.data[i * 4 + 3] = Math.round(Math.pow(v, 0.9) * 200)
  }
  ctx.putImageData(img, 0, 0)
}

let pending = false
function update() {
  if (pending) return
  pending = true
  requestAnimationFrame(() => {
    pending = false
    for (const item of items) {
      compute(item)
      render(item)
    }
    summary()
  })
}

function summary() {
  const done = items.filter(i => i.data)
  if (!done.length) return
  const method = state.method
  const maps = done.map(i => (method === 'model' ? i.model?.ms : method === 'heuristic' ? i.heuristic?.ms : 0) ?? 0)
  const [, place] = timed(() => {
    if (method !== 'centre') for (const i of done) locate((method === 'model' ? i.model : i.heuristic)!.map, state.params[method])
  })
  const avg = maps.reduce((a, b) => a + b, 0) / done.length
  const name = method === 'model' ? 'learned model' : method === 'heuristic' ? 'heuristic' : 'centre point'
  const marked = done.filter(i => labels[i.photo.id] && i.point)
  const score = marked.length ? marked.reduce((s, i) => s + agreement(i, i.point!), 0) / marked.length : 0
  readout.innerHTML =
    (method === 'centre'
      ? `<b>${done.length}</b> photos, every crop centred.<br>Switch to the model to see the difference.`
      : `<b>${done.length}</b> photos · ${name}<br>map <b>${avg.toFixed(1)} ms</b> · point <b>${(place / done.length).toFixed(2)} ms</b> per photo`) +
    (marked.length ? `<br><b>${marked.length}</b> marked · crops keep your point <b>${(score * 100).toFixed(0)}%</b>` : '')
}

/** Save marks: to the dev server when there is one, and in this browser. */
function save() {
  try {
    localStorage.setItem('focuspoint-labels', JSON.stringify(labels))
  } catch {}
  fetch('labels', {method: 'POST', body: JSON.stringify(labels)}).catch(() => {})
}

async function load() {
  try {
    labels = {...JSON.parse(localStorage.getItem('focuspoint-labels') ?? '{}')}
  } catch {}
  try {
    const res = await fetch('labels.json', {cache: 'no-store'})
    if (res.ok) labels = {...(await res.json()), ...labels}
  } catch {}
  for (const item of items) render(item)
  summary()
}

/** Share of the five container crops that keep the marked point. */
function agreement(item: Item, p: {x: number; y: number}) {
  const mine = labels[item.photo.id]!
  const {width: W, height: H} = item.photo
  let kept = 0
  for (const aspect of [3, 16 / 9, 1, 4 / 5, 9 / 16]) {
    const r = crop(p, W, H, aspect)
    if (mine.x * W >= r.x && mine.x * W <= r.x + r.width && mine.y * H >= r.y && mine.y * H <= r.y + r.height) kept++
  }
  return kept / 5
}

function addFiles(files: FileList | null) {
  for (const file of files ?? []) {
    if (!file.type.startsWith('image/')) continue
    const url = URL.createObjectURL(file)
    const probe = new Image()
    probe.onload = () => {
      const photo: Photo = {id: url, alt: file.name, author: file.name, page: '', src: url, width: probe.naturalWidth, height: probe.naturalHeight}
      items.push(tile(photo, true))
    }
    probe.src = url
  }
}

controls()
load()
for (const photo of (window as unknown as {PHOTOS: Array<Photo>}).PHOTOS) items.push(tile(photo))
