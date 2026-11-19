// gpubench.html?load: AR-page-like rendering on the main thread while the benchmark runs in its worker, so the GPU
// numbers include the GPU's other job. Like web/ar.js: a 480x640 2D canvas that gets a new image every frame (the camera
// picture; here the benchmark's recording frame, re-uploaded as a fresh ImageBitmap each time) and a three.js canvas on
// top (antialias, pixel ratio up to 2, cover-fit to the window) drawing the tree model with the AR page's lights, at
// 30 frames per second (the camera rate). Records the interval between drawn frames per phase (stutter check).
//   startLoad(frameUrl) -> { setPhase(name), report() -> text lines }
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export async function startLoad(frameUrl) {
  const W = 480, H = 640;
  const stage = document.createElement('div');
  Object.assign(stage.style, { position: 'fixed', inset: '0', zIndex: '-1', overflow: 'hidden', background: '#000' });
  const bg = document.createElement('canvas'), gl = document.createElement('canvas');
  stage.append(bg, gl);
  document.body.prepend(stage);
  document.body.style.background = 'transparent';
  const s = Math.max(innerWidth / W, innerHeight / H), rect = { w: W * s, h: H * s, x: (innerWidth - W * s) / 2, y: (innerHeight - H * s) / 2 };
  for (const c of [bg, gl]) Object.assign(c.style, { position: 'absolute', left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.w}px`, height: `${rect.h}px`, opacity: '0.35' });
  bg.width = W; bg.height = H;
  const bgx = bg.getContext('2d');
  const renderer = new THREE.WebGLRenderer({ canvas: gl, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setSize(rect.w, rect.h, false);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(1, 2, 1);
  scene.add(sun);
  const cam = new THREE.PerspectiveCamera(50, W / H, 0.01, 100);
  cam.position.set(0, 0.6, 2.2);
  cam.lookAt(0, 0.5, 0);
  const tree = new THREE.Group();
  scene.add(tree);
  await new Promise((res) => new GLTFLoader().load('./assets/tree.glb', (g) => { tree.add(g.scene); res(); }, undefined, () => res()));
  const blob = await (await fetch(frameUrl)).blob();
  const pixels = await (async () => {  // the frame as ImageData, re-uploaded every frame as a new ImageBitmap
    const b = await createImageBitmap(blob), c = new OffscreenCanvas(W, H), x = c.getContext('2d');
    x.drawImage(b, 0, 0);
    b.close();
    return x.getImageData(0, 0, W, H);
  })();
  let phase = 'baseline (rendering only)', last = 0, odd = false, busy = false;
  const ft = new Map();
  const loop = async (t) => {
    requestAnimationFrame(loop);
    odd = !odd;
    if (odd || busy) return;  // 30 fps
    busy = true;
    const bmp = await createImageBitmap(pixels);
    bgx.drawImage(bmp, 0, 0);
    bmp.close();
    tree.rotation.y = t / 2000;
    renderer.render(scene, cam);
    if (last) { if (!ft.has(phase)) ft.set(phase, []); ft.get(phase).push(t - last); }
    last = t;
    busy = false;
  };
  requestAnimationFrame(loop);
  const stats = (v) => {
    const a = [...v].sort((x, y) => x - y), q = (p) => a[Math.min(a.length - 1, Math.floor(p * a.length))];
    return `${a.length} frames, interval median ${q(0.5).toFixed(1)} ms, p95 ${q(0.95).toFixed(1)}, max ${a[a.length - 1].toFixed(1)}, ` +
           `${(100 * a.filter((x) => x > 50).length / a.length).toFixed(1)} % over 50 ms`;
  };
  return {
    setPhase: (p) => { phase = p; },
    report: () => [...ft].map(([p, v]) => `render load, ${p}: ${stats(v)}`),
  };
}
