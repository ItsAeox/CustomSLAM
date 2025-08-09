import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

let scene, camera, renderer;
let cactusAnchor; // parent for cactus so scaling is isolated
let W = 0, H = 0;

export async function initRenderer(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0); // transparent
  // Size will be set after we know video W/H via configureCameraFromIntrinsics

  scene = new THREE.Scene();

  // Temp camera; proper intrinsics are applied later
  camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  camera.matrixAutoUpdate = false;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  // Add cactus (under an anchor)
  const loader = new GLTFLoader();
  const glb = await loader.loadAsync('./cactus.glb');
  const cactus = glb.scene;
  cactus.scale.setScalar(0.1); // ~10cm to start

  cactusAnchor = new THREE.Object3D();
  cactusAnchor.matrixAutoUpdate = false;
  cactusAnchor.add(cactus);
  scene.add(cactusAnchor);

  // Put cactus 60 cm in front of world origin initially
  const place = new THREE.Matrix4().makeTranslation(0, 0, -0.6);
  cactusAnchor.matrix.copy(place);

  // Axes for sanity
  scene.add(new THREE.AxesHelper(0.2));

  renderer.render(scene, camera);
}

// Call this once you know fx, fy, cx, cy and the actual pixel size of the video
export function configureCameraFromIntrinsics({ fx, fy, cx, cy, width, height }) {
  W = width; H = height;

  // Vertical FOV from fy
  const fovYdeg = 2 * Math.atan(0.5 * H / fy) * 180 / Math.PI;
  camera.fov = fovYdeg;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();

  // Principal point offset (cx, cy)
  camera.setViewOffset(
    W, H,
    (W * 0.5 - cx), // + right
    (H * 0.5 - cy), // + down
    W, H
  );

  // Ensure canvas matches video pixels; avoid CSS stretching
  renderer.setSize(W, H, false);
  renderer.domElement.style.width  = `${W}px`;
  renderer.domElement.style.height = `${H}px`;
}

// Update the CAMERA from world-from-camera pose (already in three.js basis)
export function updateFromPose_Twc_threeBasis(T_wc_array) {
  if (!T_wc_array || T_wc_array.length !== 16) return;
  camera.matrix.fromArray(T_wc_array);
  camera.updateMatrixWorld(true);

  renderer.render(scene, camera);
}

let t = 0;
export function debugWobbleCamera() {
  const T = new THREE.Matrix4()
    .makeRotationY(0.2 * Math.sin(t));
  camera.matrix.copy(T);
  camera.updateMatrixWorld(true);
  renderer.render(scene, camera);
  t += 0.05;
}
