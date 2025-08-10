import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

let scene, camera, renderer;
let cactusAnchor; // parent for cactus so scaling is isolated
let W = 0, H = 0;

const raycaster = new THREE.Raycaster();
const tapPlane  = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0); // y=0 plane


export async function initRenderer(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0); // transparent
  renderer.domElement.style.backgroundColor = 'transparent';

  scene = new THREE.Scene();

  // Temp camera; proper intrinsics applied later
  camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  camera.matrixAutoUpdate = false;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  // Load cactus under an anchor
  const loader = new GLTFLoader();
  const glb = await loader.loadAsync(`./cactus.glb?v=${Date.now()}`);
  const cactus = glb.scene;
  cactus.scale.setScalar(0.4); // ~10 cm to start

  cactusAnchor = new THREE.Object3D();
  cactusAnchor.matrixAutoUpdate = false;
  cactusAnchor.add(cactus);
  scene.add(cactusAnchor);

  // Initial placement ~60 cm forward from world origin
  const place = new THREE.Matrix4().makeTranslation(0, 0, -0.6);
  cactusAnchor.matrix.copy(place);

  // Sanity helpers
  scene.add(new THREE.AxesHelper(0.2));
  const proofCube = new THREE.Mesh(
    new THREE.BoxGeometry(0.05,0.05,0.05),
    new THREE.MeshNormalMaterial()
  );
  proofCube.position.set(0,0,-0.6);
  scene.add(proofCube);

  renderer.render(scene, camera);
}

// Configure three.js camera from intrinsics used by SLAM
export function configureCameraFromIntrinsics({ fx, fy, cx, cy, width, height }) {
  W = width; H = height;

  // Vertical FOV from fy
  const fovYdeg = 2 * Math.atan(0.5 * H / fy) * 180 / Math.PI;
  camera.fov = fovYdeg;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();

  // Principal point offset
  camera.setViewOffset(
    W, H,
    (W * 0.5 - cx),
    (H * 0.5 - cy),
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

  const M = new THREE.Matrix4().fromArray(T_wc_array);

  camera.matrix.copy(M).invert(); 
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  camera.matrixAutoUpdate = false;

  // Mark dirty and render
  camera.matrixWorldNeedsUpdate = true;
  renderer.render(scene, camera);
}

// Place anchor by explicit world matrix (rarely needed externally)
export function placeAnchorAtWorldMatrix(mat4Array) {
  if (!cactusAnchor) return;
  cactusAnchor.matrix.fromArray(mat4Array);
  cactusAnchor.matrixAutoUpdate = false;
}

// Place anchor from a tap on screen; intersect y=0 plane or 0.6 m along ray
export function placeAnchorAtScreen(clientX, clientY) {
  if (!cactusAnchor || !renderer || !camera) return;

  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -(((clientY - rect.top) / rect.height) * 2 - 1)
  );

  raycaster.setFromCamera(ndc, camera);

  const hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(tapPlane, hit)) {
    const M = new THREE.Matrix4().makeTranslation(hit.x, hit.y, hit.z);
    cactusAnchor.matrix.copy(M);
  } else {
    const p = new THREE.Vector3()
      .copy(raycaster.ray.origin)
      .addScaledVector(raycaster.ray.direction, 0.6);
    const M = new THREE.Matrix4().makeTranslation(p.x, p.y, p.z);
    cactusAnchor.matrix.copy(M);
  }
  cactusAnchor.matrixAutoUpdate = false;
  renderer.render(scene, camera);
}

let t = 0;
export function debugWobbleCamera() {
  // small lateral translation + yaw, so the cube cannot stay centered
  const T = new THREE.Matrix4()
    .makeTranslation(0.05 * Math.sin(t), 0, 0)
    .multiply(new THREE.Matrix4().makeRotationY(0.25 * Math.sin(t)));

  camera.matrix.copy(T);
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  camera.matrixAutoUpdate = false;
  camera.matrixWorldNeedsUpdate = true;

  renderer.render(scene, camera);
  t += 0.05;
}
