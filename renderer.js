import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

let scene, camera, renderer;
let cactusAnchor; // parent for cactus so scaling is isolated
let W = 0, H = 0;

const raycaster = new THREE.Raycaster();
const tapPlane  = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0); // y=0 plane

export async function initRenderer(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(W, H, false);
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
  cactus.scale.setScalar(0.2);

  cactusAnchor = new THREE.Object3D();
  cactusAnchor.matrixAutoUpdate = false;
  cactusAnchor.add(cactus);
  scene.add(cactusAnchor);

  // Initial placement ~60 cm forward from world origin
  const place = new THREE.Matrix4().makeTranslation(0, 0, -2);
  cactusAnchor.matrix.copy(place);

  renderer.render(scene, camera);
  initPlaneGizmo(0.7);
}

// --- Dominant plane gizmo ---
let planeGroup = null;        // parent group (so we can scale/rotate together)
let planeMesh = null;         // translucent filled quad
let planeGrid = null;         // grid overlay (very visible)
let planeDot  = null;         // centroid marker
let planeEq   = null;         // THREE.Plane for raycast (n·x + c = 0)

export function initPlaneGizmo(size = 1) {
  if (planeGroup) return;

  planeGroup = new THREE.Group();
  planeGroup.visible = false;
  scene.add(planeGroup);

  // Quad (faces +Z in local space)
  const geo = new THREE.PlaneGeometry(size, size, 1, 1);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x00ff88,
    opacity: 0.35,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  planeMesh = new THREE.Mesh(geo, mat);
  planeGroup.add(planeMesh);

  // Grid helper lives in XZ; rotate it so its normal is +Z (same as PlaneGeometry)
  planeGrid = new THREE.GridHelper(size, 10, 0x66ffff, 0x66ffff);
  planeGrid.rotation.x = Math.PI / 2;
  planeGroup.add(planeGrid);

  // Centroid marker
  planeDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.03, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xff00ff })
  );
  planeGroup.add(planeDot);

  planeEq = new THREE.Plane(new THREE.Vector3(0,1,0), 0);
}

/**
 * params: [nx,ny,nz, d, cx,cy,cz, inliers] already in THREE basis (x,-y,-z) from WASM
 */
export function updatePlaneFromParams(params) {
  if (!planeGroup || !params || params.length < 8) return;

  const n = new THREE.Vector3(params[0], params[1], params[2]).normalize();
  const d = params[3];
  const c = new THREE.Vector3(params[4], params[5], params[6]);
  const inliers = params[7] | 0;

  // Rotate local +Z to the plane normal n
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1), n);
  planeGroup.quaternion.copy(q);
  planeGroup.position.copy(c);
  planeDot.position.set(0,0,0); // centered in the group

  // 🔎 Auto-scale plane size to distance from camera so you can always see it
  const dist = c.distanceTo(camera.position);
  const s = Math.min(Math.max(dist * 1.5, 0.4), 8.0); // clamp [0.4, 8] world units
  planeMesh.scale.set(s, s, 1);
  planeGrid.scale.set(s, s, 1);

  // Show when we have enough support
  planeGroup.visible = inliers >= 35;

  // ✅ Update the infinite raycast plane for taps: THREE.Plane uses n·x + constant = 0
  // Our exported 'd' is in the same form, so do NOT negate it.
  if (!planeEq) planeEq = new THREE.Plane();
  planeEq.set(n, d);

  // Force a render in case camera pose didn't change this frame
  renderer.render(scene, camera);
}

export function setPlaneVisible(v) { if (planeGroup) planeGroup.visible = !!v; }

// --- Point cloud debug (Three.js) ---
let pcObj = null;
let pcGeom = null;
let pcMat  = null;
const BASIS_FLIP = new THREE.Vector3(1, -1, -1); // SLAM -> Three basis (x, -y, -z)

/**
 * Create the point cloud object (once).
 * @param {number} maxPoints - initial capacity; can grow automatically.
 * @param {number} sizePx - point size in pixels (screen-space).
 */
export function initPointCloud(maxPoints = 2000, sizePx = 3) {
  if (pcObj) return;

  pcGeom = new THREE.BufferGeometry();
  // preallocate; we'll adjust drawRange each update
  pcGeom.setAttribute('position', new THREE.Float32BufferAttribute(maxPoints * 3, 3));
  pcGeom.setDrawRange(0, 0);

  // sizeAttenuation=false => size in pixels, steady for debugging
  pcMat = new THREE.PointsMaterial({ size: sizePx, sizeAttenuation: false });
  pcObj = new THREE.Points(pcGeom, pcMat);
  pcObj.frustumCulled = false; // tiny points can be culled too aggressively

  scene.add(pcObj);
}

/**
 * Update the point cloud from a flat array/TypedArray of [x0,y0,z0, x1,y1,z1, ...]
 * Coordinates are assumed in SLAM world; we flip to Three basis internally.
 * @param {Array|Float32Array|Float64Array} xyz
 */
export function updatePointCloud(xyz) {
  if (!xyz) return;
  if (!pcObj) initPointCloud(Math.max(2000, Math.floor((xyz.length / 3) * 1.2)));

  const arr = (xyz.length !== undefined && xyz.BYTES_PER_ELEMENT)
    ? xyz
    : Float32Array.from(xyz);

  const wantN = (arr.length / 3) | 0;
  let buf = pcGeom.getAttribute('position');

  // Grow buffer if needed
  if (wantN * 3 > buf.array.length) {
    const newCap = Math.ceil(wantN * 1.5) * 3;
    const newAttr = new THREE.Float32BufferAttribute(newCap, 3);
    // no need to copy old data; we'll refill below
    pcGeom.setAttribute('position', newAttr);
    buf = newAttr;
  }

  // Fill with basis flip (x, -y, -z)
  const dst = buf.array;
  for (let i = 0, j = 0; i < wantN; ++i) {
    const x = arr[j++], y = arr[j++], z = arr[j++];
    dst[i * 3 + 0] = x * BASIS_FLIP.x;
    dst[i * 3 + 1] = y * BASIS_FLIP.y;
    dst[i * 3 + 2] = z * BASIS_FLIP.z;
  }

  pcGeom.setDrawRange(0, wantN);
  buf.needsUpdate = true;
  pcGeom.computeBoundingSphere?.();
}

/** Optional utilities */
export function setPointCloudVisible(v) { if (pcObj) pcObj.visible = !!v; }
export function setPointSize(px) { if (pcMat) pcMat.size = px; }


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

  // Use the same pattern as your debugWobble: set both local and world
  camera.matrix.copy(M);                 // local
  camera.matrixWorld.copy(M);            // world (no parent, so same)
  camera.matrixWorldInverse.copy(M).invert();  // view = T_cw

  camera.matrixAutoUpdate = false;
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

  // If we have a detected plane, intersect that; else fall back to 0.6m along the ray
  const hit = new THREE.Vector3();
  let placed = false;

  if (planeEq && planeMesh?.visible) {
    if (raycaster.ray.intersectPlane(planeEq, hit)) {
      const M = new THREE.Matrix4().makeTranslation(hit.x, hit.y, hit.z);
      cactusAnchor.matrix.copy(M);
      placed = true;
    }
  }

  if (!placed) {
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
