// The AR scene of the AlvaAR and WebXR versions, identical to web/ar.js: the same lights, the same plant (assets/tree.glb,
// normalised to height 1 with its base centre at the origin) and the same size rule (0.6 x the camera's height above
// the surface it stands on).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export function makeScene() {
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  scene.add(sun);
  return { scene, sun };
}

export function loadTree(url) {
  return new Promise((res, rej) => new GLTFLoader().load(url, (g) => {
    const box = new THREE.Box3().setFromObject(g.scene);
    const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
    g.scene.position.set(-centre.x, -box.min.y, -centre.z);
    const wrap = new THREE.Group();
    wrap.add(g.scene);
    wrap.scale.setScalar(1 / (size.y || 1));
    const proto = new THREE.Group();
    proto.add(wrap);
    res(proto);
  }, undefined, rej));
}

export const TREE_HEIGHT = (cameraHeight) => 0.6 * cameraHeight;  // ar.js placeCactus: size = 0.6 * floor.height

// Pinhole intrinsics of the WebVIO page for a W x H image: 69 deg horizontal on the long side (web/ar.js default)
export function phoneIntrinsics(W, H, hfovLongDeg = 69) {
  const f = (Math.max(W, H) / 2) / Math.tan(hfovLongDeg * Math.PI / 360);
  return { fx: f, fy: f, cx: W / 2, cy: H / 2 };
}

// OpenCV pinhole -> three.js projection (as ar.js layout())
export function setProjection(cam, K, W, H) {
  const n = cam.near, f = cam.far;
  cam.projectionMatrix.set(
    2 * K.fx / W, 0, 1 - 2 * K.cx / W, 0,
    0, 2 * K.fy / H, 2 * K.cy / H - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
}
