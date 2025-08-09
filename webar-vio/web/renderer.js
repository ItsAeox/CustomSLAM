import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let scene, camera, renderer, cactus;
export async function initRenderer(canvas) {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(canvas.width, canvas.height);
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, canvas.width/canvas.height, 0.01, 1000);
    camera.position.set(0,0,0); // we'll move the world instead

    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    light.position.set(1,1,1);
    scene.add(light);

    const loader = new GLTFLoader();
    const glb = await loader.loadAsync('cactus.glb');
    cactus = glb.scene;
    scene.add(cactus);
    cactus.scale.set(0.1,0.1,0.1);
}

export function updateCactusPose(T_wc_array) {
    if (!cactus) return;
    // Convert T_wc (world-from-camera) into object transform.
    // Easiest: keep camera at origin and move the cactus by the inverse (T_cw).
    const m = new THREE.Matrix4();
    m.fromArray(T_wc_array);                 // column-major already
    m.invert();                              // T_cw
    cactus.matrixAutoUpdate = false;
    cactus.matrix.copy(m);
    renderer.render(scene, camera);
}
