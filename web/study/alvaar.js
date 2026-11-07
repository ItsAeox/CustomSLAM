// AlvaAR version of the blind study (claude/STUDY.md §1): AlvaAR's released bundle (vendor/alva_ar.js, sha256 d0b74612...,
// commit 7796af5, unmodified) driven as in its own imu.html example (IMU class from its examples, findCameraPoseWithIMU,
// objects placed on its virtual ground 10 units below the start, objects hidden while lost). Changed for the study,
// equal to the WebVIO version: the camera stream (640x480 request), AlvaAR's intrinsics set from the same 69 deg
// horizontal-FOV assumption (f = 465.6 px at 480x640), the render projection from those intrinsics (its example
// renders with a fixed 60 deg camera), the plant, lights and size rule (study/scene.js), one object, the task overlay.
import * as THREE from 'three';
import { AlvaAR } from './vendor/alva_ar.js';
import { AlvaARConnectorTHREE } from './vendor/alva_ar_three.js';
import { IMU } from './vendor/imu.js';
import { Overlay, canvasBrightness, finishCondition, startGate, versionIndex } from './common.js';
import { TREE_HEIGHT, loadTree, makeScene, phoneIntrinsics, setProjection } from './scene.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), cv = $('cv'), gl = $('gl');

startGate(versionIndex(), async () => {
  let ov = null;
  try {
    const imu = await IMU.Initialize();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } });
    video.srcObject = stream;
    await video.play();
    const W = video.videoWidth, H = video.videoHeight;  // portrait phone: 480 x 640
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    const K = phoneIntrinsics(W, H);
    // AlvaAR.Initialize(width, height, fov): with width < height the fov is horizontal -> f = (W/2) / tan(fov/2)
    const fovDeg = 2 * Math.atan((W / 2) / K.fx) * 180 / Math.PI;
    const alva = await AlvaAR.Initialize(W, H, W < H ? fovDeg : 2 * Math.atan((H / 2) / K.fy) * 180 / Math.PI);

    const renderer = new THREE.WebGLRenderer({ canvas: gl, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const { scene, sun } = makeScene();
    const cam = new THREE.PerspectiveCamera(50, W / H, 0.01, 1000);
    scene.add(cam);
    const applyPose = AlvaARConnectorTHREE.Initialize(THREE);
    const ground = new THREE.Mesh(new THREE.CircleGeometry(1000, 64), new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide }));
    ground.rotation.x = Math.PI / 2; ground.position.y = -10;  // AlvaAR imu.html: the ground plane 10 units below
    scene.add(ground);
    const proto = await loadTree(new URL('../assets/tree.glb', import.meta.url).href);
    let tree = null, tracking = false;
    const ray = new THREE.Raycaster();

    let rect = null;
    const layout = () => {  // cover-fit the camera image into the window (as ar.js)
      const s = Math.max(innerWidth / W, innerHeight / H);
      rect = { s, w: W * s, h: H * s, x: (innerWidth - W * s) / 2, y: (innerHeight - H * s) / 2 };
      for (const c of [cv, gl]) Object.assign(c.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.w}px`, height: `${rect.h}px` });
      renderer.setSize(rect.w, rect.h, false);
      setProjection(cam, K, W, H);
    };
    layout(); addEventListener('resize', layout);

    ov = new Overlay({
      objectScreen: () => {
        if (!tree) return null;
        cam.updateMatrixWorld();
        const v = tree.position.clone().project(cam);
        return { x: rect.x + (v.x + 1) / 2 * rect.w, y: rect.y + (1 - v.y) / 2 * rect.h, visible: tree.visible && v.z > -1 && v.z < 1 };
      },
      brightness: () => canvasBrightness(cv),
    }, (log) => { stream.getTracks().forEach((t) => t.stop()); finishCondition('alvaar', { ...log, alvaar: 'd0b74612 (7796af5)', image: [W, H], f: K.fx }); });

    $('stage').addEventListener('pointerdown', (e) => {
      if (ov.tap(e.clientX, e.clientY) || !tracking) return;
      const ndc = new THREE.Vector2(((e.clientX - rect.x) / rect.w) * 2 - 1, -((e.clientY - rect.y) / rect.h) * 2 + 1);
      cam.updateMatrixWorld();
      ray.setFromCamera(ndc, cam);
      const hit = ray.intersectObject(ground)[0];
      if (!hit) return;
      if (tree) scene.remove(tree);
      tree = proto.clone();
      tree.position.copy(hit.point);
      tree.scale.setScalar(TREE_HEIGHT(cam.position.y - ground.position.y));
      scene.add(tree);
      ov.placed();
    });

    const loop = () => {
      ctx.drawImage(video, 0, 0, W, H);
      const frame = ctx.getImageData(0, 0, W, H);
      const pose = alva.findCameraPoseWithIMU(frame, imu.orientation, imu.motion);
      tracking = !!pose;
      if (pose) {
        applyPose(pose, cam.quaternion, cam.position);
        sun.position.set(cam.position.x, cam.position.y + 2, cam.position.z);
      }
      if (tree) tree.visible = tracking;  // AlvaAR's example hides objects while lost
      renderer.render(scene, cam);
      ov.frame(tracking);
      if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(loop); else requestAnimationFrame(loop);
    };
    loop();
  } catch (e) {
    ov ??= new Overlay({}, (log) => finishCondition('alvaar', log));
    ov.fail(String(e.message || e));
  }
});
