const Module = await createVioModule();
Module.initSystem(w,h, fx,fy,cx,cy);

const video = document.querySelector('video');
const off = new OffscreenCanvas(w,h);
const ctx = off.getContext('2d', { willReadFrequently: true });

async function pump() {
    ctx.drawImage(video, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h); // RGBA
    const ptr = Module._malloc(imageData.data.length);
    Module.HEAPU8.set(imageData.data, ptr);
    const stride = imageData.width * 4; // bytes per row
    Module.feedFrame(ptr, performance.now() * 1e-3, stride, true);
    Module._free(ptr);

    const pose = Module.getPose(); // Float64Array[16]
    updateCactusPose(pose);
    requestAnimationFrame(pump);
}
