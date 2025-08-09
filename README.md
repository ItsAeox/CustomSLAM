# Custom SLAM Pipeline for Markerless Tracking in WebAR (Based on ORBSLAM3)

## Overview

This project implements a custom SLAM (Simultaneous Localization and Mapping) pipeline for markerless tracking in WebAR applications, leveraging the capabilities of [ORBSLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). The pipeline enables robust real-time camera pose estimation and environment mapping without the need for physical markers, making it ideal for immersive AR experiences on the web.

## Features

- **Markerless Tracking:** No physical markers required for localization.
- **Real-Time Performance:** Optimized for web environments using WebAssembly and WebGL.
- **Multi-Platform Support:** Compatible with modern browsers and mobile devices.
- **Based on ORBSLAM3:** Utilizes proven algorithms for feature extraction, matching, and pose estimation.
- **Extensible Architecture:** Modular design for easy integration and customization.

## Getting Started

### Prerequisites

- Node.js and npm
- Modern web browser (Chrome, Firefox, Safari, Edge)
- [Emscripten](https://emscripten.org/) for building C++ code to WebAssembly

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/custom-slam-webar.git
    cd custom-slam-webar
    ```

2. Install dependencies:
    ```bash
    npm install
    ```

3. Build the WebAssembly module:
    ```bash
    npm run build:wasm
    ```

4. Start the development server:
    ```bash
    npm start
    ```

### Usage

- Open your browser and navigate to `http://localhost:3000`.
- Grant camera permissions when prompted.
- Move your device to start markerless tracking and mapping.

## Architecture

- **Frontend:** JavaScript/TypeScript, WebGL for rendering, WebAssembly for SLAM core.
- **SLAM Core:** ORBSLAM3 algorithms compiled to WebAssembly.
- **Camera Interface:** Accesses device camera via WebRTC/Web APIs.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, improvements, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [ORBSLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [Emscripten](https://emscripten.org/)
- WebAR community
