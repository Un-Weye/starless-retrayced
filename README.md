# [Python Black Hole Raytracer](http://rantonels.github.io/starless)

Starless-retrayced is a CPU black hole raytracer in numpy suitable for both informative diagrams and decent wallpaper material, bult on the base of [rantonels starless](http://rantonels.github.io/starless)

## Features

- Full geodesic raytracing in Schwarzschild geometry
- Predicts distortion of arbitrary objects defined implicitly
- Alpha-blended accretion disk
- Optional blackbody mode for accretion disk with realistic redshift (doppler + gravitational)
- Sky distortion
- Dust
- Bloom postprocessing by Airy disk convolution with spectral dependence
- Completely parallel - renders chunks of the image using numpy arrays arithmetic
- Multicore (with multiprocessing)
- Easy debugging by saving masks and intermediate results as images

## (Possible) future features

- Non-stationary observers, aberration, render from inside the event horizon
- Kerr geometry (rotating black holes with frame dragging)
- Integration of time variable, time dependent objects (e.g. orbiting sphere with retarded time rendering) with worldtube-null geodesic intersection
- Generic redshift framework for solid colour objects
- GPU compatible raytracing

## Installation and usage

Please refer to the [Wiki](https://github.com/Un-Weye/starless-retrayced/wiki).
