# HF Oblique FMCW Chirpsounder Hunter & Predictive MUF Tracker

An autonomous Python DSP application for the **Airspy HF+** that intercepts diagonal FMCW ionospheric chirpsounders (such as AN/TRQ-35, Digisonde, and OTH radar sounders) and actively hops ahead of the sweep up to 30 MHz to plot a real-time Oblique Ionogram and measure the Maximum Usable Frequency (MUF).

## How It Works
1. **Stage 1 — Sentinel Watch (2D Radon Transform):** Monitors a 96 kHz wideband passband at a user-defined tripwire frequency (e.g., 5.500 MHz). A rolling background-whitened spectrogram is evaluated across candidate chirp slopes (50 to 200 kHz/s) using 2D Radon line integration to isolate diagonal sweeps from vertical static crashes and stationary carriers.
2. **Stage 2 — Differential Quadratic Phase De-Chirp Verifier:** Candidate detections are multiplied by the complex conjugate quadratic phase kernel $\exp(-j \pi k t^2)$ across a 16,384-sample (170.6 ms) window. Genuine linear FM sweeps collapse into a single spectral bin, producing $+6.5\text{ to }+22\text{ dB}$ of pulse-compression gain over the un-dechirped FFT, while stationary broadcast carriers and Gaussian noise are rejected.
3. **Stage 3 — Predictive Hardware Chase Mode:** Once locked onto a chirp's instantaneous frequency $f_0$ and slope $k = df/dt$, the program back-projects the sweep's 2.0 MHz UTC start time and commands the Airspy HF+ hardware PLL to hop ahead in 500 kHz increments up to 29.5 MHz—plotting verified de-chirped SNR vs. frequency until the signal fades above the ionospheric MUF.

## Requirements
* Python 3.10+ (`numpy`, `scipy`, `pillow`)
* Airspy HF+ Discovery / Dual Port (`airspyhf.dll` from SDR# Studio)