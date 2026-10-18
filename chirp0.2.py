import os
import sys
import csv
import json
import time
import queue
import ctypes
import argparse
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.signal import firwin, lfilter
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ==============================================================================
# 1. CTYPES WRAPPER FOR AIRSPYHF.DLL
# ==============================================================================

class AirspyHFTransfer(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_int),
        ("dropped_samples", ctypes.c_uint64)
    ]

AIRSPYHF_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(AirspyHFTransfer))


class AirspyHFDevice:
    def __init__(self, dll_dir="."):
        dll_dir = os.path.abspath(dll_dir)
        dll_path = os.path.join(dll_dir, "airspyhf.dll")

        if not os.path.isfile(dll_path):
            raise FileNotFoundError(
                f"Could not find 'airspyhf.dll' in: {dll_dir}\n"
                f"Pass your SDR# Studio directory via --dll-dir \"C:\\path\\to\\sdrsharp\""
            )

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

        self.lib = ctypes.CDLL(dll_path)
        self._setup_prototypes()

        self.dev_handle = ctypes.c_void_p(None)
        self.iq_queue = queue.Queue(maxsize=256)
        self.dropped_total = 0
        self.sample_rate = 192000
        self.center_freq = 5500000
        self._c_callback = AIRSPYHF_CALLBACK(self._rx_callback)

    def _setup_prototypes(self):
        self.lib.airspyhf_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.airspyhf_open.restype = ctypes.c_int
        self.lib.airspyhf_close.argtypes = [ctypes.c_void_p]
        self.lib.airspyhf_close.restype = ctypes.c_int
        self.lib.airspyhf_get_samplerates.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32
        ]
        self.lib.airspyhf_get_samplerates.restype = ctypes.c_int
        self.lib.airspyhf_set_samplerate.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.airspyhf_set_samplerate.restype = ctypes.c_int
        self.lib.airspyhf_set_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.airspyhf_set_freq.restype = ctypes.c_int
        self.lib.airspyhf_set_lib_dsp.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        self.lib.airspyhf_set_lib_dsp.restype = ctypes.c_int
        self.lib.airspyhf_set_hf_agc.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        self.lib.airspyhf_set_hf_agc.restype = ctypes.c_int
        self.lib.airspyhf_set_hf_agc_threshold.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        self.lib.airspyhf_set_hf_agc_threshold.restype = ctypes.c_int
        self.lib.airspyhf_set_hf_lna.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        self.lib.airspyhf_set_hf_lna.restype = ctypes.c_int
        self.lib.airspyhf_start.argtypes = [ctypes.c_void_p, AIRSPYHF_CALLBACK, ctypes.c_void_p]
        self.lib.airspyhf_start.restype = ctypes.c_int
        self.lib.airspyhf_stop.argtypes = [ctypes.c_void_p]
        self.lib.airspyhf_stop.restype = ctypes.c_int

    def open_and_configure(self, freq_hz, desired_rate=None, lna=True, agc=True):
        res = self.lib.airspyhf_open(ctypes.byref(self.dev_handle))
        if res != 0 or not self.dev_handle:
            raise RuntimeError(
                f"airspyhf_open() failed (code {res}). Ensure Airspy HF+ is connected and SDR# is closed."
            )

        num_rates = ctypes.c_uint32(0)
        self.lib.airspyhf_get_samplerates(self.dev_handle, ctypes.byref(num_rates), 0)
        rates_array = (ctypes.c_uint32 * num_rates.value)()
        self.lib.airspyhf_get_samplerates(self.dev_handle, rates_array, num_rates.value)
        supported_rates = list(rates_array)

        if desired_rate in supported_rates:
            self.sample_rate = desired_rate
        else:
            self.sample_rate = min(supported_rates)

        self.lib.airspyhf_set_samplerate(self.dev_handle, ctypes.c_uint32(self.sample_rate))
        self.lib.airspyhf_set_lib_dsp(self.dev_handle, ctypes.c_uint8(1))
        self.lib.airspyhf_set_hf_agc(self.dev_handle, ctypes.c_uint8(1 if agc else 0))
        if agc:
            self.lib.airspyhf_set_hf_agc_threshold(self.dev_handle, ctypes.c_uint8(0))
        self.lib.airspyhf_set_hf_lna(self.dev_handle, ctypes.c_uint8(1 if lna else 0))
        self.set_center_freq(freq_hz)

    def set_center_freq(self, freq_hz):
        self.center_freq = int(freq_hz)
        if self.dev_handle:
            self.lib.airspyhf_set_freq(self.dev_handle, ctypes.c_uint32(self.center_freq))

    def flush_queue(self):
        while not self.iq_queue.empty():
            try:
                self.iq_queue.get_nowait()
            except queue.Empty:
                break

    def _rx_callback(self, transfer_ptr):
        transfer = transfer_ptr.contents
        n = transfer.sample_count
        if n <= 0 or not transfer.samples:
            return 0
        if transfer.dropped_samples > 0:
            self.dropped_total += transfer.dropped_samples

        float_view = np.ctypeslib.as_array(transfer.samples, shape=(n * 2,))
        iq_copy = float_view.view(np.complex64).copy()
        try:
            self.iq_queue.put_nowait(iq_copy)
        except queue.Full:
            self.dropped_total += n
        return 0

    def start(self):
        res = self.lib.airspyhf_start(self.dev_handle, self._c_callback, None)
        if res != 0:
            raise RuntimeError(f"airspyhf_start() failed (code {res})")

    def stop_and_close(self):
        if self.dev_handle:
            self.lib.airspyhf_stop(self.dev_handle)
            self.lib.airspyhf_close(self.dev_handle)
            self.dev_handle = ctypes.c_void_p(None)


# ==============================================================================
# 2. AUTOMATED DATA LOGGER & COMPOSITE IONOGRAM + WATERFALL PNG SNAPSHOTTER
# ==============================================================================

class ChirpsounderArchiver:
    """
    Saves each verified chirpsounder run to disk:
      1. Master CSV table (chirpsounder_master_log.csv)
      2. Detailed per-chirp JSON profile (chirp_XXXX_...json)
      3. Composite 1200x650 PNG Snapshot combining the Intercept Waterfall & Oblique Ionogram
    """
    def __init__(self, log_dir="chirp_logs"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "chirpsounder_master_log.csv")
        self._ensure_csv_header()

        try:
            self.font_title = ImageFont.truetype("consolab.ttf", 15)
            self.font_bold = ImageFont.truetype("consolab.ttf", 12)
            self.font_reg = ImageFont.truetype("consola.ttf", 11)
            self.font_small = ImageFont.truetype("consola.ttf", 10)
        except Exception:
            default_f = ImageFont.load_default()
            self.font_title = default_f
            self.font_bold = default_f
            self.font_reg = default_f
            self.font_small = default_f

    def _ensure_csv_header(self):
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "intercept_utc", "completed_utc", "est_2mhz_start_utc",
                    "sweep_rate_khz_s", "sentinel_center_mhz", "lowest_detected_mhz",
                    "verified_muf_mhz", "peak_snr_db", "peak_comp_gain_db",
                    "num_hops_sampled", "end_reason", "points_mhz_snr", "snapshot_png"
                ])

    def save_chirp_record(self, chirp_dict, fs_wf=96000.0):
        cid = chirp_dict["id"]
        dt_utc = datetime.fromtimestamp(chirp_dict["t_detect_initial"], tz=timezone.utc)
        ts_slug = dt_utc.strftime("%Y%m%d_%H%M%SZ")
        rate_khz = chirp_dict["rate_hz_s"] / 1000.0
        muf_mhz = chirp_dict["highest_seen_mhz"]

        base_name = f"chirp_{cid:04d}_{ts_slug}_{rate_khz:.0f}kHzs_MUF{muf_mhz:.2f}MHz"
        png_filename = base_name + ".png"
        json_filename = base_name + ".json"
        png_path = os.path.join(self.log_dir, png_filename)
        json_path = os.path.join(self.log_dir, json_filename)

        # 1. Render Composite PNG Snapshot (Waterfall Intercept + Oblique Ionogram)
        self._render_composite_png(chirp_dict, png_path, fs_wf=fs_wf)

        # 2. Save Full JSON Telemetry Record
        pts_formatted = [[round(m, 4), round(s, 2)] for m, s in chirp_dict["points"]]
        json_payload = {
            "id": cid,
            "intercept_utc": chirp_dict["intercept_utc_iso"],
            "completed_utc": chirp_dict.get("completed_utc_iso", ""),
            "est_2mhz_start_utc": chirp_dict["est_2mhz_utc"],
            "sweep_rate_hz_s": chirp_dict["rate_hz_s"],
            "sweep_rate_khz_s": rate_khz,
            "sentinel_center_hz": chirp_dict["sentinel_freq_hz"],
            "lowest_seen_mhz": round(chirp_dict["lowest_seen_mhz"], 4),
            "verified_muf_mhz": round(muf_mhz, 4),
            "peak_snr_db": round(chirp_dict["peak_snr_db"], 2),
            "peak_compression_gain_db": round(chirp_dict["peak_comp_db"], 2),
            "end_reason": chirp_dict.get("end_reason", ""),
            "ionogram_points_mhz_db": pts_formatted,
            "snapshot_png": png_filename
        }
        with open(json_path, mode="w", encoding="utf-8") as jf:
            json.dump(json_payload, jf, indent=2)

        # 3. Append Row to Master CSV
        pts_str = "; ".join(f"{m:.3f}MHz:{s:+.1f}dB" for m, s in chirp_dict["points"])
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                cid,
                chirp_dict["intercept_utc_iso"],
                chirp_dict.get("completed_utc_iso", ""),
                chirp_dict["est_2mhz_utc"],
                f"{rate_khz:.1f}",
                f"{chirp_dict['sentinel_freq_hz'] / 1e6:.4f}",
                f"{chirp_dict['lowest_seen_mhz']:.4f}",
                f"{muf_mhz:.4f}",
                f"{chirp_dict['peak_snr_db']:.2f}",
                f"{chirp_dict['peak_comp_db']:.2f}",
                len(chirp_dict["points"]),
                chirp_dict.get("end_reason", ""),
                pts_str,
                png_filename
            ])

        chirp_dict["saved_png"] = png_filename
        return png_path

    def _render_composite_png(self, c, out_path, fs_wf=96000.0):
        W, H = 1200, 650
        img = Image.new("RGB", (W, H), (11, 14, 22))
        draw = ImageDraw.Draw(img)

        # Header Banner
        draw.rectangle([0, 0, W, 52], fill=(18, 24, 38), outline=(45, 60, 90))
        title_str = (
            f"OBLIQUE FMCW CHIRPSOUNDER INTERCEPT #{c['id']:04d}  |  "
            f"SWEEP RATE: {c['rate_hz_s']/1000:.1f} kHz/s  |  "
            f"VERIFIED MUF: {c['highest_seen_mhz']:.2f} MHz"
        )
        draw.text((16, 8), title_str, fill=(0, 255, 204), font=self.font_title)

        sub_str = (
            f"Intercept UTC: {c['intercept_utc_iso']}   |   "
            f"Est 2.0 MHz Start UTC: {c['est_2mhz_utc']}   |   "
            f"Peak SNR: +{c['peak_snr_db']:.1f} dB (De-Chirp Gain: +{c['peak_comp_db']:.1f} dB)   |   "
            f"Status: {c.get('end_reason', '')}"
        )
        draw.text((16, 31), sub_str, fill=(180, 205, 240), font=self.font_reg)

        # ----------------------------------------------------------------------
        # LEFT PANEL: FROZEN SENTINEL WATERFALL SNAPSHOT AT INTERCEPT
        # ----------------------------------------------------------------------
        wf_x0, wf_y0 = 75, 85
        wf_w, wf_h = 360, 520
        draw.text((wf_x0, 64), f"SENTINEL INTERCEPT WATERFALL ({c['sentinel_freq_hz']/1e6:.3f} MHz)", fill=(0, 230, 255), font=self.font_bold)

        wf_arr = c.get("intercept_wf_rgb")
        if wf_arr is not None:
            wf_pil = Image.fromarray(wf_arr, mode="RGB").resize((wf_w, wf_h), Image.Resampling.BILINEAR)
            img.paste(wf_pil, (wf_x0, wf_y0))
        else:
            draw.rectangle([wf_x0, wf_y0, wf_x0 + wf_w, wf_y0 + wf_h], fill=(8, 10, 16))

        draw.rectangle([wf_x0, wf_y0, wf_x0 + wf_w, wf_y0 + wf_h], outline=(60, 80, 115), width=1)

        # Frequency scale along left edge of waterfall snapshot
        s_center = c["sentinel_freq_hz"]
        for f_off in range(-40000, 40001, 10000):
            rel = ((fs_wf / 2.0) - f_off) / fs_wf
            gy = wf_y0 + int(rel * wf_h)
            rf_khz = (s_center + f_off) / 1000.0
            draw.line([(wf_x0 - 6, gy), (wf_x0, gy)], fill=(120, 140, 180), width=1)
            draw.text((12, gy - 6), f"{rf_khz:.0f}k", fill=(140, 160, 200), font=self.font_small)

        # Draw Radon diagonal detection line on the waterfall snapshot if available
        line_coords = c.get("intercept_line_coords")
        if line_coords is not None:
            f0_off, f1_off, k_frames = line_coords
            y0 = wf_y0 + int(np.clip(((fs_wf / 2.0) - f0_off) / fs_wf, 0.0, 1.0) * wf_h)
            y1 = wf_y0 + int(np.clip(((fs_wf / 2.0) - f1_off) / fs_wf, 0.0, 1.0) * wf_h)
            # Scale X coordinates proportionally to resized width
            orig_w = wf_arr.shape[1] if wf_arr is not None else 480
            x0 = wf_x0 + int((k_frames / orig_w) * wf_w)
            x1 = wf_x0 + 2
            draw.line([(x0, y0), (x1, y1)], fill=(51, 255, 119), width=2)

        # ----------------------------------------------------------------------
        # RIGHT PANEL: OBLIQUE IONOGRAM PLOT (2 MHz to 30 MHz)
        # ----------------------------------------------------------------------
        px0, py0 = 495, 85
        px1, py1 = 1175, 520
        pw = px1 - px0
        ph = py1 - py0

        draw.text((px0, 64), "OBLIQUE IONOGRAM: VERIFIED DE-CHIRP SNR (dB) vs. FREQUENCY (2 - 30 MHz)", fill=(0, 255, 204), font=self.font_bold)
        draw.rectangle([px0, py0, px1, py1], fill=(9, 12, 20), outline=(60, 80, 115), width=1)

        # Frequency X-Grid (2 MHz to 30 MHz)
        for mhz in range(2, 31, 2):
            gx = px0 + int(((mhz - 2.0) / 28.0) * pw)
            draw.line([(gx, py0), (gx, py1)], fill=(22, 30, 48), width=1)
            draw.text((gx - 10, py1 + 8), f"{mhz}M", fill=(130, 150, 185), font=self.font_small)

        # SNR Y-Grid (0 dB to 40 dB)
        for snr_g in range(0, 41, 10):
            gy = py1 - int((snr_g / 40.0) * ph)
            draw.line([(px0, gy), (px1, gy)], fill=(22, 30, 48), width=1)
            draw.text((px0 - 38, gy - 6), f"+{snr_g}dB", fill=(130, 150, 185), font=self.font_small)

        # Plot Ionogram curve & points
        pts = c.get("points", [])
        if pts:
            poly_xy = []
            for mhz_val, snr_val in pts:
                cx = px0 + int(np.clip(((mhz_val - 2.0) / 28.0) * pw, 0, pw))
                cy = py1 - int(np.clip((snr_val / 40.0) * ph, 0, ph))
                poly_xy.append((cx, cy))

            if len(poly_xy) >= 2:
                draw.line(poly_xy, fill=(0, 229, 255), width=2)

            for (mhz_val, snr_val), (cx, cy) in zip(pts, poly_xy):
                dot_col = (51, 255, 119) if snr_val > 0.0 else (255, 70, 70)
                draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=dot_col, outline=(0, 0, 0))

            # Draw Vertical MUF Marker Line
            muf_mhz = c["highest_seen_mhz"]
            mx = px0 + int(np.clip(((muf_mhz - 2.0) / 28.0) * pw, 0, pw))
            draw.line([(mx, py0), (mx, py1)], fill=(255, 204, 0), width=2)
            label_x = mx - 135 if mx > px1 - 150 else mx + 8
            draw.text(
                (label_x, py0 + 14),
                f"MUF: {muf_mhz:.2f} MHz\n({c['rate_hz_s']/1000:.1f} kHz/s)",
                fill=(255, 204, 0), font=self.font_bold
            )

        # Bottom Hop Table Summary inside PNG
        draw.rectangle([px0, 550, px1, 630], fill=(15, 20, 32), outline=(45, 60, 90))
        hop_summary = "  ".join(f"{m:.2f}M:{s:+.1f}dB" for m, s in pts[:12])
        hop_summary2 = "  ".join(f"{m:.2f}M:{s:+.1f}dB" for m, s in pts[12:24])
        draw.text((px0 + 10, 558), "Measured Hops (MHz : De-Chirp SNR):", fill=(136, 221, 255), font=self.font_small)
        draw.text((px0 + 10, 578), hop_summary, fill=(51, 255, 119), font=self.font_small)
        if hop_summary2:
            draw.text((px0 + 10, 598), hop_summary2, fill=(51, 255, 119), font=self.font_small)

        img.save(out_path, format="PNG")


# ==============================================================================
# 3. RADON CHIRP DETECTOR + DIFFERENTIAL DE-CHIRP GAIN VERIFIER
# ==============================================================================

class ChirpsounderEngine:
    CANDIDATE_RATES_HZ_S = np.array([
        50000.0, 62500.0, 100000.0, 125000.0, 200000.0
    ], dtype=np.float64)

    def __init__(self, sdr, archiver, wf_span_hz=96000):
        self.sdr = sdr
        self.archiver = archiver
        self.fs_in = float(sdr.sample_rate)
        self.wf_decim = max(1, int(round(self.fs_in / wf_span_hz)))
        self.fs_wf = self.fs_in / self.wf_decim
        self.wf_taps = firwin(65, 0.88 / self.wf_decim).astype(np.float32)
        self.wf_zi = np.zeros(len(self.wf_taps) - 1, dtype=np.complex64)

        self.fft_size = 4096
        self.frame_hop = int(round(self.fs_wf * 0.025))
        self.dt_frame = self.frame_hop / self.fs_wf
        self.hz_per_bin = self.fs_wf / self.fft_size
        self.fft_window = np.hanning(self.fft_size).astype(np.float32)
        self.wf_ring = np.zeros(self.fft_size, dtype=np.complex64)

        self.dechirp_len = 16384
        self.dechirp_win = np.hanning(self.dechirp_len).astype(np.float32)
        self.dechirp_ring = np.zeros(self.dechirp_len, dtype=np.complex64)
        self.dechirp_fill = 0
        self.wf_accum = 0

        self.hist_len = 32
        self.spec_hist = np.zeros((self.fft_size, self.hist_len), dtype=np.float32)
        self.frames_collected = 0
        self.bg_psd = None

        self.mode = "SENTINEL"
        self.sentinel_freq_hz = int(sdr.center_freq)
        self.chase_enabled = True
        self.max_chase_freq_hz = 29500000
        self.hop_step_hz = 500000
        self.min_compression_gain_db = 6.5

        self.next_chirp_id = 1
        self.active_chirp = None
        self.completed_logs = []
        self.selected_log_idx = 0
        self.last_radon_score = 0.0
        self.last_compression_gain_db = 0.0
        self.last_best_rate_khz = 100.0
        self.detected_line_coords = None
        self.last_saved_png_path = None

        # Callback hook so Engine can grab current RGB waterfall array from GUI at the moment of intercept
        self.get_waterfall_snapshot_fn = None

    def set_sentinel_freq(self, freq_hz):
        self.sentinel_freq_hz = int(freq_hz)
        if self.mode == "SENTINEL":
            self.sdr.set_center_freq(self.sentinel_freq_hz)
            self.reset_history()

    def reset_history(self):
        self.spec_hist.fill(0.0)
        self.frames_collected = 0
        self.dechirp_fill = 0
        self.bg_psd = None
        self.detected_line_coords = None

    def _evaluate_dechirp_compression(self, iq_16k, rate_hz_s, expected_offset_hz=None, gate_bw_hz=12000.0):
        n = len(iq_16k)
        t_centered = (np.arange(n, dtype=np.float64) - (n / 2.0)) / self.fs_wf

        raw_spec = np.fft.fftshift(np.fft.fft(iq_16k * self.dechirp_win))
        raw_psd = 10.0 * np.log10((np.abs(raw_spec) / n) ** 2 + 1e-16)

        dechirped_iq = iq_16k * np.exp(-1j * np.pi * rate_hz_s * (t_centered ** 2)).astype(np.complex64)
        dc_spec = np.fft.fftshift(np.fft.fft(dechirped_iq * self.dechirp_win))
        dc_psd = 10.0 * np.log10((np.abs(dc_spec) / n) ** 2 + 1e-16)

        hz_per_bin = self.fs_wf / n
        noise_ref_db = float(np.percentile(dc_psd, 98))
        median_db = float(np.median(dc_psd))

        if expected_offset_hz is not None:
            c_bin = int(round((expected_offset_hz + (self.fs_wf / 2.0)) / hz_per_bin))
            half_b = max(16, int(round(gate_bw_hz / hz_per_bin)))
            lo = max(16, c_bin - half_b)
            hi = min(n - 16, c_bin + half_b)
        else:
            lo = 64
            hi = n - 64

        if hi <= lo:
            return 0.0, 0.0, 0.0

        sub_dc = dc_psd[lo:hi]
        pk_rel = int(np.argmax(sub_dc))
        pk_bin = lo + pk_rel
        dc_peak_db = float(sub_dc[pk_rel])

        sweep_bins = int(round(3000.0 / hz_per_bin))
        r_lo = max(0, pk_bin - sweep_bins)
        r_hi = min(n, pk_bin + sweep_bins + 1)
        raw_local_peak_db = float(np.max(raw_psd[r_lo:r_hi]))

        compression_gain_db = dc_peak_db - raw_local_peak_db
        excess_above_98th = dc_peak_db - noise_ref_db
        true_snr_db = (dc_peak_db - median_db) if excess_above_98th > 3.5 else 0.0

        f_mid_hz = ((pk_bin / n) - 0.5) * self.fs_wf
        T_win = n / self.fs_wf
        f_now_offset_hz = f_mid_hz + rate_hz_s * (0.5 * T_win)

        return true_snr_db, compression_gain_db, f_now_offset_hz

    def _run_radon_chirp_detector(self):
        if self.frames_collected < 24 or self.dechirp_fill < self.dechirp_len:
            return

        K = 22
        sub_mat = self.spec_hist[:, -K:]
        t_rel = np.arange(-K + 1, 1, dtype=np.float32) * self.dt_frame

        best_score = -100.0
        best_rate = 100000.0
        best_end_bin = self.fft_size // 2

        for rate in self.CANDIDATE_RATES_HZ_S:
            bin_shifts = np.round((rate * t_rel) / self.hz_per_bin).astype(np.int32)
            min_shift = int(bin_shifts[0])

            valid_start = max(20, -min_shift + 4)
            valid_end = self.fft_size - 20
            if valid_end <= valid_start:
                continue

            end_bins = np.arange(valid_start, valid_end, dtype=np.int32)
            bin_grid = end_bins[:, None] + bin_shifts[None, :]
            time_grid = np.arange(K, dtype=np.int32)[None, :]

            traj_vals = np.maximum(
                sub_mat[bin_grid, time_grid],
                np.maximum(sub_mat[bin_grid - 1, time_grid], sub_mat[bin_grid + 1, time_grid])
            )

            active_fraction = np.mean(traj_vals > 5.0, axis=1)
            mean_snr = np.mean(traj_vals, axis=1)

            scores = np.where(active_fraction >= 0.75, mean_snr, -50.0)
            max_idx = int(np.argmax(scores))
            sc = float(scores[max_idx])

            if sc > best_score:
                best_score = sc
                best_rate = float(rate)
                best_end_bin = int(end_bins[max_idx])

        self.last_radon_score = max(0.0, best_score)
        self.last_best_rate_khz = best_rate / 1000.0

        if best_score >= 8.5:
            radon_end_off_hz = (best_end_bin - (self.fft_size / 2.0)) * self.hz_per_bin
            true_snr, comp_gain, f_now_off = self._evaluate_dechirp_compression(
                self.dechirp_ring, best_rate, expected_offset_hz=radon_end_off_hz, gate_bw_hz=15000.0
            )
            self.last_compression_gain_db = comp_gain

            if comp_gain >= self.min_compression_gain_db and true_snr >= 13.0:
                now = time.time()
                f_abs_hz = self.sdr.center_freq + f_now_off

                f_start_offset_hz = f_now_off - best_rate * ((K - 1) * self.dt_frame)
                self.detected_line_coords = (f_start_offset_hz, f_now_off, K)

                sec_since_2mhz = max(0.0, (f_abs_hz - 2000000.0) / best_rate)
                est_start_utc = datetime.fromtimestamp(now - sec_since_2mhz, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-4]
                intercept_iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                cid = self.next_chirp_id
                self.next_chirp_id += 1

                wf_snap = self.get_waterfall_snapshot_fn() if self.get_waterfall_snapshot_fn else None

                self.active_chirp = {
                    "id": cid,
                    "t_detect_initial": now,
                    "t_detect": now,
                    "intercept_utc_iso": intercept_iso,
                    "sentinel_freq_hz": self.sentinel_freq_hz,
                    "f_detect_hz": f_abs_hz,
                    "rate_hz_s": best_rate,
                    "est_2mhz_utc": est_start_utc,
                    "lowest_seen_mhz": f_abs_hz / 1e6,
                    "highest_seen_mhz": f_abs_hz / 1e6,
                    "peak_snr_db": true_snr,
                    "peak_comp_db": comp_gain,
                    "points": [(f_abs_hz / 1e6, true_snr)],
                    "missed_hops": 0,
                    "target_freq_hz": f_abs_hz,
                    "hop_settle_until": 0.0,
                    "intercept_wf_rgb": wf_snap,
                    "intercept_line_coords": self.detected_line_coords
                }

                if self.chase_enabled:
                    self._schedule_next_chase_hop(now)
                else:
                    self._finalize_chirp_log(reason="Passband Intercept Only")

    def _schedule_next_chase_hop(self, now):
        ch = self.active_chirp
        if not ch:
            return

        f_now_hz = ch["f_detect_hz"] + ch["rate_hz_s"] * (now - ch["t_detect"])
        next_center_hz = int(round((f_now_hz + self.hop_step_hz) / 10000.0) * 10000)

        if next_center_hz > self.max_chase_freq_hz:
            self._finalize_chirp_log(reason="Reached 29.5 MHz HF Ceiling")
            return

        self.mode = "CHASE"
        ch["target_freq_hz"] = float(next_center_hz)
        ch["hop_settle_until"] = now + 0.080
        self.sdr.set_center_freq(next_center_hz)
        self.sdr.flush_queue()
        self.dechirp_fill = 0
        self.wf_accum = 0

    def _process_chase_step(self):
        ch = self.active_chirp
        if not ch:
            self.mode = "SENTINEL"
            return

        now = time.time()
        if now < ch["hop_settle_until"] or self.dechirp_fill < self.dechirp_len:
            return

        f_pred_hz = ch["f_detect_hz"] + ch["rate_hz_s"] * (now - ch["t_detect"])
        dist_to_center = f_pred_hz - ch["target_freq_hz"]

        if -12000.0 <= dist_to_center <= 22000.0:
            true_snr, comp_gain, f_now_off = self._evaluate_dechirp_compression(
                self.dechirp_ring, ch["rate_hz_s"], expected_offset_hz=dist_to_center, gate_bw_hz=12000.0
            )
            self.last_compression_gain_db = comp_gain

            if comp_gain >= (self.min_compression_gain_db - 1.0) and true_snr >= 12.0:
                measured_mhz = (ch["target_freq_hz"] + f_now_off) / 1e6
                ch["points"].append((measured_mhz, true_snr))
                ch["highest_seen_mhz"] = max(ch["highest_seen_mhz"], measured_mhz)
                ch["peak_snr_db"] = max(ch["peak_snr_db"], true_snr)
                ch["peak_comp_db"] = max(ch["peak_comp_db"], comp_gain)
                ch["missed_hops"] = 0

                ch["f_detect_hz"] = ch["target_freq_hz"] + f_now_off
                ch["t_detect"] = now

                self._schedule_next_chase_hop(now)

            elif dist_to_center > 10000.0:
                ch["missed_hops"] += 1
                ch["points"].append((ch["target_freq_hz"] / 1e6, 0.0))
                if ch["missed_hops"] >= 2:
                    self._finalize_chirp_log(reason=f"Faded above MUF ({ch['highest_seen_mhz']:.2f} MHz)")
                else:
                    self._schedule_next_chase_hop(now)

        elif dist_to_center > 30000.0:
            ch["missed_hops"] += 1
            if ch["missed_hops"] >= 2:
                self._finalize_chirp_log(reason=f"Faded above MUF ({ch['highest_seen_mhz']:.2f} MHz)")
            else:
                self._schedule_next_chase_hop(now)

    def _finalize_chirp_log(self, reason="Completed"):
        if self.active_chirp:
            ch = self.active_chirp
            ch["end_reason"] = reason
            ch["completed_utc_iso"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            # Automatically archive CSV row, JSON telemetry, and composite PNG snapshot!
            try:
                saved_png = self.archiver.save_chirp_record(ch, fs_wf=self.fs_wf)
                self.last_saved_png_path = saved_png
            except Exception as e:
                sys.stderr.write(f"[!] Error saving chirp snapshot: {e}\n")

            # Release large numpy RGB snapshot from RAM after saving to PNG
            ch["intercept_wf_rgb"] = None

            self.completed_logs.insert(0, ch)
            self.selected_log_idx = 0
            if len(self.completed_logs) > 50:
                self.completed_logs.pop()
            self.active_chirp = None

        self.mode = "SENTINEL"
        self.sdr.set_center_freq(self.sentinel_freq_hz)
        self.sdr.flush_queue()
        self.reset_history()

    def process_iq_chunk(self, iq_samples):
        if len(iq_samples) == 0:
            return None

        iq_samples = iq_samples - np.mean(iq_samples)
        wf_filt, self.wf_zi = lfilter(self.wf_taps, 1.0, iq_samples, zi=self.wf_zi)
        wf_iq = wf_filt[::self.wf_decim]
        n_wf = len(wf_iq)
        if n_wf == 0:
            return None

        if n_wf >= self.dechirp_len:
            self.dechirp_ring[:] = wf_iq[-self.dechirp_len:]
            self.dechirp_fill = self.dechirp_len
        else:
            self.dechirp_ring = np.roll(self.dechirp_ring, -n_wf)
            self.dechirp_ring[-n_wf:] = wf_iq
            self.dechirp_fill = min(self.dechirp_len, self.dechirp_fill + n_wf)

        latest_psd = None
        idx = 0
        while idx < n_wf:
            take = min(n_wf - idx, self.frame_hop - self.wf_accum, self.fft_size)
            self.wf_ring = np.roll(self.wf_ring, -take)
            self.wf_ring[-take:] = wf_iq[idx:idx + take]
            self.wf_accum += take
            idx += take

            if self.wf_accum >= self.frame_hop:
                self.wf_accum = 0
                windowed = self.wf_ring * self.fft_window
                spec = np.fft.fftshift(np.fft.fft(windowed))
                psd_db = 10.0 * np.log10((np.abs(spec) / self.fft_size) ** 2 + 1e-16)
                latest_psd = psd_db

                if self.mode == "SENTINEL":
                    if self.bg_psd is None:
                        self.bg_psd = psd_db.copy()
                    else:
                        self.bg_psd = 0.94 * self.bg_psd + 0.06 * psd_db

                    whitened = np.maximum(0.0, psd_db - self.bg_psd)
                    self.spec_hist[:, :-1] = self.spec_hist[:, 1:]
                    self.spec_hist[:, -1] = whitened
                    self.frames_collected += 1

                    if self.frames_collected >= 24 and (self.frames_collected % 4 == 0):
                        self._run_radon_chirp_detector()
                else:
                    self._process_chase_step()

        return latest_psd


# ==============================================================================
# 4. SIDEWAYS WATERFALL + LIVE OBLIQUE IONOGRAM & INTERACTIVE LOG VIEWER GUI
# ==============================================================================

def build_sdr_colormap():
    lut = np.zeros((256, 3), dtype=np.uint8)
    stops = [
        (0,   (0, 0, 18)),
        (64,  (0, 35, 130)),
        (128, (0, 190, 210)),
        (185, (240, 230, 30)),
        (230, (255, 50, 0)),
        (255, (255, 255, 255))
    ]
    for i in range(len(stops) - 1):
        idx0, c0 = stops[i]
        idx1, c1 = stops[i + 1]
        span = idx1 - idx0
        for j in range(span + 1):
            t = j / span
            lut[idx0 + j, 0] = int(c0[0] + t * (c1[0] - c0[0]))
            lut[idx0 + j, 1] = int(c0[1] + t * (c1[1] - c0[1]))
            lut[idx0 + j, 2] = int(c0[2] + t * (c1[2] - c0[2]))
    return lut


class ChirpsounderHunterApp:
    def __init__(self, root, sdr, engine):
        self.root = root
        self.sdr = sdr
        self.engine = engine
        self.engine.get_waterfall_snapshot_fn = lambda: self.wf_rgb.copy()

        self.root.title("Airspy HF+ Chirpsounder Hunter (Auto CSV/JSON Logger + Ionogram PNG Snapshotter)")
        self.root.geometry("1420x830")
        self.root.configure(bg="#101218")

        self.canvas_w = 1400
        self.canvas_h = 720
        self.axis_w = 95
        self.spec_w = 110
        self.wf_x0 = self.axis_w + self.spec_w
        self.wf_w = 480
        self.right_x0 = self.wf_x0 + self.wf_w

        self.cmap = build_sdr_colormap()
        self.wf_rgb = np.zeros((self.canvas_h, self.wf_w, 3), dtype=np.uint8)
        self.smoothed_psd = None

        self.db_min = tk.DoubleVar(value=-105.0)
        self.db_max = tk.DoubleVar(value=-45.0)
        self.comp_var = tk.DoubleVar(value=6.5)
        self.chase_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.sdr.start()
        self.root.after(20, self.poll_iq_queue)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101218")
        style.configure("TLabel", background="#101218", foreground="#dcdce6", font=("Consolas", 10))
        style.configure("TButton", font=("Consolas", 10, "bold"))
        style.configure("TCheckbutton", background="#101218", foreground="#00ffcc", font=("Consolas", 10, "bold"))

        ctrl = ttk.Frame(self.root, padding=(8, 6))
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Sentinel(Hz):").pack(side=tk.LEFT, padx=2)
        self.center_var = tk.StringVar(value=str(self.engine.sentinel_freq_hz))
        center_entry = ttk.Entry(ctrl, textvariable=self.center_var, width=9, font=("Consolas", 10))
        center_entry.pack(side=tk.LEFT, padx=2)
        center_entry.bind("<Return>", lambda e: self.apply_sentinel_freq())
        ttk.Button(ctrl, text="Set Watch", width=9, command=self.apply_sentinel_freq).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Checkbutton(
            ctrl, text="Auto-Chase Up to 30 MHz",
            variable=self.chase_var,
            command=lambda: setattr(self.engine, "chase_enabled", bool(self.chase_var.get()))
        ).pack(side=tk.LEFT, padx=6)

        ttk.Label(ctrl, text="Min De-Chirp Gain(dB):").pack(side=tk.LEFT, padx=(8, 2))
        tk.Scale(
            ctrl, variable=self.comp_var, from_=4.0, to=14.0, resolution=0.5,
            orient=tk.HORIZONTAL, length=80, bg="#101218", fg="#00ffcc", highlightthickness=0,
            command=lambda v: setattr(self.engine, "min_compression_gain_db", float(v))
        ).pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Floor:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Scale(ctrl, variable=self.db_min, from_=-135, to=-60, orient=tk.HORIZONTAL, length=60,
                 bg="#101218", fg="#dcdce6", highlightthickness=0).pack(side=tk.LEFT)

        ttk.Label(ctrl, text="Ceil:").pack(side=tk.LEFT, padx=(6, 2))
        tk.Scale(ctrl, variable=self.db_max, from_=-90, to=-10, orient=tk.HORIZONTAL, length=60,
                 bg="#101218", fg="#dcdce6", highlightthickness=0).pack(side=tk.LEFT)

        ttk.Button(ctrl, text="Open Logs Folder", command=self.open_logs_folder).pack(side=tk.RIGHT, padx=4)
        ttk.Button(ctrl, text="Simulate Chirp Test", command=self.inject_test_chirp).pack(side=tk.RIGHT, padx=6)

        hdr = ttk.Frame(self.root, padding=(10, 2))
        hdr.pack(fill=tk.X)
        self.status_lbl = ttk.Label(
            hdr,
            text=f"LOGGING TO: {self.engine.archiver.log_dir} | Click any row in Log Table below to view its Ionogram",
            font=("Consolas", 10, "bold"), foreground="#33ff77"
        )
        self.status_lbl.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            self.root, width=self.canvas_w, height=self.canvas_h,
            bg="#080a10", highlightthickness=1, highlightbackground="#2a3040"
        )
        self.canvas.pack(padx=8, pady=4)
        self.wf_image_item = self.canvas.create_image(self.wf_x0, 0, anchor=tk.NW)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def apply_sentinel_freq(self):
        try:
            val = int(float(self.center_var.get()))
            self.engine.set_sentinel_freq(val)
            self.wf_rgb.fill(0)
        except ValueError:
            pass

    def open_logs_folder(self):
        log_dir = self.engine.archiver.log_dir
        if os.name == "nt":
            os.startfile(log_dir)

    def on_canvas_click(self, event):
        """Allows clicking any row in the bottom-right log table to display its Ionogram curve."""
        px0 = self.right_x0 + 48
        ly0 = 375
        if event.x >= px0 and event.y >= ly0 + 30:
            row_idx = (event.y - (ly0 + 30)) // 22
            if self.engine.active_chirp:
                row_idx -= 1
            if 0 <= row_idx < len(self.engine.completed_logs):
                self.engine.selected_log_idx = int(row_idx)

    def inject_test_chirp(self):
        fs = self.sdr.sample_rate
        dur = 0.85
        n = int(fs * dur)
        t = np.arange(n, dtype=np.float64) / fs
        f0 = -38000.0
        k = 100000.0
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * (t ** 2))
        chirp_iq = (0.015 * np.exp(1j * phase)).astype(np.complex64)
        for i in range(0, n, 8192):
            try:
                self.sdr.iq_queue.put_nowait(chirp_iq[i:i + 8192])
            except queue.Full:
                break

    def freq_offset_to_y(self, offset_hz):
        span = self.engine.fs_wf
        rel = ((span / 2.0) - offset_hz) / span
        return int(rel * self.canvas_h)

    def render_frame(self, psd_db):
        if self.smoothed_psd is None:
            self.smoothed_psd = psd_db.copy()
        else:
            self.smoothed_psd = 0.55 * self.smoothed_psd + 0.45 * psd_db

        col_raw = psd_db[::-1]
        col_smooth = self.smoothed_psd[::-1]

        y_old = np.linspace(0, 1, len(col_raw))
        y_new = np.linspace(0, 1, self.canvas_h)
        col_db = np.interp(y_new, y_old, col_raw)
        spec_db = np.interp(y_new, y_old, col_smooth)

        vmin = self.db_min.get()
        vmax = max(vmin + 5.0, self.db_max.get())

        norm = np.clip((col_db - vmin) / (vmax - vmin), 0.0, 1.0)
        color_indices = (norm * 255.0).astype(np.uint8)
        self.wf_rgb[:, 1:, :] = self.wf_rgb[:, :-1, :]
        self.wf_rgb[:, 0, :] = self.cmap[color_indices]

        img = Image.fromarray(self.wf_rgb, mode="RGB")
        self.wf_photo = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.wf_image_item, image=self.wf_photo)

        self.canvas.delete("overlay")

        saved_note = f" | Last Saved: {os.path.basename(self.engine.last_saved_png_path)}" if self.engine.last_saved_png_path else ""
        if self.engine.mode == "SENTINEL":
            self.status_lbl.config(
                text=f"SENTINEL @ {self.sdr.center_freq/1e6:.4f} MHz | "
                     f"Radon: +{self.engine.last_radon_score:.1f}dB ({self.engine.last_best_rate_khz:.0f}kHz/s) | "
                     f"De-Chirp Gain: {self.engine.last_compression_gain_db:+.1f}dB{saved_note}",
                foreground="#33ff77"
            )
        else:
            ch = self.engine.active_chirp
            self.status_lbl.config(
                text=f"CHASING SOUNDER #{ch['id']} ({ch['rate_hz_s']/1000:.1f} kHz/s) | "
                     f"Hop: {self.sdr.center_freq/1e6:.3f} MHz | De-Chirp Gain: {self.engine.last_compression_gain_db:+.1f} dB | "
                     f"Verified MUF: {ch['highest_seen_mhz']:.2f} MHz",
                foreground="#ffcc00"
            )

        self.canvas.create_line(self.axis_w, 0, self.axis_w, self.canvas_h, fill="#2a3040", width=1, tags="overlay")
        self.canvas.create_line(self.wf_x0, 0, self.wf_x0, self.canvas_h, fill="#3a4560", width=2, tags="overlay")
        self.canvas.create_line(self.right_x0, 0, self.right_x0, self.canvas_h, fill="#00ffcc", width=2, tags="overlay")

        for f_off in range(-40000, 40001, 10000):
            gy = self.freq_offset_to_y(f_off)
            rf_khz = (self.sdr.center_freq + f_off) / 1000.0
            self.canvas.create_line(self.axis_w - 6, gy, self.wf_x0, gy, fill="#1c2230", dash=(2, 4), tags="overlay")
            self.canvas.create_text(
                self.axis_w - 8, gy, text=f"{rf_khz:.0f}k",
                fill="#8899bb", anchor=tk.E, font=("Consolas", 9), tags="overlay"
            )

        spec_norm = np.clip((spec_db - vmin) / (vmax - vmin), 0.0, 1.0)
        xs = self.axis_w + 4 + spec_norm * (self.spec_w - 10)
        coords = np.empty(self.canvas_h * 2, dtype=np.int32)
        coords[0::2] = xs.astype(np.int32)
        coords[1::2] = np.arange(self.canvas_h)
        self.canvas.create_line(*coords.tolist(), fill="#00e5ff", width=1, tags="overlay")

        if self.engine.detected_line_coords is not None:
            f0_off, f1_off, k_frames = self.engine.detected_line_coords
            y0 = self.freq_offset_to_y(f0_off)
            y1 = self.freq_offset_to_y(f1_off)
            x0 = self.wf_x0 + k_frames
            x1 = self.wf_x0 + 1
            self.canvas.create_line(x0, y0, x1, y1, fill="#33ff77", width=2, dash=(4, 2), tags="overlay")

        # RIGHT HALF PANE 1 (TOP): REAL-TIME / SELECTED OBLIQUE IONOGRAM
        px0 = self.right_x0 + 48
        px1 = self.canvas_w - 20
        py0 = 40
        py1 = 330
        pw = px1 - px0
        ph = py1 - py0

        plot_chirp = self.engine.active_chirp
        if not plot_chirp and self.engine.completed_logs:
            sel = min(self.engine.selected_log_idx, len(self.engine.completed_logs) - 1)
            plot_chirp = self.engine.completed_logs[sel]

        ion_title = (
            f"OBLIQUE IONOGRAM (CHIRP #{plot_chirp['id']} | {plot_chirp.get('saved_png', 'LIVE')})"
            if plot_chirp else "LIVE OBLIQUE IONOGRAM: VERIFIED DE-CHIRP SNR (dB) vs. FREQUENCY (2 - 30 MHz)"
        )
        self.canvas.create_text(px0, 14, text=ion_title, fill="#00ffcc", anchor=tk.W, font=("Consolas", 10, "bold"), tags="overlay")
        self.canvas.create_rectangle(px0, py0, px1, py1, fill="#0b0e17", outline="#334466", width=1, tags="overlay")

        for mhz in range(2, 31, 2):
            gx = px0 + int(((mhz - 2.0) / 28.0) * pw)
            self.canvas.create_line(gx, py0, gx, py1, fill="#161c2e", dash=(2, 4), tags="overlay")
            self.canvas.create_text(gx, py1 + 12, text=f"{mhz}M", fill="#7788aa", font=("Consolas", 8), tags="overlay")

        for snr_g in range(0, 41, 10):
            gy = py1 - int((snr_g / 40.0) * ph)
            self.canvas.create_line(px0, gy, px1, gy, fill="#161c2e", dash=(2, 4), tags="overlay")
            self.canvas.create_text(px0 - 6, gy, text=f"+{snr_g}dB", fill="#7788aa", anchor=tk.E, font=("Consolas", 8), tags="overlay")

        if plot_chirp and plot_chirp["points"]:
            pts = plot_chirp["points"]
            poly_coords = []
            for mhz_val, snr_val in pts:
                cx = px0 + int(np.clip(((mhz_val - 2.0) / 28.0) * pw, 0, pw))
                cy = py1 - int(np.clip((snr_val / 40.0) * ph, 0, ph))
                poly_coords.extend([cx, cy])
                col = "#33ff77" if snr_val > 0.0 else "#ff5555"
                self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=col, outline="#000000", tags="overlay")

            if len(poly_coords) >= 4:
                self.canvas.create_line(*poly_coords, fill="#00e5ff", width=2, tags="overlay")

            muf_mhz = plot_chirp["highest_seen_mhz"]
            mx = px0 + int(np.clip(((muf_mhz - 2.0) / 28.0) * pw, 0, pw))
            self.canvas.create_line(mx, py0, mx, py1, fill="#ffcc00", width=2, dash=(4, 3), tags="overlay")
            self.canvas.create_text(
                mx + 6, py0 + 12,
                text=f"MUF: {muf_mhz:.2f} MHz\n({plot_chirp['rate_hz_s']/1000:.1f} kHz/s)",
                fill="#ffcc00", anchor=tk.NW, font=("Consolas", 9, "bold"), tags="overlay"
            )

        # RIGHT HALF PANE 2 (BOTTOM): INTERACTIVE ARCHIVED CHIRPSOUNDER LOG TABLE
        ly0 = 375
        self.canvas.create_text(
            px0, ly0 - 18,
            text="ARCHIVED CHIRPSOUNDER LOGS (CLICK ANY ROW TO INSPECT IONOGRAM | AUTO-SAVED TO ./chirp_logs/)",
            fill="#00ffcc", anchor=tk.W, font=("Consolas", 9, "bold"), tags="overlay"
        )
        self.canvas.create_rectangle(px0, ly0, px1, self.canvas_h - 12, fill="#0b0e17", outline="#334466", width=1, tags="overlay")

        headers = "ID   RATE         EST 2MHz UTC   START MHz   MUF MHz     COMP GAIN   SAVED SNAPSHOT"
        self.canvas.create_text(px0 + 8, ly0 + 14, text=headers, fill="#88ddff", anchor=tk.W, font=("Consolas", 8, "bold"), tags="overlay")
        self.canvas.create_line(px0, ly0 + 26, px1, ly0 + 26, fill="#223048", width=1, tags="overlay")

        rows_to_draw = []
        if self.engine.active_chirp:
            c = self.engine.active_chirp
            rows_to_draw.append((
                -1,
                f"#{c['id']:<3d} {c['rate_hz_s']/1000:5.1f} kHz/s  {c['est_2mhz_utc']:<14s} "
                f"{c['lowest_seen_mhz']:6.3f}M     {c['highest_seen_mhz']:6.3f}M     "
                f"+{c['peak_comp_db']:4.1f} dB    [CHASING & RECORDING...]",
                "#ffcc00"
            ))

        for idx_c, c in enumerate(self.engine.completed_logs[:12]):
            png_short = c.get("saved_png", c["end_reason"])
            if len(png_short) > 30:
                png_short = "..." + png_short[-27:]
            rows_to_draw.append((
                idx_c,
                f"#{c['id']:<3d} {c['rate_hz_s']/1000:5.1f} kHz/s  {c['est_2mhz_utc']:<14s} "
                f"{c['lowest_seen_mhz']:6.3f}M     {c['highest_seen_mhz']:6.3f}M     "
                f"+{c['peak_comp_db']:4.1f} dB    {png_short}",
                "#00ffff" if idx_c == self.engine.selected_log_idx else "#33ff77"
            ))

        if not rows_to_draw:
            self.canvas.create_text(
                px0 + 16, ly0 + 50,
                text="Listening for diagonal FMCW chirpsounders... (Click 'Simulate Chirp Test' top-right to test logger)",
                fill="#556680", anchor=tk.W, font=("Consolas", 9), tags="overlay"
            )
        else:
            for r_idx, (log_idx, r_txt, r_col) in enumerate(rows_to_draw):
                ry = ly0 + 40 + r_idx * 22
                if ry < self.canvas_h - 20:
                    if log_idx == self.engine.selected_log_idx and not self.engine.active_chirp:
                        self.canvas.create_rectangle(px0 + 2, ry - 10, px1 - 2, ry + 10, fill="#162838", outline="", tags="overlay")
                    self.canvas.create_text(px0 + 8, ry, text=r_txt, fill=r_col, anchor=tk.W, font=("Consolas", 8, "bold"), tags="overlay")

    def poll_iq_queue(self):
        latest_psd = None
        for _ in range(25):
            try:
                iq_chunk = self.sdr.iq_queue.get_nowait()
            except queue.Empty:
                break

            psd = self.engine.process_iq_chunk(iq_chunk)
            if psd is not None:
                latest_psd = psd

        if latest_psd is not None:
            self.render_frame(latest_psd)

        self.root.after(20, self.poll_iq_queue)

    def on_close(self):
        self.sdr.stop_and_close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Airspy HF+ FMCW Chirpsounder Hunter & Automated Ionogram Archiver")
    parser.add_argument("--dll-dir", type=str, default=".", help="Folder containing airspyhf.dll")
    parser.add_argument("--freq", type=int, default=5500000, help="Sentinel tripwire frequency in Hz (default: 5500000)")
    parser.add_argument("--log-dir", type=str, default="chirp_logs", help="Output folder for CSV, JSON, and PNG snapshots")
    args = parser.parse_args()

    sdr = AirspyHFDevice(dll_dir=args.dll_dir)
    try:
        sdr.open_and_configure(freq_hz=args.freq, lna=True, agc=True)
    except Exception as e:
        sys.stderr.write(f"[!] Hardware Initialization Error: {e}\n")
        sys.exit(1)

    archiver = ChirpsounderArchiver(log_dir=args.log_dir)
    engine = ChirpsounderEngine(sdr=sdr, archiver=archiver, wf_span_hz=96000)

    root = tk.Tk()
    app = ChirpsounderHunterApp(root, sdr, engine)
    root.mainloop()


if __name__ == "__main__":
    main()