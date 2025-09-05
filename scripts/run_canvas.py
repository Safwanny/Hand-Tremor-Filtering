import argparse, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from htfilter.filters import (
    moving_average, ema, savgol,
    butter_lowpass, bessel_lowpass, notch_then_butter,
    one_euro_series, kalman_series
)

# Make sure we have an interactive backend
try:
    matplotlib.get_backend()
except Exception:
    matplotlib.use("TkAgg")

def apply_filter(name, x, fs, kw):
    n = name.lower()
    if n == "moving_avg": return moving_average(x, k=int(kw.get("ma",7)))
    if n == "ema":        return ema(x, alpha=float(kw.get("alpha",0.2)))
    if n == "savgol":     return savgol(x, win=int(kw.get("win",9)), poly=int(kw.get("poly",3)))
    if n == "butter":     return butter_lowpass(x, fs, fc=float(kw.get("fc",5.0)), order=int(kw.get("order",2)))
    if n == "bessel":     return bessel_lowpass(x, fs, fc=float(kw.get("fc",5.0)), order=int(kw.get("order",2)))
    if n == "notch_then_butter":
        return notch_then_butter(x, fs,
                                 notch_f0=float(kw.get("notch_f0",8.0)),
                                 notch_q=float(kw.get("notch_q",30.0)),
                                 fc=float(kw.get("fc",5.0)),
                                 order=int(kw.get("order",2)))
    if n == "oneeuro":
        return one_euro_series(x, freq=fs,
                               min_cutoff=float(kw.get("oneeuro_min",1.0)),
                               beta=float(kw.get("oneeuro_beta",0.02)),
                               dcutoff=float(kw.get("oneeuro_dcut",1.0)))
    if n == "kalman":
        return kalman_series(x, fs,
                             q_pos=float(kw.get("q_pos",1e-3)),
                             q_vel=float(kw.get("q_vel",1e-2)),
                             r_meas=float(kw.get("r_meas",1e-2)))
    return np.asarray(x, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="butter",
                    choices=["moving_avg","ema","savgol","butter","bessel","notch_then_butter","oneeuro","kalman"])
    ap.add_argument("--fc", type=float, default=5.0)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ma", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--win", type=int, default=9)
    ap.add_argument("--poly", type=int, default=3)
    ap.add_argument("--notch_f0", type=float, default=8.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    ap.add_argument("--oneeuro_min", type=float, default=1.0)
    ap.add_argument("--oneeuro_beta", type=float, default=0.02)
    ap.add_argument("--oneeuro_dcut", type=float, default=1.0)
    ap.add_argument("--fs_hint", type=float, default=120.0)
    args = ap.parse_args()

    # Buffers
    T, X, Y = [], [], []
    t0 = time.time()
    is_drawing = False

    # Figure layout:
    #  - ax0: path (raw in light, filtered in bold)
    #  - ax1: time-series X raw vs filtered
    #  - ax2: spectrum of X
    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2,2, height_ratios=[2,1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    raw_path,   = ax0.plot([], [], lw=1, alpha=0.6, label="raw path")
    filt_path,  = ax0.plot([], [], lw=2, label=f"{args.filter} path")
    ax0.set_title("Draw inside this window (hold trackpad button to 'ink')")
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.invert_yaxis()
    ax0.legend(loc="upper right")

    raw_ts,  = ax1.plot([], [], lw=1, label="raw x")
    filt_ts, = ax1.plot([], [], lw=2, label=f"{args.filter} x")
    ax1.set_title("Time-series (X)"); ax1.legend(loc="upper right")

    spec_line, = ax2.plot([], [])
    ax2.set_title("Magnitude Spectrum (X)"); ax2.set_xlim(0, 20)

    # Event handlers (no OS permissions needed)
    def on_press(event):
        nonlocal is_drawing
        is_drawing = True

    def on_release(event):
        nonlocal is_drawing
        is_drawing = False

    def on_motion(event):
        # event.xdata/ydata are None if outside axes; use figure-relative pixels as fallback
        if event.inaxes != ax0:
            return
        t = time.time() - t0
        x = event.xdata; y = event.ydata
        if x is None or y is None:
            return
        # Record regardless; emphasize strokes when mouse pressed
        T.append(t); X.append(x); Y.append(y)

    cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
    cid2 = fig.canvas.mpl_connect('button_release_event', on_release)
    cid3 = fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.ion()
    last_redraw = 0.0

    while plt.fignum_exists(fig.number):
        now = time.time()
        if now - last_redraw > 0.03 and len(T) > 10:
            t = np.array(T); x = np.array(X); y = np.array(Y)
            # Estimate fs from timestamps
            dt = np.diff(t)
            dt = dt[(dt>0) & (dt<1.0)]
            fs = 1.0/np.median(dt) if dt.size>5 else float(args.fs_hint)

            xf = apply_filter(args.filter, x, fs, vars(args))
            yf = apply_filter(args.filter, y, fs, vars(args))

            # Update path view (normalize path extents)
            raw_path.set_data(x, y)
            filt_path.set_data(xf, yf)
            ax0.relim(); ax0.autoscale_view()

            # Update time-series (last N seconds)
            Nmax = min(len(t), 4000)
            raw_ts.set_data(t[-Nmax:], x[-Nmax:])
            filt_ts.set_data(t[-Nmax:], xf[-Nmax:])
            ax1.relim(); ax1.autoscale_view()

            # Update spectrum (X)
            if len(x) > 256:
                Xd = x - np.mean(x)
                N = min(4096, len(Xd))
                win = np.hanning(N)
                f = np.fft.rfftfreq(N, d=1.0/fs)
                mag = np.abs(np.fft.rfft(Xd[-N:]*win))
                spec_line.set_data(f, mag)
                ax2.set_xlim(0, 20)
                ax2.relim(); ax2.autoscale_view(scaley=True)

            fig.suptitle(f"fs≈{fs:.1f} Hz | filter={args.filter}")
            plt.pause(0.001)
            last_redraw = now
        else:
            plt.pause(0.01)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()
