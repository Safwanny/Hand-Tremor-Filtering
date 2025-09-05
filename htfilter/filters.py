import numpy as np
from scipy.signal import butter, bessel, filtfilt, savgol_filter, iirnotch

# --- helpers ---------------------------------------------------------------

def _safe_norm_cutoff(fc, fs, lo=1e-3, hi=0.99):
    """Return a normalized cutoff Wn in (0,1) even if fs is momentarily bad."""
    if fs is None or fs <= 0:
        return 0.5  # benign default
    wn = float(fc) / (float(fs) / 2.0)
    return float(np.clip(wn, lo, hi))

def _safe_notch_w0(f0, fs, lo=1e-3, hi=0.99):
    """Return normalized notch center in (0,1)."""
    if fs is None or fs <= 0:
        return 0.25
    w0 = float(f0) / (float(fs) / 2.0)
    return float(np.clip(w0, lo, hi))

# --- Simple smoothers ------------------------------------------------------

def moving_average(x, k=5):
    x = np.asarray(x, dtype=float)
    if k <= 1 or x.size == 0:
        return x
    k = int(k)
    w = np.ones(k)/k
    return np.convolve(x, w, mode="same")

def ema(x, alpha=0.2):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    a = float(alpha)
    for i in range(1, len(x)):
        y[i] = a*x[i] + (1-a)*y[i-1]
    return y

def savgol(x, win=9, poly=3):
    x = np.asarray(x, dtype=float)
    win = int(win)
    if win % 2 == 0:
        win += 1
    if x.size < win:
        return x
    poly = min(int(poly), win-1)
    return savgol_filter(x, window_length=win, polyorder=poly)

# --- Classic IIRs (with safe cutoffs) -------------------------------------

def butter_lowpass(x, fs, fc=5.0, order=2):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    wn = _safe_norm_cutoff(fc, fs)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x, method="gust")

def bessel_lowpass(x, fs, fc=5.0, order=2):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    wn = _safe_norm_cutoff(fc, fs)
    b, a = bessel(order, wn, btype="low", norm='phase')
    return filtfilt(b, a, x, method="gust")

def notch_then_butter(x, fs, notch_f0=8.0, notch_q=30.0, fc=5.0, order=2):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    w0 = _safe_notch_w0(notch_f0, fs)
    bn, an = iirnotch(w0, notch_q)
    y = filtfilt(bn, an, x, method="gust")
    wn = _safe_norm_cutoff(fc, fs)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, y, method="gust")

# --- One-Euro --------------------------------------------------------------

class OneEuro:
    def __init__(self, freq, min_cutoff=1.0, beta=0.02, dcutoff=1.0):
        self.freq = float(max(freq, 1e-3))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_prev = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff):
        return 1.0/(1.0 + (self.freq/(2.0*np.pi*max(cutoff, 1e-3))))

    def __call__(self, x):
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        ad = self._alpha(self.dcutoff)
        dx_hat = ad*dx + (1-ad)*self._dx_prev
        cutoff = self.min_cutoff + self.beta*abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self._x_prev
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat

def one_euro_series(x, freq, min_cutoff=1.0, beta=0.02, dcutoff=1.0):
    f = OneEuro(freq, min_cutoff, beta, dcutoff)
    return np.array([f(xi) for xi in x], dtype=float)

# --- Tiny constant-velocity Kalman (1D) -----------------------------------

class Kalman1D:
    def __init__(self, dt=0.01, q_pos=1e-3, q_vel=1e-2, r_meas=1e-2):
        self.dt = float(max(dt, 1e-3))
        self.x = np.zeros((2,1))
        self.P = np.eye(2)*1e-1
        self.Q = np.array([[q_pos, 0],[0, q_vel]], dtype=float)
        self.R = np.array([[r_meas]], dtype=float)

    def update(self, z):
        F = np.array([[1, self.dt],[0, 1]], dtype=float)
        H = np.array([[1, 0]], dtype=float)
        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        # Update
        y = np.array([[float(z)]]) - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(2) - (K @ H)) @ self.P
        return float(self.x[0,0])

def kalman_series(x, fs, q_pos=1e-3, q_vel=1e-2, r_meas=1e-2):
    dt = 1.0/float(max(fs, 1e-3))
    kf = Kalman1D(dt, q_pos, q_vel, r_meas)
    return np.array([kf.update(xi) for xi in x], dtype=float)
