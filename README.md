# Hand Tremor Filtering

this project was made to study and experiement with different filter techniques to reduce unwanted movement and ensure a uniform and controlled path.

this was achieved by using the a trackpad and matplotlib for visualization.

## project structure

```markdown
hand-tremor-filtering/
├─ requirements.txt
├─ README.md
├─ htfilter/
│  ├─ __init__.py
│  └─ filters.py
└─ scripts/
   ├─ __init__.py
   └─ run_canvas.py
```
## getting started
### 1. install dependencies
```bash
pip install -r requirements.txt
```
### 2. using the filters
```bash
# Smooth and stable; good general demo
python -m scripts.run_canvas --filter butter --fc 5 --order 2

# Phase-friendly alternative (less passband distortion)
python -m scripts.run_canvas --filter bessel --fc 5 --order 2

# Notch the peak, then smooth
python -m scripts.run_canvas --filter notch_then_butter --notch_f0 8 --fc 5 --order 2

# One-Euro: increase beta for more responsiveness (less smoothing on fast moves)
python -m scripts.run_canvas --filter oneeuro --oneeuro_min 1.0 --oneeuro_beta 0.02

# Moving average (window size ~ "buffer length")
python -m scripts.run_canvas --filter moving_avg --ma 9

# Exponential moving average (alpha↑ = snappier, alpha↓ = smoother)
python -m scripts.run_canvas --filter ema --alpha 0.25

# Savitzky–Golay (shape-preserving, use odd window; poly < window)
python -m scripts.run_canvas --filter savgol --win  nine  --poly 3

# Tiny constant-velocity Kalman (tune noise terms later if you like)
python -m scripts.run_canvas --filter kalman
```

### 3. How to use the UI
Top panel: your path. Thin line = raw; thicker line = filtered.
Bottom-left: X-coordinate over time (raw vs. filtered).
Bottom-right: X-axis spectrum; tremor typically shows a bump ~6–12 Hz.
Window title shows estimated sampling rate fs≈… Hz and the active filter.
 
## Tweaking parameters
Each filter comes with adjustable parameters to control smoothness, responsiveness, and tremor suppression strength.
Here are some tuning tips:
### Cutoff frequency (--fc)
- Lower values → stronger smoothing (removes more tremor, but laggy)
- Higher values → more responsive (less smoothing, keeps tremor)
### Filter order (--order)
- Higher = sharper cutoff (stronger but may overshoot/oscillate)
### One-Euro filter (--oneeuro_min, --oneeuro_beta)
- --oneeuro_min: baseline cutoff
- --oneeuro_beta: how much cutoff adapts with speed (higher = more responsive strokes)
### Moving Average (--ma)
- Window length in samples (larger = smoother, slower response)
### Exponential Moving Average (--alpha)
- α close to 1 → fast reaction, little smoothing
- α small (e.g., 0.1) → heavy smoothing
### Savitzky–Golay (--win, --poly)
- --win: odd number window size
- --poly: polynomial order (< window size)
### Kalman filter (--filter kalman)
- Works well for slow, deliberate motion
- Try --fc 3 for steady drawing, --fc 6–8 for faster strokes

## Notes
Filters can be combined or extended; check htfilter/filters.py to implement your own.

## Suggested Experiments
- Compare Butterworth vs. Bessel at the same cutoff.
- Draw circles with Moving Average using different window sizes (--ma 5, --ma 15).
- Try Kalman filter on both slow and fast motion.
- Test One-Euro filter with different --oneeuro_beta values and see how responsiveness changes.
- Explore tremor frequency peaks in the spectrum view to tune notch filters.
