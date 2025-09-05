# Hand-Tremor-Filtering

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

# tip for kalman: For slow deliberate motion, try --fc 3. For faster strokes, try --fc 6–8.
```

### 3. How to use the UI
Top panel: your path. Thin line = raw; thicker line = filtered.
Bottom-left: X-coordinate over time (raw vs. filtered).
Bottom-right: X-axis spectrum; tremor typically shows a bump ~6–12 Hz.
Window title shows estimated sampling rate fs≈… Hz and the active filter.
 
