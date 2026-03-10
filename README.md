# Chakra Signs

Real-time hand gesture recognition with custom gesture training, combination detection, and step-by-step animation replay.

## Stack

| Layer | Technology |
|-------|-----------|
| Hand tracking | MediaPipe |
| Computer vision | OpenCV |
| ML (static gestures) | scikit-learn |
| ML (sequences) | PyTorch |
| API | FastAPI |
| Frontend | React + TypeScript + Three.js |

## Quick start (Phase 1–2: live hand detection)

### 1. Create a virtual environment

```bash
cd chakra-signs
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell)**: `.\.venv\Scripts\Activate.ps1`
- **macOS/Linux**: `source .venv/bin/activate`

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 3. Run the hand detector

```bash
python backend/scripts/run_detection.py
```

A window opens showing your webcam with hand landmarks drawn in real time.

**Controls**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Print current landmarks to the console |

Optional runtime overrides:

```bash
python backend/scripts/run_detection.py --camera-index 0 --width 1280 --height 720
python backend/scripts/run_detection.py --max-hands 2 --min-detection-confidence 0.75
python backend/scripts/run_detection.py --no-mirror
```

### Configuration

Edit `config.json` in the project root to change detection thresholds or camera settings.
Runtime flags override config values for the current run only.

### Current Phase 1–2 refinements

- Typed config validation with safe defaults (`backend/app/core/config.py`)
- Cleaner detector runner with CLI overrides and logging (`backend/scripts/run_detection.py`)
- Mirror mode configurable from config or CLI (`camera.mirror` / `--no-mirror`)
- Better startup/setup docs for cross-platform environments

## Project structure

```
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes (Phase 9)
│   │   ├── core/         # Config, shared types
│   │   ├── detection/    # Hand landmark detection
│   │   ├── training/     # Model training pipeline (Phase 4)
│   │   ├── inference/    # Live gesture classification (Phase 5)
│   │   ├── sequences/    # Combination engine (Phase 6-7)
│   │   ├── animation/    # Replay / export (Phase 8)
│   │   └── utils/
│   ├── data/             # Datasets, models, metrics
│   ├── scripts/          # Runnable entry points
│   └── tests/
├── frontend/             # React + Three.js viewer (Phase 8+)
├── config.json           # All tuneable thresholds
└── README.md
```

## Phases

- [x] Phase 1 — Project skeleton
- [x] Phase 2 — Live hand landmark detection
- [ ] Phase 3 — Dataset recorder
- [ ] Phase 4 — Static gesture classifier
- [ ] Phase 5 — Real-time inference
- [ ] Phase 6 — Combination engine
- [ ] Phase 7 — Sequence model (PyTorch)
- [ ] Phase 8 — Animation / replay viewer
- [ ] Phase 9 — API and UI polish
- [ ] Phase 10 — Final cleanup
