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

## Quick start

### 1. Create a virtual environment

```bash
cd chakra-signs
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the hand detector (Phase 2)

```bash
python backend/scripts/run_detection.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Print current landmarks to the console |

### 4. Record gesture samples (Phase 3)

```bash
python backend/scripts/record_gestures.py
```

| Key | Action |
|-----|--------|
| `n` | Type a new gesture label (Enter to confirm, Esc to cancel) |
| `r` | Toggle recording on/off |
| `d` | Print dataset summary to the console |
| `q` | Quit (bumps dataset version if samples were saved) |

### 5. Record Naruto hand signs

```bash
python backend/scripts/record_naruto.py
```

| Key | Action |
|-----|--------|
| `1`–`9`, `0`, `-`, `=` | Select one of 12 Naruto hand signs |
| `r` | Toggle recording on/off |
| `d` | Print dataset summary |
| `q` | Quit |

### 6. Train the model (Phase 4)

```bash
python backend/scripts/train_model.py --labels bird boar dog dragon ox tiger snake rat horse monkey hare ram
```

### Configuration

Edit `config.json` in the project root to change detection thresholds, camera settings, or recording parameters.

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
- [x] Phase 3 — Dataset recorder
- [x] Phase 4 — Static gesture classifier
- [ ] Phase 5 — Real-time inference
- [ ] Phase 6 — Combination engine
- [ ] Phase 7 — Sequence model (PyTorch)
- [ ] Phase 8 — Animation / replay viewer
- [ ] Phase 9 — API and UI polish
- [ ] Phase 10 — Final cleanup
