## ASL Landmark Recognition (MediaPipe + MLP)

This project recognizes ASL alphabet gestures using **MediaPipe Hands** landmarks and a small **MLP** classifier.

### What’s in this repo
- Python scripts to **extract** landmarks from an image dataset, **train** an MLP model, and run **real-time** webcam inference.

### What’s not in this repo
To keep the GitHub repo lightweight, large/generated files are intentionally ignored via `.gitignore`, including:
- The image dataset folder (`asl_alphabet_train/`) and any `.zip` archives
- Generated landmark CSVs (e.g. `asl_landmark_data.csv`)
- Trained model/labels (e.g. `.h5`, `.npy`)

---

## Setup

1) Create/activate a virtual environment (recommended)

2) Install dependencies:

`pip install -r requirements.txt`

---

## Get the dataset

Place the **ASL Alphabet** dataset folder here:

`asl_alphabet_train/`

It should contain subfolders like `A/`, `B/`, ..., `space/`, `del/`, `nothing/`.

---

## Extract landmarks (dataset → CSV)

This reads images from `asl_alphabet_train/` and produces `asl_landmark_data.csv`.

`python main.py extract`

---

## Train model (CSV → model)

This reads `asl_landmark_data.csv` and produces:
- `asl_landmark_mlp_model.h5`
- `asl_landmark_labels.npy`

`python main.py train`

---

## Run real-time recognition (webcam)

Requires that you have already trained/saved the model artifacts.

`python main.py realtime`

Controls (in the webcam window):
- `c` to confirm the currently stable letter
- `q` to quit

---

## Optional: collect your own samples

Captures hand landmarks from your webcam and saves `landmark_data.csv`:

`python main.py collect`

Press `s` to save a sample, `q` to quit.

