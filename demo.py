"""
EEG-FM-Analyzer demo
====================

How do pre-trained EEG foundation models "see" a dataset? This script answers it in three steps:

    STEP 1  Preprocess   clean each EEG recording and cut it into 2 s windows        (preprocessing.py)
    STEP 2  Benchmark    encode the windows with each model and plot UMAPs            (models_bench.py)
    STEP 3  Report       gather every UMAP plot into one PDF                          (models_bench.py)

How to use it:
    1. Pick the steps to run in "WHAT TO RUN".
    2. Turn datasets on/off with "enabled" in "DATASETS".
    3. Run from the repository root:   python demo.py

Every step can be stopped and re-run: recordings, embeddings and plots that already exist are skipped.

Outputs:
    PreProcessed-Datasets/<DATASET>/    clean windows, PSD plots, params.json             (step 1)
    Model-Benchmarks/<model>/           embeddings, UMAP plots, benchmark-info.json       (step 2)
    Reports/UMAP-Full-Report.pdf        the PDF report                                    (step 3)
"""
import mne


# =====================================================================================================
# WHAT TO RUN
# =====================================================================================================

RUN_PREPROCESSING = False   # Step 1
RUN_BENCHMARK = False       # Step 2 (needs step 1; a GPU is recommended)
RUN_REPORT = True           # Step 3 (needs step 2; no GPU needed)

OVERWRITE = False           # True: redo recordings / plots that already exist


# =====================================================================================================
# CHANNEL TREATMENTS
# A channel treatment receives each raw recording and returns it with the channels renamed or dropped,
# so that the names match a standard montage (see MONTAGES.md).
# It must be a top-level function: it is sent to the parallel workers.
# =====================================================================================================

def ds006576_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """ "EEG_Fp1" -> "Fp1" """
    raw.rename_channels({ch: ch.split('_')[-1] for ch in raw.ch_names})
    return raw


# ds006923 uses a Biosemi 128 cap (channels A1..D32). Its BIDS files give the 10-20 position of some channels,
# e.g. "A1 (Cz)" (exact) or "A5 (approx: P1)" (approximate).
DS006923_EXACT_1020 = {
    'A1': 'Cz', 'A3': 'CPz', 'A10': 'PO7', 'A15': 'O1', 'A19': 'Pz', 'A21': 'POz', 'A23': 'Oz', 'A25': 'Iz', 'A28': 'O2',
    'B7': 'PO8', 'B11': 'P8', 'B14': 'TP8', 'B20': 'C2', 'B22': 'C4', 'B24': 'C6', 'B26': 'T8', 'B27': 'FT8',
    'C7': 'F8', 'C8': 'AF8', 'C16': 'Fp2', 'C17': 'Fpz', 'C19': 'AFz', 'C21': 'Fz', 'C23': 'FCz', 'C29': 'Fp1', 'C30': 'AF7',
    'D7': 'F7', 'D8': 'FT7', 'D14': 'C1', 'D19': 'C3', 'D21': 'C5', 'D23': 'T7', 'D24': 'TP7', 'D31': 'P7',
}
DS006923_APPROX_1020 = {
    'A5': 'P1', 'A7': 'P3', 'A17': 'PO3', 'A30': 'PO4', 'A32': 'P2', 'B2': 'CP2', 'B4': 'P4', 'B10': 'P10', 'B13': 'P6', 'B16': 'CP6',
    'B18': 'CP4', 'B29': 'FC6', 'B31': 'FC4', 'C4': 'F4', 'C5': 'F6', 'C11': 'FC2', 'C12': 'F2', 'C15': 'AF4', 'C24': 'FC1', 'C25': 'F1',
    'C28': 'AF3', 'D4': 'F3', 'D5': 'F5', 'D10': 'FC5', 'D12': 'FC3', 'D16': 'CP1', 'D26': 'CP5', 'D28': 'CP3', 'D29': 'P5', 'D32': 'P9',
}
DS006923_USE_APPROX = False  # True: the approximate channels also get their 10-20 name (more channels for LaBraM, EEGPT and BENDR)

def ds006923_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Biosemi names clash with 10-20 names at other positions (Biosemi "C3" is frontal), so the models that pick channels
    by name (LaBraM, EEGPT, BENDR) would misread them. This sets the biosemi128 positions, gives the 10-20 name to the
    channels that have one, and prefixes the others with "BS-" (no model knows them). Use it with montage="keep".
    """
    raw.set_montage("biosemi128")
    mapping = {**DS006923_EXACT_1020, **(DS006923_APPROX_1020 if DS006923_USE_APPROX else {})}
    raw.rename_channels({ch: mapping.get(ch, f"BS-{ch}") for ch in raw.ch_names})
    return raw


# =====================================================================================================
# DATASETS
#   name          MOABB class name or EEGDash id (also the output folder, in upper case)
#   source        "moabb" or "eegdash"
#   montage       MNE montage name (see MONTAGES.md), or "keep" if the channel treatment sets the positions
#   channels      channel treatment function, or None to keep the channels as they are
#   preprocessed  True if the dataset is already cleaned: only channels, montage, resampling and windowing run
#   file_filter   (optional) only load the files whose path contains this text
#   enabled       False to skip the dataset
# =====================================================================================================

DATASETS = [
    dict(name="BNCI2014_001", source="moabb", montage="colin27_1020", channels=None,
         preprocessed=False, enabled=False),   # Motor imagery (left hand, right hand, feet, tongue)

    dict(name="ds006576", source="eegdash", montage="colin27_1020", channels=ds006576_channels,
         preprocessed=False, enabled=False),   # REM sleep and memory differentiation

    dict(name="ds007537", source="eegdash", montage="colin27_1020", channels=None,
         preprocessed=False, enabled=False),   # EEG, eye-tracking and physiology during smartphone use

    dict(name="ds007609", source="eegdash", montage="EGI_256", channels=None,
         preprocessed=False, enabled=False),   # Resting state and trait anxiety

    dict(name="ds006923", source="eegdash", montage="keep", channels=ds006923_channels,
         preprocessed=True, file_filter="acq-epochs", enabled=True),
         # Resting state of juvenile offenders: 4 epochs of 50 s (eyes closed/open), 128 Hz, average reference.
         # "acq-epochs" = the epoched file (the other file of each subject is continuous and already band-passed)
]


# =====================================================================================================
# SETTINGS
# =====================================================================================================

# Step 1: every dataset ends at the same rate and window size, so all models can be compared on it.
# models_bench.py then resamples / regroups the windows for each model.
WINDOWING = dict(
    down_freq=256,          # Hz: the highest rate any model needs (BrainOmni, BENDR, EEGPT, LUNA)
    window_size=512,        # samples per window: 512 / 256 Hz = 2 s
    ref_channels=[],        # [] = common average reference
    value_multiplier=1.0,   # converts the data to Volts (1e-6 if stored in µV)
    n_jobs=4,               # recordings processed in parallel (more jobs = more RAM)
)

# Step 1, only for raw datasets (preprocessed=False)
CLEANING = dict(
    l_freq=0.1,             # Hz: lowest high-pass among the models' papers
    h_freq=75,              # Hz: lowest low-pass among the models' papers
    notch_freqs=[50, 60],   # remove both power-line frequencies, the same way for every dataset
    bad_ch_tolerance=0.2,   # drop a recording if more than 20% of its channels are bad
    ransac=True,            # better bad-channel detection (slower)
    ica=True,               # ICA for eye/heart artifacts (False = SSP, faster)
    treat_artifacts=True,
)

# Step 2: "windows" = one UMAP point per window; "windows-detailed" also plots the events of each recording and subject
WINDOWS_MODE = "windows-detailed"

# Step 2: the models to compare. Delete a line to skip a model.
def models():
    from models_bench import BenchBrainomni, BenchLabram, BenchBENDR, BenchBIOT, BenchEEGPT, BenchREVE, BenchLUNA, BenchCBraMod
    return [BenchBrainomni, BenchLabram, BenchBENDR, BenchBIOT, BenchEEGPT, BenchREVE, BenchLUNA, BenchCBraMod]

# Step 3
REPORT_PATH = "Reports/UMAP-Full-Report.pdf"


# =====================================================================================================
# STEP 1: PREPROCESS
# =====================================================================================================

def preprocess(dataset):
    from preprocessing import DatasetManager

    manager = DatasetManager()  # one manager per dataset (it remembers the montage)

    if dataset["source"] == "moabb":
        manager.load_moabb_dataset(dataset["name"].upper())
    else:
        manager.load_eegdash_dataset(dataset["name"].lower(), file_filter=dataset.get("file_filter"))

    cleaning = dict(skip_cleaning=True) if dataset["preprocessed"] else CLEANING
    manager.preprocess_and_window(
        output_dir=dataset["name"].upper(),
        channels_treatment=dataset["channels"],
        montage=dataset["montage"],
        overwrite=OVERWRITE,
        **WINDOWING,
        **cleaning,
    )


# =====================================================================================================
# STEP 2: BENCHMARK
# =====================================================================================================

def benchmark(dataset):
    ds_name = dataset["name"].upper()
    windows_path = f"./PreProcessed-Datasets/{ds_name}/PreProcessed-EEG"

    # 1. Load each model. A model that doesn't fit the dataset (e.g. BIOT needs bipolar channels) is skipped,
    #    and the reason is saved in Model-Benchmarks/<model>/benchmark-info.json.
    benches = []
    for model_class in models():
        try:
            benches.append(model_class(windows_path, ds_name))
        except Exception as e:
            print(f"[{model_class.MODEL_NAME}] [ERROR] Skipping {ds_name}: {type(e).__name__}: {e}")
            model_class.update_benchmark_info(ds_name, errors=[f"Model skipped: {type(e).__name__}: {e}"])

    # 2. The UMAPs to draw: one point per window, then one point per recording / per subject
    #    (average, median or standard deviation of its windows).
    #    The model encodes the windows only once; the other variations reuse the cached embeddings.
    variations = [(WINDOWS_MODE, "recording")]
    for group_by in ["recording", "subject"]:
        for mode in ["avg", "median", "std"]:
            variations.append((mode, group_by))

    for mode, group_by in variations:
        for bench in benches:
            try:
                bench.use_latent_space(mode, group_by=group_by)
                bench.umap(overwrite=OVERWRITE)
            except Exception as e:
                print(f"[{bench.MODEL_NAME}] [ERROR] {mode} by {group_by} failed on {ds_name}: {type(e).__name__}: {e}")
                bench.update_benchmark_info(ds_name, errors=[f"{mode} by {group_by} failed: {type(e).__name__}: {e}"])


# =====================================================================================================
# STEP 3: REPORT
# =====================================================================================================

def report():
    from models_bench import UMAPReport

    model_names = sorted(model_class.MODEL_NAME for model_class in models())
    UMAPReport(base_dir="Model-Benchmarks", models=model_names).build(REPORT_PATH, title="UMAP Full Report")


# =====================================================================================================

if __name__ == "__main__":
    datasets = [dataset for dataset in DATASETS if dataset["enabled"]]

    if RUN_PREPROCESSING:
        for dataset in datasets:
            preprocess(dataset)

    if RUN_BENCHMARK:
        for dataset in datasets:
            benchmark(dataset)

    if RUN_REPORT:
        report()
