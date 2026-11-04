from models_bench import BenchBrainomni, BenchCBraMod, BenchLabram, BenchBENDR, BenchBIOT, BenchEEGPT, BenchREVE, BenchLUNA
from preprocessing import DatasetManager
import mne
from models_bench import UMAPReport

EEGDASH_DS_NAMES = [
    "ds006576", # (EEGDash) The role of REM sleep in neural differentiation of memories in the hippocampus -> montage="standard_1020"
#    "ds007609", # (EEGDash) Resting-State EEG and Trait Anxiety -> montage="EGI_256"
    "ds007537", # (OpenNeuro - EEGDash) A multimodal dataset of EEG, eye-tracking, and physiological signals during smartphone interaction -> 
]

EEGDASH_DS_MONTAGES = [
    "colin27_1020",
#    "EGI_256",
    "colin27_1020",
]

MOABB_DS_NAMES = [
    "BNCI2014_001"
]

MOABB_DS_MONTAGES = [
    "colin27_1020"
]

# Already preprocessed datasets (skip_cleaning=True): only channels, montage, resampling and windowing
EEGDASH_PREPROCESSED_DS_NAMES = [
    "ds006923", # (OpenNeuro - EEGDash) Resting state of juvenile offenders and non offenders. 4 epochs of 50s (eyes Closed-Open-Closed-Open), 128 Hz, Biosemi 128
]

EEGDASH_PREPROCESSED_DS_MONTAGES = [
    "keep", # Set by ds006923_ch_treatment (biosemi128)
]

EEGDASH_PREPROCESSED_DS_FILES = [
    "acq-epochs", # Epoched file of each subject (the other one, desc-preprocessed, is continuous and band-passed 1-40 Hz)
]

def ds006576_ch_treatment(raw:mne.io.Raw):
    #raw.drop_channels(['A1','A2']) # -> Reference channels??
    raw.rename_channels({ch:ch.split('_')[-1] for ch in raw.ch_names})
    return raw

# ds006923 (Biosemi 128 ABCD): 10-20 position of each channel, from its BIDS channel name (e.g. "A1 (Cz)", "A5 (approx: P1)")
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
DS006923_USE_APPROX = False # True: the "approx" channels also get the 10-20 name (more channels for LaBraM, EEGPT and BENDR)

def ds006923_ch_treatment(raw:mne.io.Raw):
    """
    Biosemi names (A1..D32) clash with 10-20 names of other positions (Biosemi C3 is frontal, A1/A2 are not the earlobes), so the models
    that pick channels by name (LaBraM, EEGPT, BENDR) would read them as the 10-20 ones. Sets the biosemi128 positions, gives the 10-20 name
    to the channels that have one and prefixes the others with "BS-" (no model knows them). Use montage="keep".
    """
    raw.set_montage("biosemi128")
    mapping = {**DS006923_EXACT_1020, **(DS006923_APPROX_1020 if DS006923_USE_APPROX else {})}
    raw.rename_channels({ch: mapping.get(ch, f"BS-{ch}") for ch in raw.ch_names})
    return raw

def preprocess():
    # for DS_NAME, DS_MONTAGE in zip(MOABB_DS_NAMES, MOABB_DS_MONTAGES):
    #     dsManager = DatasetManager()
    #     dsManager.load_moabb_dataset(DS_NAME.upper(), subject_limit=0)
    #     #dsManager.inspect_dataset()
    #     dsManager.preprocess_and_window(
    #         output_dir = DS_NAME.upper(), 
    #         channels_treatment = None,
    #         l_freq = 0.1, # minimal floor l_freq of models.
    #         h_freq = 75, # Minimal ceil h_freq of models.
    #         down_freq = 256, # The models needs 200Hz-256Hz. So, i'm setting the maximum here.
    #         notch_freqs = [50, 60], # Remove both problematic frequencies
    #         bad_ch_tolerance = 0.2,
    #         ransac = True,
    #         ica = True,
    #         window_size=512,
    #         n_jobs=4,
    #         ref_channels=[],
    #         overwrite=False,
    #         montage=DS_MONTAGE, # None is Auto-detect
    #         treat_artifacts=True,
    #         value_multiplier=1.0 # Necessary to convert the dataset to V
    #     )

    # for DS_NAME, DS_MONTAGE in zip(EEGDASH_DS_NAMES, EEGDASH_DS_MONTAGES):
    #     dsManager = DatasetManager()
    #     dsManager.load_eegdash_dataset(DS_NAME.lower(), subject_limit=0) 
    #     dsManager.inspect_dataset()
    #     dsManager.preprocess_and_window(
    #         output_dir = DS_NAME.upper(), 
    #         channels_treatment = ds006576_ch_treatment if DS_NAME == "ds006576" else None,
    #         l_freq = 0.1, # minimal floor l_freq of models.
    #         h_freq = 75, # Minimal ceil h_freq of models.
    #         down_freq = 256, # The models needs 200Hz-256Hz. So, i'm setting the maximum here.
    #         notch_freqs = [50, 60], # Remove both problematic frequencies
    #         bad_ch_tolerance = 0.2,
    #         ransac = True,
    #         ica = True,
    #         window_size=512,
    #         n_jobs=4,
    #         ref_channels=[],
    #         overwrite=False,
    #         montage=DS_MONTAGE,
    #         treat_artifacts=True,
    #         value_multiplier=1.0 # Necessary to convert the dataset to V
    #     )

    for DS_NAME, DS_MONTAGE, DS_FILES in zip(EEGDASH_PREPROCESSED_DS_NAMES, EEGDASH_PREPROCESSED_DS_MONTAGES, EEGDASH_PREPROCESSED_DS_FILES):
        dsManager = DatasetManager()
        dsManager.load_eegdash_dataset(DS_NAME.lower(), subject_limit=0, file_filter=DS_FILES)
        dsManager.preprocess_and_window(
            output_dir = DS_NAME.upper(),
            channels_treatment = ds006923_ch_treatment if DS_NAME == "ds006923" else None,
            skip_cleaning = True, # Already preprocessed: no bad channels detection, ICA, notch or band-pass
            down_freq = 256, # Same as the other datasets (ds006923 is 128 Hz: upsampled)
            window_size=512, # Epochs are windowed inside each epoch (50s at 256 Hz = 25 windows of 512)
            n_jobs=4,
            ref_channels=[], # Already average reference
            overwrite=False,
            montage=DS_MONTAGE,
            value_multiplier=1.0 # Already in V
        )

def bench(windows_mode="windows-detailed", overwrite=False):
    """
    windows_mode: "windows" (basic plots: labels + event) or "windows-detailed" (+ recordings_events/ and subjects_events/).
    overwrite: False skips the plots that already exist.
    """
    # One point by window (no aggregation), then every aggregation (avg/median/std) by recording and by subject
    variations = [(windows_mode, "recording")] + [(mode, group_by) for group_by in ["recording", "subject"] for mode in ["avg", "median", "std"]]

    #for DS_NAME in MOABB_DS_NAMES + EEGDASH_DS_NAMES + EEGDASH_PREPROCESSED_DS_NAMES:
    for DS_NAME in EEGDASH_PREPROCESSED_DS_NAMES:
        base_windows_path = f"./PreProcessed-Datasets/{DS_NAME.upper()}/PreProcessed-EEG"
        benches = []
        for bench_class in [
            BenchBrainomni,
            BenchLabram,
            BenchBENDR,
            BenchBIOT,
            BenchEEGPT,
            BenchREVE,
            BenchLUNA,
            BenchCBraMod
        ]:
            # A model that does not fit the dataset is skipped (errors saved in Model-Benchmarks/<model>/benchmark-info.json)
            try:
                benches.append(bench_class(base_windows_path, DS_NAME.upper()))
            except Exception as e:
                print(f"[{bench_class.MODEL_NAME}] [ERROR] Skipping {DS_NAME.upper()}: {type(e).__name__}: {e}")
                bench_class.update_benchmark_info(DS_NAME.upper(), errors=[f"Model skipped: {type(e).__name__}: {e}"])

        for mode, group_by in variations:
            for model_bench in benches:
                try:
                    model_bench.use_latent_space(mode, group_by=group_by)
                    model_bench.umap(overwrite=overwrite)
                except Exception as e:
                    print(f"[{model_bench.MODEL_NAME}] [ERROR] {mode} by {group_by} failed on {DS_NAME.upper()}: {type(e).__name__}: {e}")
                    model_bench.update_benchmark_info(model_bench.DS_NAME, errors=[f"{mode} by {group_by} failed: {type(e).__name__}: {e}"])

if __name__ == "__main__":
    #preprocess()
    #bench(windows_mode="windows-detailed", overwrite=False)
    #bench(windows_mode="windows", overwrite=False)

    report = UMAPReport(base_dir="Model-Benchmarks", models=["bendr", "biot", "brainomni_base", "cbramod", "eegpt", "labram", "luna", "reve"])
    report.build("Reports/UMAP-Full-Report.pdf", title="UMAP Full Report")
