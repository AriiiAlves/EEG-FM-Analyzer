# Coisas a se fazer

# Treating query

Básicos
- Pesquisar datasets de treino (usados por artigos)

- Ver como pegar datasets já processados (juvenille offenders) com o captador de dataset

# Notas

- Harmonics don't matter here. 100 Hz and 120 Hz are above your h_freq = 75, so the band-pass already removes them.
- The per-recording plots are one per recording, not per subject. BNCI2014_001 has 9 subjects × 2 sessions × 6 runs = 108 recordings. If you'd rather have one plot per subject with their runs combined, it's a small change: group by description["subject"] instead of the recording index.

- MOABB is a curated benchmark of about 70+ BCI datasets: motor imagery, P300/ERP, SSVEP, c-VEP and resting state. Each one is wrapped by hand, with paradigm metadata and event mapping.
- EEGDash indexes hundreds of BIDS-formatted datasets from OpenNeuro/NEMAR (ds00xxxx IDs). These are mostly cognitive and clinical research recordings, not BCI benchmarks.

They overlap a little, since a few BCI datasets also appear on OpenNeuro, but mostly they are complementary.

Other handlers in braindecode with different data

- TUH / TUHAbnormal: the Temple University Hospital clinical EEG corpus, used for abnormal-EEG detection. It’s huge, needs a manual download with a data agreement, and you pass a local path.
- NMT: a clinical normal/abnormal corpus from Pakistan. It also uses a local path.
- SleepPhysionet (Sleep-EDF) and SleepPhysionetChallenge2018: sleep staging datasets that download automatically.
- HGD: the High-Gamma motor dataset (Schirrmeister). It’s also in MOABB as Schirrmeister2017.
- BCICompetitionIVDataset4: ECoG finger flexion.
- BIDSDataset: loads any local BIDS folder, such as OpenNeuro data downloaded yourself.

# Available Montages

Defining the montage is very important. If the dataset does not contain exaclty the same channel names of selected montage, the preprocessing is not going to work.

The available montages is defined by `mne` (it is possible to see with `mne.channels.get_builtin_montages(descriptions=True)`)

Script used:

```py
montages = mne.channels.get_builtin_montages(descriptions=True)

for mnt, description in montages:
    channels = mne.channels.make_standard_montage(mnt).ch_names
    print(f"- `{mnt}` - {description}\n\t- ", end="")
    for ch in channels:
        print(ch, end=", ")
    print("")
```

# Good material

- https://neuroanalyzer.org/tutorials/proc_ref.html

# Outros

Startup takes several minutes. The run started at 17:56, but params.json was only written at 18:02.

The .venv is on the network drive, so every import reads thousands of small files over the network. A plain import check I ran earlier didn't finish within 120 s.
main.py also imports models_bench (torch + BrainOmni) at the top, even though only test_preprocessing() runs.
Each worker re-imports preprocessing.py's libraries (mne, braindecode, eegdash, moabb) when it starts.
Choosing the montage also reads the first recording before any worker starts.