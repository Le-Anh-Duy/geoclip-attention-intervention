# Kaggle offline detector probing

This directory contains four private Kaggle notebooks:

1. `geoclip-detector-offline-assets` downloads and exports all model assets on Kaggle with Internet enabled.
2. `geoclip-detector-causal-probing` mounts those outputs, Img2GPS3K, and the ARC-AGI-3 competition source, then runs fully offline on `NvidiaRtxPro6000`.
3. `wedetect-img2gps3k-cache` runs WeDetect once over all 2,997 Img2GPS3K test images and exports reusable low-threshold proposals.
4. `wedetect-geoclip-signal-search` consumes that cache and searches all GeoCLIP layers and heads, structured layer groups, proposal doses, and CLS query scope with a discovery/locked-holdout protocol.

The experiment tests a counterfactual detector-agreement hypothesis rather than only unioning every proposal. Grounding DINO and prompt-free WeDetect produce independent 16x16 patch masks. The notebook compares their intersection, union, disagreement, proposal-count dose, boost/suppress polarity, and early/middle/late layer interventions. It reports label-based accuracy/error and label-free prediction displacement.

Push in dependency order:

```powershell
python kaggle/build_notebooks.py
kaggle kernels push -p kaggle/geoclip-detector-offline-assets
kaggle kernels status giahuytran1/geoclip-detector-offline-assets
kaggle kernels push -p kaggle/geoclip-detector-causal-probing --accelerator NvidiaRtxPro6000
```

The full-test WeDetect-only results and interpretation are recorded in
`results/wedetect-geoclip-signal-search/README.md`.
