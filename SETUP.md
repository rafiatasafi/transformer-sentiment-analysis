# SentimentScope setup

Use Python 3.11 or 3.12 for a conservative environment choice. The starter notebook
is `starter/SentimentScope_starter.ipynb`; an unused Copy1 notebook, if present locally, is excluded from the submission.

## VS Code on macOS

Open the repository folder. Install the Microsoft Python and Jupyter extensions.
From the VS Code terminal at the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sentimentscope --display-name "Python (SentimentScope)"
```

If Python 3.12 is unavailable, install it first or use an available supported Python.
Select the SentimentScope kernel (or the `.venv` interpreter) in the notebook.
Run cells in order. Extraction is automatic. The tokenizer requires internet on
first use; the IMDB archive is already supplied. No Hugging Face account is needed
for the public tokenizer. On macOS, the notebook selects MPS when available; CPU
is the fallback. CUDA takes priority in the course workspace.

## Course Nvidia GPU workspace

Upload the completed notebook, `aclImdb_v1.tar.gz`, and `requirements.txt`.
Keep the archive beside the notebook. Inspect the provided environment before
installing packages; preserve its GPU-compatible PyTorch installation. Install
missing dependencies as necessary. Restart the kernel after changing packages.
Verify the notebook prints `Using device: cuda`, then run cells from top to bottom.
If memory is insufficient, lower BATCH_SIZE from 32 to 16 or 8.

For local preparation, run through the model shape assertions. The accuracy,
training, and test sections run full data passes and are best executed on the GPU.
The untrained validation baseline is optional. The included saved notebook outputs document the completed local training run.

## Outputs and submission

The notebook creates `starter/artifacts/` (or `artifacts/` when launched beside the
archive): graphs, tokenizer, best model checkpoint, history, test metrics, and a
results report. Keep these files together for batch inference. The final notebook
section defines `SentimentPredictor`, which reloads the checkpoint and tokenizer.
In a fresh kernel, first execute the imports, clean_review function, and model class
definitions as instructed there.

Choose hyperparameters using validation data. Evaluate the official test set only
after finalizing the model. The >75% rubric target must be demonstrated by measured
results; the implementation does not guarantee it. Save the notebook with real
outputs after GPU execution and include the report and labeled graphs in submission.
The checkpoint contains weights/configuration rather than optimizer state; it supports
inference, not exact training resumption. Dependency ranges are compatibility bounds,
not a tested lockfile; record `python -m pip freeze` in the training environment.
