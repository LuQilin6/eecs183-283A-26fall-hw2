# Assignment 2: Tokenization, Word Embeddings, and Vector Algebra

Implement tokenization, Skip-Gram embeddings, and vector operations in
`hw2.ipynb`. Answer the starred (★) questions in `report.pdf`.
The assignment is worth 100 points, plus 10 bonus points.

## Google Colab setup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Louis0324/eecs183-283A-26fall-hw2/blob/main/hw2.ipynb)

1. Open the notebook above and use **File → Save a copy in Drive** to keep your edits.
2. Under **Runtime → Change runtime type**, choose **2026.07** for the runtime
   version and a GPU if available. CPU works too, but training takes longer.
   This [Colab runtime](https://research.google.com/colaboratory/runtime-version-faq.html)
   provides Python 3.12, which supports the homework's pinned libraries.
3. Run the first code cell. It downloads the homework files and installs the
   required libraries and language models; this can take several minutes.
4. When installation finishes, choose **Runtime → Restart session**, then run
   that cell again. Allow Google Drive access when prompted. Once it prints
   **Setup ready**, complete the TODOs and run the remaining cells in order.

Your outputs are saved in `MyDrive/cs183-hw2/results/`. After a disconnect,
reconnect and rerun setup; completed files remain on Drive, but Python variables
and unfinished training do not. Keep saving your notebook edits to Drive.

`requirements.txt` lists Python libraries; `requirements-models.txt` lists the
English, French, and Japanese spaCy models used in Parts 3–4. The setup cell
handles both automatically, while keeping Colab's supplied PyTorch and Jupyter.

## Submission

Run the notebook's **Prepare your submission** cell, even if your work is
incomplete. Upload your report when prompted, or set `include_report = False`
in that cell to submit without one. Canceling the upload also skips the report. The helper captures
your edited notebook and downloads `submission.zip`; it also saves a copy in
`MyDrive/cs183-hw2/`. Upload this ZIP to Gradescope.

Partial submissions are accepted. The helper includes all available visible files
in `results/` and warns about missing files, invalid outputs, or syntax errors.
These warnings do not block packaging; the notebook itself must be readable.
Missing or incorrect work loses the corresponding credit, while independent
completed parts are still graded. A syntax error causes its code cell to be
skipped, so fix syntax errors even in unfinished sections when possible.
A complete submission has the following layout:

```text
submission.zip
├── hw2.ipynb
├── report.pdf
└── results/
    ├── skipgram_scores.npy
    ├── solve_analogy.json
    ├── W_en_fr.npy
    ├── W_en_ja.npy
    └── ... other generated outputs
```

**Local or manual packaging:** save your notebook as `hw2.ipynb`, put
any available `report.pdf` beside it, and run the following from the homework directory:

```bash
python make_submission.py
```

Use `--notebook`, `--report`, `--results`, or `--out` for other paths, or
`--no-report` to exclude an existing report. If automatic
notebook capture fails in Colab, download your edited notebook using
**File → Download → Download .ipynb**. Download your `results/` folder from Drive,
extract it locally, and package it with your edited notebook and final report:

```bash
python make_submission.py --notebook /path/to/edited.ipynb --report /path/to/report.pdf --results /path/to/results
```

The starter notebook in Colab's `/content/cs183-hw2` clone does **not** contain
your browser edits; use the submission cell or your downloaded notebook.

## Local setup (optional)

From the homework directory:

```bash
conda create -n hw2 python=3.10 pip -y
conda activate hw2
# Linux CPU installation; omit this line on macOS.
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-models.txt
python -m ipykernel install --user --name hw2 --display-name "Python (hw2)"
jupyter lab
```

Select **Python (hw2)** and open `hw2.ipynb`. Local training defaults to CPU.

## Tips

- Keep function signatures, output filenames, and tokenizer/model versions as
  provided. Use ordinary Python cells, without shell commands or IPython magics.
- Test your implementation on a small corpus before full training. The default
  training run uses 10 epochs and batches of 1024.
- Use the notebook's specified subset for bag-of-words context analysis;
  generating all pairs for the full corpus can exhaust RAM.
- After changing an implementation, rerun the cells that depend on it so saved
  results match your submitted code.
