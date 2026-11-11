# 🛡️ Sentinel — Multi-Signal Email Spam & Phishing Detection System

An end-to-end, layered machine learning system for detecting spam and phishing emails in real time — built to go beyond traditional single-model text classifiers by fusing semantic understanding, anomaly detection, and structural/behavioral signals into one explainable pipeline.

---

## 🚩 Motivation

Most spam detection projects rely on a single technique — TF-IDF + Naive Bayes, or a standalone fine-tuned transformer — trained and evaluated on the same static dataset. This project instead asks a harder, more realistic question:

> **Does a model trained on classic phishing/spam data actually generalize to modern, sophisticated attacks — including LLM-generated phishing?**

The answer, measured honestly against held-out external benchmarks, is: **not without deliberate design** — and this repo documents both the limitation and a working mitigation.

---

## 🏗️ Architecture

The system combines four complementary signal sources into a single meta-classifier:

| Layer | Technique | Purpose |
|---|---|---|
| **Semantic** | `all-MiniLM-L6-v2` sentence embeddings | Captures meaning/intent, catching paraphrased or reworded spam that keyword filters miss |
| **Anomaly Detection** | Autoencoder trained only on legitimate mail | Flags stylistic deviations from "normal" — designed to catch novel/zero-day patterns without needing labeled examples of every new attack |
| **Structural/Behavioral** | Engineered features: SPF/DKIM/DMARC checks, sender/reply-to mismatch, URL analysis (IP-literal links, suspicious TLDs, link entropy), urgency-language detection | Captures signals text embeddings alone can't see |
| **Meta-Classifier** | LightGBM (gradient-boosted trees) | Fuses all signals into a final Spam / Suspicious / Not Spam verdict |
| **Explainability** | SHAP (TreeExplainer) | Produces human-readable reasons behind every prediction, rather than a black-box score |

```
Email → [Feature Extraction] → [MiniLM Embedding] → [Anomaly Score]
                                        ↓
                          [LightGBM Meta-Classifier]
                                        ↓
                    Spam / Suspicious / Not Spam + SHAP Explanation
```

---

## 🧠 Anomaly Detection: The Autoencoder Layer

Most spam filters are purely **supervised** — they can only recognize attack patterns they were explicitly trained on. This project adds an **unsupervised anomaly detection layer** specifically to address that blind spot: catching phishing/spam that doesn't resemble anything in the labeled training data at all.

**How it works:**
- A feedforward autoencoder (encoder-decoder neural network) is trained **only on legitimate ("ham") emails** — it never sees a single spam or phishing example during its own training.
- It learns to compress and reconstruct the MiniLM embedding of a legitimate email with low error, since it's only ever practiced on "normal" patterns.
- When a new email comes in, the autoencoder tries to reconstruct its embedding. A **legitimate email reconstructs with low error** (it looks like what the model learned as "normal"). A **spam/phishing email — especially one stylistically unlike anything seen before — reconstructs with high error**, since the model has no learned pattern to compress it efficiently.
- This **reconstruction error becomes an "anomaly score"**, fed into the final LightGBM classifier alongside the other engineered features.

**Why this matters:** this is the component specifically designed to catch **zero-day attacks** — phishing campaigns using entirely new templates, wording, or structure that no supervised classifier has ever been trained on, since it doesn't need a labeled example of the attack to flag it as unusual.

**Retraining note:** the autoencoder was retrained on the ham portion of the full unified training set (SpamAssassin + CyberCop + Enron, ~22,800 legitimate emails) to give it a broader, more representative baseline of "normal" mail before being used in the final pipeline.

---

## 📊 Datasets & Evaluation Methodology

**Training data:**
- SpamAssassin Public Corpus
- CyberCop phishing dataset
- Enron-Spam (labeled)

**External evaluation (held out, never trained on):**
- **Nazario Phishing Corpus** — classic, real-world phishing targeting corporate correspondence
- **E-PhishLLM** (Pajola et al., 2025) — modern, LLM-generated phishing, purpose-built to expose the generalization gap in models trained on older data

### Baseline Model Results
Trained only on SpamAssassin + CyberCop + Enron:

| Test Set | Accuracy | ROC-AUC | Phishing Recall |
|---|---|---|---|
| Nazario | 82% | 0.94 | 68% |
| E-PhishLLM | 60% | 0.72 | 31% |

This confirms a real, documented finding: models trained on older-style phishing generalize noticeably worse to modern, professionally-worded, LLM-generated phishing.

### Enhanced Model Results
Retrained with additional exposure to 70% of E-PhishLLM (evaluated strictly on the untouched 30% holdout — methodologically separate from the baseline result):

| Test Set | Accuracy | ROC-AUC |
|---|---|---|
| Nazario | 92% | 0.97 |
| E-PhishLLM (30% holdout) | 91% | 0.97 |

**Why report both:** the baseline demonstrates the real, honest limitation; the enhanced model demonstrates a legitimate, measured mitigation — evaluated on genuinely unseen data, not the same set used for training.

---

## 💻 Application

Two deployment modes, built with Flask:

1. **Live Inbox Monitor** — connects to Gmail via IMAP, automatically checks for new mail every 45 seconds, displays classification results with direct links back to Gmail
2. **Manual & Batch Check** — for users who don't want to connect a full mailbox: paste a subject/body directly, or upload up to 50 `.eml`/`.json` files for batch analysis, with full scan history (expandable detail view, individual delete, clear-all)

---

## 🧰 Tech Stack

- **ML:** sentence-transformers (MiniLM), PyTorch (autoencoder), LightGBM, SHAP, scikit-learn
- **Backend:** Flask, imap-tools (Gmail IMAP integration)
- **Data processing:** pandas, numpy

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up Gmail access
# Create a .env file in the project root:
GMAIL_ADDRESS=youremail@gmail.com
GMAIL_APP_PASSWORD=your_generated_app_password

# Run the app
python -u -m src.webapp.app
```

Visit `http://localhost:5000` for the live inbox monitor, `http://localhost:5000/manual` for manual/batch checking.

**Note:** trained model files, raw datasets, and credentials are excluded from this repo (see `.gitignore`) — training scripts are included under `src/models/` and `src/utils/` for full reproducibility.

---

## 📁 Project Structure

```
src/
├── ingestion/        # Gmail IMAP fetching
├── features/         # Header, URL, and text-signal feature extraction
├── models/           # Embedding, autoencoder, classifier training & evaluation
├── pipeline/          # End-to-end inference pipeline
├── utils/             # Dataset loading, unification, preprocessing
└── webapp/            # Flask app + templates
```

---

## 🔬 Key Findings & Honest Limitations

- Feature fusion (semantic + structural + anomaly-based) meaningfully outperforms single-model, text-only approaches on classic phishing/spam benchmarks.
- **A real generalization gap exists** between classic and modern LLM-generated phishing — consistent with recent literature (Pajola et al., 2025) — and is not fully solved by this project alone; this is an active, open research problem industry-wide.
- Targeted exposure to modern phishing samples during training substantially narrows this gap, evaluated with proper train/test separation to avoid inflated claims.
- The system is designed for individual/small-scale deployment (on-device inference where feasible); it does not claim to match the detection scale of commercial systems like Gmail, which benefit from network-effect learning across billions of users.

---