# Sovereign Intelligent Document Processing (IDP) System

This repository hosts a **production-grade, sovereign IDP system** designed for high-security government deployments (GCP Compute Engine). It processes **APAR** and **Disciplinary** records using strictly local AI models, ensuring **zero data egress** and full compliance with data sovereignty norms.

---

## 🛡️ Core Principles (Non-Negotiable)

1.  **Sovereignty**: No external APIs (OpenAI, Gemini, Vertex AI). All inference is local.
2.  **Accuracy > Speed**: Deterministic pipelines prioritized over probabilistic guessing.
3.  **Token Safety**: Hard limits (4,000 chars) prevent context overflow and hallucinations.
4.  **Auditability**: Every step (OCR page, LLM call, Validation) is logged.
5.  **Failure Safety**: The pipeline never crashes on partial failures; it logs and continues.

---

## 🏗️ Architecture

The system operates in two mutually exclusive modes to ensure stability and memory isolation.

### 1. OCR Mode (Memory-Optimized)
-   **Engine**: [Surya OCR](https://github.com/VikParuchuri/surya) (Deep Learning, Indic-capable).
-   **Workflow**:
    1.  Convert PDF → Images (High-res rendering).
    2.  Process Page-by-Page.
    3.  **Explicit Memory Cleanup**: Garbage collection triggers after every page.
    4.  **Output**: `ocr/page_X.txt` (raw text) and `full_text.txt` (tagged with source file/section).

### 2. NLP Mode (Token-Safe & Structured)
-   **Engine**: **Phi-3 Mini** (3.8B parameters, local execution via Hugging Face).
-   **Workflow**:
    1.  Load `full_text.txt`.
    2.  **Classify Document**: APAR vs. Disciplinary.
    3.  **Execute Specialized Pipeline** (see below).
    4.  **Generate Output**: JSON, CSV, DOCX.

---

## 🧠 Processing Pipelines

### A. APAR Pipeline (Multi-Pass Strategy)
To ensure **100% token safety** and prevent hallucinations, we use a **4-Pass Hierarchical Strategy**:

| Pass | Name | Description | Safety Mechanism |
| :--- | :--- | :--- | :--- |
| **1** | **Deterministic Segmentation** | Splits text by year using Regex patterns (e.g., "APAR FOR THE YEAR"). | **No LLM**. Deterministic boundaries. |
| **2** | **Year-wise Extraction** | Extracts fields (Grading, Pen Picture) for ONE year at a time. | Input strict limit: **< 4,000 chars**. |
| **3** | **Global Metadata** | Extracts Officer Name/DOB from the first 3 pages. | Input strict limit: **< 4,000 chars**. |
| **4** | **Merge & Validate** | Combines all data. Checks for missing fields. | **Confidence Flags** added if data is suspicious. |

**Validation Layer**:
-   Flags `LOW_CONFIDENCE` if Officer Name is missing.
-   Flags `MISSING_GRADING` if a year record lacks grading.
-   **Aborts** if no APAR years are detected.

### B. Disciplinary Pipeline (Context-Aware)
-   **Input**: Pre-categorized folders (`brief_background`, `po_brief`, `co_brief`, `io_report`).
-   **Logic**: The system parses `full_text.txt` for section tags (e.g., `=== SECTION: brief_background ===`) and sends ONLY relevant text to the LLM.
-   **Output**: 4 Separate DOCX reports in formal administrative language.

---

## 📂 Project Structure

```
gcp_idp_project/
├── app.py                  # Core Logic (OCR + NLP Pipelines)
├── requirements.txt        # Dependencies (Torch, Transformers, Surya)
├── ingest/                 # Input Data Root
│   ├── apar/               # APAR PDFs
│   └── disciplinary/       # Case Folders
│       ├── case_001/
│       │   ├── brief_background/
│       │   ├── po_brief/
│       │   └── ...
├── output/                 # Structured Results
│   └── case_001/
│       ├── ocr/            # Raw Page Text
│       ├── apar_data.json  # Final JSON
│       ├── apar_result.csv # Flat CSV
│       └── *.docx          # Word Reports
├── logs/                   # Audit Logs (app.log)
├── models/                 # Local Model Cache
└── scripts/                # Execution Wrappers
    ├── setup_vm.sh         # One-click Install
    ├── run_ocr.sh          # OCR Trigger
    └── run_nlp.sh          # NLP Trigger
```

---

## 🚀 Usage

### 1. Setup
```bash
cd scripts
./setup_vm.sh
```

### 2. Run OCR (First Pass)
```bash
# Process an APAR folder or Disciplinary Case
./run_ocr.sh ../ingest/apar
```
*Generates `output/<case_id>/full_text.txt`.*

### 3. Run NLP (Second Pass)
```bash
# Process the OCR output
./run_nlp.sh ../output
```
*Generates final JSON/CSV/DOCX reports.*

---

## 📊 Output Formats

1.  **APAR JSON** (`apar_data.json`):
    ```json
    {
      "officer_name": "John Doe",
      "dob": "01/01/1980",
      "confidence_flags": [],
      "apar_records": [
        { "year": "2020-21", "grading": "8.5", "pen_picture": "..." }
      ]
    }
    ```
2.  **APAR CSV**: Flat table for analytics.
3.  **APAR DOCX**: Official table format.
4.  **Disciplinary DOCX**: `brief_background.docx`, `io_report.docx`, etc.

---

## ⚠️ Compliance & Audit Notes
-   **Data Storage**: All data resides on the VM disk. No cloud buckets used by default.
-   **Logs**: Check `logs/app.log` for a detailed trace of every LLM call, token usage, and validation error.
-   **Models**: Ensure `surya-ocr` and `phi-3-mini` are downloaded during setup. Internet access can be disabled after initial download.
