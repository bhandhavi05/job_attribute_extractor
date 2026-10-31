# Job Description Attribute Extractor

Extract structured information from a job description with a fine-tuned Transformer named-entity recognition (NER) model. The project includes a Streamlit interface and a FastAPI backend.

## Features

- Extracts company, role, location, education, experience, salary, work mode, and skills.
- Returns each detected value with a confidence score.
- Shows the result as readable fields and JSON in the web interface.
- Includes an API health check and interactive API documentation.

## Project structure

```text
backend/                 FastAPI application and request/response schemas
frontend/                Streamlit user interface
models/job_ner_model/    Trained model and tokenizer
src/                     Extraction, training, and data preparation code
data/                    Raw, processed, and annotated datasets
notebooks/               Dataset analysis notebook
requirements.txt         Python dependencies
```

## Requirements

- Python 3.10 or newer
- The model files under `models/job_ner_model/` (included in this repository)

## Run the project step by step

Run commands from the project root: the `job-description-attribute-extractor` folder that contains `requirements.txt`, `backend/`, and `frontend/`.

### 1. Check Python

Install Python 3.10 or newer if it is not already installed. Confirm Python and pip are available:

```bash
python --version
python -m pip --version
```

### 2. Open a terminal in the project root

In VS Code, open the project root folder, then select **Terminal → New Terminal**. Confirm the terminal is in the directory containing `requirements.txt`:

```bash
dir        # Windows PowerShell
ls         # macOS / Linux
```

If needed, change to the project directory (replace the example path with your actual location):

```powershell
cd "C:\path\to\job-description-attribute-extractor"
```

### 3. Create and activate a virtual environment

Create the environment once:

```bash
python -m venv .venv
```

Activate it in **each new terminal** where you run the app:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

When activated, the environment name (usually `.venv`) appears in the terminal prompt. If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.

### 4. Install the dependencies

With the environment active, install the packages (this is needed the first time, or after dependencies change):

```bash
python -m pip install -r requirements.txt
```

### 5. Start the FastAPI backend

In the first terminal, from the project root with the virtual environment active, run:

```bash
python -m uvicorn backend.main:app --reload
```

Wait for the server to start. It loads the model from `models/job_ner_model/` and serves the API at `http://127.0.0.1:8000`. Keep this terminal open while using the app. You can check the API at `http://127.0.0.1:8000/health` or open its interactive docs at `http://127.0.0.1:8000/docs`.

### 6. Start the Streamlit frontend

Open a **second terminal** in the same project root. Activate the virtual environment in that terminal, then run:

```bash
streamlit run frontend/app.py
```

Keep this terminal open too. Streamlit prints a local URL, usually `http://localhost:8501`; open it in your browser. The frontend connects to the backend at `http://127.0.0.1:8000`.

### 7. Extract attributes

Paste a job description into the text box and select **Extract Attributes**. Review the detected fields, confidence scores, and JSON output. Use **Download JSON** to save the result.

### 8. Stop the application

In each terminal running a server, press `Ctrl+C` to stop it. To use the project again later, open two terminals, navigate to the project root, activate `.venv` in both, then repeat steps 5 and 6. You do not need to recreate the virtual environment or reinstall dependencies unless they have changed.

## API

The FastAPI server runs on `http://127.0.0.1:8000` by default.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | API status message |
| `GET` | `/health` | Health check |
| `POST` | `/extract` | Extract attributes from a job description |

Example request:

```json
{
  "text": "Acme is hiring a Software Engineer in Hyderabad. Requires a bachelor's degree and 2 years of Python experience."
}
```

The extraction response contains `success` and an `attributes` object. Attribute values are lists of objects containing `value` and `confidence`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Model and training

The backend loads the tokenizer and token classification model from `models/job_ner_model/` when it starts. Training and data preparation utilities are in `src/`; the notebooks and datasets are provided for analysis and experimentation.
