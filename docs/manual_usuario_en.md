# User Manual - Colon Cancer Diagnostic Simulator (ColonAI)

Welcome to **ColonAI**, a clinical-grade simulation tool built to assist medical practitioners in the early prediction and diagnosis of colon cancer using Machine Learning and Deep Learning architectures.

## 1. System Requirements and Startup

To start the system, you must boot both the REST API Backend and the Application Frontend.

1. **Start the API Server**:
   Launch your terminal and run:
   ```bash
   uvicorn src.api.main_api:app --reload --port 8000
   ```
   *This will boot the Machine Learning and Convolutional Neural Networks securely.*

2. **Start the Frontend Application**:
   Open a second terminal window and run:
   ```bash
   streamlit run src/frontend/app.py
   ```
   *This will launch the Web Interface in your default browser (usually at http://localhost:8501).*

---

## 2. Using the System

### 2.1 Interface Language

At the top right corner of the application screen, you will find a dropdown menu labeled **Language/Idioma**. You can toggle between `es` (Spanish) and `en` (English) in real-time. This manual uses the English translation interface.

### 2.2 Adding and Looking Up Patients

- **Search Existing Records**: Type a patient's **DNI** (ID Document) or **NUSS** (National Insurance Number) in the "Search Document" field and click **Load Data**. 
- **Add a New Patient**: If the patient is not on file, click **New Patient**. A form will spawn for you to input their identity details (Name, Last Name, City, Weight, Height). Fill them out and hit **Save Patient to Database** at the bottom.

### 2.3 Performing a Triage Risk Calculation (Machine Learning)

To calculate a generic risk score without intrusive analytics:
1. Load the patient data successfully.
2. In the central pane, modify the **Living Habits** section (rate their smoking, alcohol use, physical activity out of 10).
3. Ensure that the Medical Analytics section, specifically `FOBT test` is set to "Unknown" and the `CEA Marker` is left empty or set to -1.0.
4. Click the prominent **Calculate Risk (AI Assistant)** button.
5. The system will yield a Triage Score (Low, Medium, High). A **SHAP Explanation Chart** will appear detailing which habits pulled the diagnosis negatively for clinical understanding.

### 2.4 Performing a Confirmatory Diagnosis

When you receive the physical lab results:
1. Change the FOBT option to `Positive` or `Negative`.
2. Input the numerical value retrieved for the Carcinoembryonic Antigen (e.g. `3.42`) into the `CEA ng/mL` field.
3. Click **Calculate Risk** again. The algorithm will dynamically switch to the Full-Clinical Model providing a high-confidence diagnosis.

### 2.5 Using Image Analysis Tools (Deep Learning)

If an endoscopy, colonoscopy, or biopsy was performed, proceed to the **Image Analysis (Biopsies & Colonoscopy)** Tab.
1. Browse and upload an image (PNG, JPG) from your hard drive exactly.
2. The AI will spin up and analyze the structure morphology.
3. A response indicating whether a "BENIGN" or "POLYP / MALIGNANT" tumor has been found will immediately be displayed.
4. Additionally, a **Grad-CAM heatmap** will render over the uploaded image, highlighting locally the hot-spots making the neural network arrive at that conclusion.

---

## 3. Disclaimers

> **WARNING**: Information furnished through the ColonAI Simulation platform does **NOT** substitute professional clinical judgement. 
All AI predictions *must* be backed by histology panels. The heatmap should simply be understood as a localization guidance system for the referring physician.
