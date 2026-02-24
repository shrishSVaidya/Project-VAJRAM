# Project VAJRAM
### **V**irtual **A**gent for **J**oint **R**isk **A**nalysis of **M**ultiple Myeloma

## 1. Executive Summary
Project VAJRAM is a specialized, agentic AI solution designed to democratize advanced care for **Multiple Myeloma (MM)**, bridging the critical gap between sophisticated oncological diagnostics and resource-constrained healthcare environments. By synergizing Google's SOTA medical foundation model - **MedGemma** (for clinical reasoning), **MedGemma 1.5** (for multimodal understanding), and **Med-SigLIP** (for high-fidelity medical imaging analysis) - VAJRAM functions as an end-to-end digital oncologist. It autonomously ingests and synthesizes fragmented patient data, including paper-based medical history, complex radiological scans, and longitudinal behavioral markers, transforming raw multimodal inputs into actionable, life-saving clinical insights for early detection and personalized management.

## 2. Problem Statement
Multiple Myeloma is the 2nd most common hematological malignancy and a complex blood cancer that is currently incurable but highly treatable. However, in most of the cases, patient outcomes are compromised by:
* **Data Fragmentation:** Critical history is trapped in paper reports and unorganized files.
* **Diagnostic Delay:** Lack of specialized hematologists/radiologists leads to late detection.
* **Reactive Intervention:**  Because disease trajectory is rarely modeled predictively, therapy often commences reactively only *after* severe, irreversible organ damage (e.g., renal failure, lytic bone fractures) has already occurred.
* **The Knowledge Translation Gap:** The treatment paradigm for Multiple Myeloma is exceptionally complex. General physicians and rural oncologists often struggle to navigate dense, rapidly changing clinical guidelines (NCCN/ESMO), leading to suboptimal, generalized therapeutic choices for Relapsed/Refractory (RRMM) patients.

## 3. Proposed Solution: The 5-Pillar Architecture

![5-Pillar Architecture](Project_VAJRAM_thumbnail_resized.jpg)

### Module 1: Intelligent Digitization (The "Digital Twin")
* **Function:** Users scan paper lab reports, handwritten prescriptions, and X-ray films via smartphone camera.
* **AI Technology:** Uses the base **MedGemma 1.5** multimodal capabilities for high-accuracy OCR and entity extraction to parse unstructured medical text into structured data (FHIR standard).
* **Value:** Solves the **Digitization & Accessibility** crisis. Creates a time-stamped, longitudinal health timeline accessible to both patient and doctor, ensuring no data point is lost.

### Module 2: Risk Stratification & Early Detection
* **Function:** Analyzes the digitized longitudinal history (blood trends, SPEP results) and patient-reported behavioral changes to calculate a "Myeloma Risk Score."
* **AI Technology:** **Dynamic LoRA Switching (`lora_risk`):** A parameter-efficient adapter fine-tuned on tabular EHR data (MIMIC-IV) injected into the base MedGemma model on-the-fly. 
* **Value:** Solves the **Initial Diagnosis** bottleneck. Empowers patients to check for "red flags" (e.g., rising Globulin + Anemia) and seek specialist care *before* kidney failure occurs.

### Module 3: Automated Diagnostic Confirmation
* **Function:** Processes uploaded DICOM/JPEG images of CT scans, MRI, and Bone Marrow Biopsy slides to detect lytic lesions and plasma cell infiltration.
* **AI Technology:** **Dynamic LoRA Switching (`lora_vision`):** An adapter fine-tuned on high-fidelity pathology and radiology datasets (SegPC, TCIA) to grant the base MedGemma model specialized spatial anomaly detection.
* **Value:** Solves the **Specialist Shortage**. Acts as an "AI Second Reader" to assist general radiologists in identifying subtle bone defects specific to Myeloma.

### Module 4: Disease Progression Modeling
* **Function:** Predicts the trajectory of the disease (e.g., progression from Smoldering Myeloma to Active MM) based on the *velocity* of change in biomarkers over time.
* **AI Technology:** **Dynamic LoRA Switching (`lora_progression`):** An instruction-tuned adapter trained on serialized temporal patient timelines (MMRF CoMMpass) to enable longitudinal predictive reasoning.
* **Value:** Solves the **Reactive Treatment** issue. Enables "Pre-emptive Intervention" by alerting doctors to accelerating disease activity before clinical symptoms appear.

### Module 5: Guideline-Based Therapy Recommender (Clinical Decision Support)
* **Function:** Acts as an oncologist's co-pilot by synthesizing the patient’s current disease state, past drug resistance, and frailty index to recommend personalized treatment pathways (e.g., VRd induction, Dara-VRd, or CAR-T cell therapy).
* **AI Technology:** **Agentic RAG (Retrieval-Augmented Generation):** The base MedGemma 1.5 model queries a custom vector database embedded with the latest NCCN and ESMO clinical guidelines for Multiple Myeloma. 
* **Value:** Solves the **Knowledge Translation Gap**. Oncology guidelines change rapidly. This module ensures that even rural doctors without hematology specializations can offer their patients state-of-the-art, evidence-based treatment regimens for both newly diagnosed and Relapsed/Refractory (RRMM) cases.


## 4. Technology stack and methodology

### 4.1 Training Methodology & Architectural Strategy

Project VAJRAM employs **Parameter-Efficient Fine-Tuning (PEFT)** and **Retrieval-Augmented Generation (RAG)** to inject specialized oncological expertise into the **MedGemma 1.5** base model without compromising its foundational medical reasoning. 

#### Training Pipeline
1. **Multimodal Data Serialization:** We curate and standardize diverse clinical inputs into model-interpretable formats. This includes parsing OCR lab reports (Module 1), sequencing EHR blood biomarkers (Module 2), standardizing bone marrow imaging (Module 3), and flattening longitudinal patient trajectories (Module 4).
2. **Domain-Specific LoRA Fine-Tuning:**  For diagnostic and prognostic tasks (Modules 2–4), we independently train lightweight **LoRA adapters** on specialized datasets. This allows the model to learn complex oncological heuristics (e.g., **CRAB criteria, R-ISS staging**) while preventing **catastrophic forgetting**.
3. **RAG-Powered Decision Support:**  To guarantee patient safety and eliminate **clinical hallucination**, Module 5 utilizes a strict RAG architecture. Instead of fine-tuning, the base model queries a vector database of the latest **NCCN and ESMO clinical guidelines** to generate evidence-based therapy pathways.

#### Edge-Deployability via Dynamic Mixture-of-Adapters (MoA)
VAJRAM is engineered specifically for resource-constrained healthcare settings. Instead of hosting multiple computationally heavy models, the system loads the foundational **MedGemma 1.5 model** into memory exactly once. 

As the diagnostic workflow progresses, the agent dynamically hot-swaps the specialized, lightweight **LoRA adapters** into VRAM on demand. This **frugal compute strategy** ensures that high-fidelity, privacy-preserving AI can be executed entirely on local, mid-tier hardware without requiring cloud connectivity.


### 4.2 Datasets used for Fine-tuning, RAG, and Evaluation:

| Dataset Name | Task Type | Modality | Usage | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Medical Lab Reports** (Synthetic & Kaggle) | OCR & Entity Extraction | Image (Scanned Docs) + Text | Fine-tuning & Eval | Used to train and test the Intelligent Digitization engine (**Module 1**). |
| **MIMIC-IV** (PhysioNet) | Risk Stratification / Tabular Prediction | Tabular / EHR (Longitudinal Labs) | Fine-tuning | Filtered via ICD codes `C90.0`/`203.0` to extract the Myeloma cohort for CRAB criteria risk scoring (**Module 2**). |
| **MIMIC-IV-Note** (PhysioNet) | Symptom Extraction / Multimodal Fusion | Text (Unstructured Clinical Notes) | Fine-tuning | Mapped to hospital admissions to extract real-world behavioral symptoms (HPI/Chief Complaint), creating a multimodal diagnostic dataset (**Module 2**). |
| **SegPC-2021** (Kaggle) | Cell Classification / Segmentation | Image (Microscopic Bone Marrow) | Fine-tuning | High-fidelity plasma cell images; used to train the LoRA Vision Engine (**Module 3**). |
| **TCIA Myeloma Collection** | External Validation / Classification | Image (Histopathology) + Text (SUMS) | Evaluation | 85 histopath samples; used as a holdout set to prove cross-dataset generalization (**Module 3**). |
| **MMRF CoMMpass** (IA15) | Progression Modeling / Time-Series | Tabular / Text (Clinical JSONs) | Fine-tuning & Eval | Flattened nested JSONs to extract critical survival targets like `days_to_death` (**Module 4**). |
| **NCCN & ESMO Clinical Guidelines** | Knowledge Retrieval (RAG) | Text (Medical PDFs) | Vector Database | Embedded into a vector store to power the Guideline-Based Therapy Recommender (**Module 5**).  |

### 4.3 Training Details

#### Module 2: Risk Stratification & Early Detection (`lora_risk`)

* **Objective:** Instruction-tune the MedGemma 1.5 base model to calculate a "Myeloma Risk Score" by analyzing a snapshot of structured tabular EHR data (lab values, demographics) combined with unstructured NLP-extracted behavioral symptoms.
* **Dataset:** Real-world retrospective data sourced from MIMIC-IV.
* **Multimodal Data Fusion Pipeline:**
    1. **Symptom Extraction:** Uses a locally hosted LLM (Qwen 2.4 3B Instruction Tuned via TGI) to scan unstructured discharge summaries and extract exactly one dense clinical sentence describing the patient's behavioral symptoms and chief complaints *on admission*.
    2. **Lab Integration:** Filters and pivots 13 critical longitudinal lab values corresponding to CRAB criteria (Creatinine, Calcium, Hemoglobin), R-ISS Staging (Albumin, Beta-2 Microglobulin, LDH), and Tumor Burden (SPEP, UPEP, Free Light Chains) into admission-level timelines.
    3. **The 3-Way Merge:** Fuses the tabular labs, the NLP-extracted symptoms, and baseline demographics (age, gender) into a unified patient state. Records lacking viable baseline labs (missing both Hemoglobin and Creatinine) are filtered out.
* **Dynamic Data Augmentation Strategy:**
    To make the model resilient to missing or incomplete real-world clinical records, the data engine generates 4 distinct training scenarios for every single patient admission:
    1. **`baseline`:** Utilizes all available fused data.
    2. **`silent_patient`:** Masks the NLP symptoms to simulate cases where a patient is unable to provide a history ("Patient unable to provide history / No symptoms recorded.").
    3. **`primary_care`:** Simulates a general practice setting by aggressively dropping specialized oncology labs (B2M, SPEP, Kappa/Lambda Ratio, LDH).
    4. **`messy_ehr`:** Simulates disorganized records by randomly dropping 1 to 2 standard CRAB lab values (Creatinine, Calcium, Hemoglobin, or Albumin).
* **Dynamic Label Generation (Ground Truth):**
         The training targets are synthesized programmatically using strict clinical rule-based logic evaluating CRAB flags and Tumor flags.[1-5]
    * **CRAB Flags:** Triggered by Anemia (Hb < 10.0), Renal impairment (Creatinine > 2.0), or Hypercalcemia (Calcium > 11.0).
    * **Tumor Flags:** Triggered by Elevated B2M ($\ge$ 3.5), Positive M-Spike (> 0.0), or an Abnormal FLC Ratio (> 1.65 or < 0.26).
    * The synthesized AI response dynamically scales from "Standard/Low Risk" to "Critical Risk: Active Multiple Myeloma with End-Organ Damage" based on the presence of these flags. It is also context-aware; for example, if CRAB flags are present but tumor markers are missing (e.g., in the `primary_care` augmentation), the target label specifically instructs an "urgent hematology referral for a full myeloma workup".
* **Dataset Formatting:**
    * Data is constructed into multi-turn conversational pairs mapping the instruction (patient profile, symptoms, and panels) to the assistant's clinical triage assessment.
    * Exported as `medgemma_risk_training_augmented.jsonl`.

#### Module 3: Automated Diagnostic Confirmation (`lora_vision`)

* **Objective:** Instruction-tune the MedGemma 1.5 (4B) Vision-Language Model to serve as an "AI Second Reader" capable of detecting malignant plasma cells within high-resolution bone marrow biopsy slides. 
* **Dataset:** TCIA SegPC-2021 dataset.
* **Vision Data Processing & Fusion Pipeline:**
    1. **One-to-Many Mask Fusion:** Dynamically maps a single Whole Slide Image (WSI) patch to multiple corresponding individual cell instance masks. These overlapping instance masks are merged into a single global segmentation mask using matrix maximum operations (`np.maximum`), ensuring no cell of interest is dropped.
    2. **WSI Patching & Augmentation:** To bypass memory bottlenecks inherent in gigapixel WSIs, the model is trained on 512x512 pixel patches. The training set utilizes a 4x data multiplier, generating multiple robust iterations per image via random cropping, random horizontal/vertical flipping (p=0.5), and color jittering (brightness/contrast adjustments). For evaluation, a deterministic, padded sliding-window grid is used.
    3. **Dynamic Labeling:** If the generated global mask patch contains any white pixels (abnormal cells present), the patch is labeled "Yes". If strictly background tissue is present, it is labeled "No".
* **Multimodal Dataset Formatting:**
    * The 85/15 train/eval split is packaged into conversational dictionaries natively compatible with MedGemma 1.5. 
    * The system prompts the model with: *"Analyze this 512x512 Bone Marrow Biopsy patch. Does it contain any plasma cells indicative of Multiple Myeloma?"* mapped directly to the dynamic binary target.
* **QLoRA Fine-Tuning Architecture:**
    * **Quantization:** The base MedGemma 1.5 model is loaded in 4-bit precision (NF4 quantization type, bfloat16 compute dtype, double quantization enabled) using `bitsandbytes` to optimize VRAM utilization.
    * **Adapter Matrix:** A LoRA adapter is injected into all major linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with a rank ($r$) of 16, an alpha of 32, and a dropout of 0.05.
    * **Hyperparameters:** Trained using the `SFTTrainer` with a fused AdamW optimizer (`adamw_torch_fused`) for 1 epoch. The learning rate is set to $2 \times 10^{-4}$ following a linear schedule with a 0.03 warmup ratio and a maximum gradient norm of 0.3. Gradient accumulation (16 steps) is utilized to simulate a larger batch size on constrained hardware.

#### Module 4: Disease Progression Modeling (`lora_progression`)

* **Objective:** Instruction-tune the MedGemma 1.5 base model to perform temporal reasoning over $n$-length clinical histories, enabling it to predict Multiple Myeloma progression (e.g., Smoldering to Active MM) or detect biochemical relapse based on biomarker velocity.
* **Dataset:** MMRF CoMMpass (Clinical Data JSON).
* **Data Processing & Serialization Pipeline:**
    1. **Longitudinal Filtering:** Patients with fewer than two recorded clinical visits are excluded to ensure sufficient data for trajectory modeling.
    2. **Chronological Sorting:** Clinical follow-up visits are ordered sequentially based on the `days_to_follow_up` metric to reconstruct a strict past-to-present timeline.
    3. **Targeted Biomarker Extraction:** The pipeline specifically extracts core CRAB criteria and tumor markers (M-Protein, Hemoglobin, Calcium, Creatinine, Platelets) from nested arrays, ignoring irrelevant EHR noise to optimize the LLM's context window.
    4. **NLP Framing:** Tabular data is transformed into a serialized text narrative (e.g., `- Day 90: M Protein: 1.2 g/dL, Hemoglobin: 11.5 g/dL`) suitable for VLM/LLM ingestion.
* **Dynamic Label Generation (Ground Truth):**
    * The training targets are dynamically generated on-the-fly using strict **International Myeloma Working Group (IMWG)** criteria for Progressive Disease (PD).
    * The algorithm scans the $n$-visit timeline to identify the patient's **nadir** (lowest historical M-Protein value).
    * **"Progression" Labels:** Assigned if the current visit demonstrates both a $\ge 25\%$ relative increase **and** an absolute increase of $\ge 0.5$ g/dL from the nadir. 
    * **"Stable/Favorable" Labels:** Assigned if the biomarkers are dropping to a new nadir or remaining below the progression threshold.
* **Dataset Formatting & Splitting:**
    * Data is constructed into multi-turn conversational pairs mapping the User Prompt (longitudinal history) to the Assistant Target (clinical reasoning and IMWG-backed prediction).
    * The total dataset is systematically split into **70% Training, 10% Evaluation, and 20% Testing** partitions using a fixed seed to ensure zero data leakage across splits.
    * The final output is serialized to disk as both a native Hugging Face `DatasetDict` (for efficient batching) and readable `.jsonl` files (for manual clinical auditing).


#### Module 5: Guideline-Based Therapy Recommender (RAG)





## 5. UseCase and Governance


## 6. Limitations and Future Directions



## 7. Conclusion
Project VAJRAM is not just a diagnostic tool; it is a comprehensive **Healthcare Orchestrator**. By combining the analytical power of AI with the logistical reality of patient needs (digitization and finance), it aims to democratize survival for Multiple Myeloma patients.


## 6. References
1. Rajkumar, S. V., Dimopoulos, M. A., Palumbo, A., et al. (2014). International Myeloma Working Group updated criteria for the diagnosis of multiple myeloma. The Lancet Oncology, 15(12), e538-e548.
2. Palumbo, A., Avet-Loiseau, H., Oliva, S., et al. (2015). Revised International Staging System for Multiple Myeloma: A Report From Intergroupe Francophone du Myelome, Oncology Wing, and Blood and Marrow Transplant Clinical Trials Network. Journal of Clinical Oncology, 33(26), 2863-2869.
3. Katzmann, J. A., Clark, R. J., Abraham, R. S., et al. (2002). Serum reference intervals and diagnostic ranges for free kappa and free lambda light chains: relative sensitivity for detection of monoclonal light chains. Clinical Chemistry, 48(9), 1437-1444.
4. NCCN Clinical Practice Guidelines in Oncology (NCCN Guidelines) for Multiple Myeloma, Version 1.2025.
5. Nakaya A, Fujita S, Satake A, Nakanishi T, Azuma Y, Tsubokura Y, Hotta M, Yoshimura H, Ishii K, Ito T, Nomura S. Impact of CRAB Symptoms in Survival of Patients with Symptomatic Myeloma in Novel Agent Era. Hematol Rep. 2017 Feb 23;9(1):6887. doi: 10.4081/hr.2017.6887. PMID: 28286629; PMCID: PMC5337823.
6. Kumar, S., Paiva, B., Anderson, K. C., Durie, B., Landgren, O., Moreau, P., ... & Avet-Loiseau, H. (2016). International Myeloma Working Group consensus criteria for response and minimal residual disease assessment in multiple myeloma. The Lancet Oncology, 17(8), e328-e346. https://doi.org/10.1016/S1470-2045(16)30206-6