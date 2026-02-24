import pandas as pd
import os

# --- 1. Define Paths ---
BASE_DIR = r"E:\ajja\data\mimiciv\3.1\hosp"
DIAGNOSES_FILE = os.path.join(BASE_DIR, "diagnoses_icd.csv.gz")
PATIENTS_FILE = os.path.join(BASE_DIR, "patients.csv.gz")
LABEVENTS_FILE = os.path.join(BASE_DIR, "labevents.csv.gz")

OUTPUT_COHORT = os.path.join(os.getcwd(), "data", "mm_patient_cohort.csv")
OUTPUT_LABS = os.path.join(os.getcwd(), "data", "mm_longitudinal_labs.csv")

print("Starting extraction process...")

# --- 2. Find Multiple Myeloma Patients ---
# ICD-9 codes start with 2030, ICD-10 codes start with C900
print("Finding Myeloma patients from diagnoses_icd...")
df_diag = pd.read_csv(DIAGNOSES_FILE, compression='gzip')

# Filter for Myeloma
mm_mask = (
    (df_diag['icd_version'] == 9) & (df_diag['icd_code'].str.startswith('2030', na=False)) |
    (df_diag['icd_version'] == 10) & (df_diag['icd_code'].str.startswith('C900', na=False))
)
mm_diagnoses = df_diag[mm_mask]

# Get unique subject_ids
mm_subject_ids = mm_diagnoses['subject_id'].unique()
print(f"Found {len(mm_subject_ids)} unique Multiple Myeloma patients.")

# --- 3. Extract Demographics & Mortality ---
print("Extracting patient demographics...")
df_patients = pd.read_csv(PATIENTS_FILE, compression='gzip')
mm_patients_df = df_patients[df_patients['subject_id'].isin(mm_subject_ids)]
mm_patients_df.to_csv(OUTPUT_COHORT, index=False)
print(f"Saved patient demographics to {OUTPUT_COHORT}")

# --- 4. Extract Relevant Lab Events (Chunked to save RAM) ---
# Critical Myeloma Markers: 
# 50912: Creatinine, 50893: Calcium, 51222: Hemoglobin, 50976: Total Protein
TARGET_LAB_IDS = [
    # CRAB Criteria
    50912, 50893, 51222, 
    
    # R-ISS Staging (Albumin + B2M + LDH)
    50862, 53085, 52022, 50881, 50954, 
    
    # Tumor Burden (SPEP, UPEP, Light Chains, Ratio)
    50975, 51098, 51625, 51627, 51626
]

print(f"Extracting longitudinal labs from 2.5GB file (This will take a few minutes)...")

chunk_size = 5_000_000 # Read 5 million rows at a time
first_chunk = True

# Process labevents in chunks
for chunk in pd.read_csv(LABEVENTS_FILE, compression='gzip', chunksize=chunk_size):
    # Filter for our specific patients AND our specific target lab tests
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(mm_subject_ids)) & 
        (chunk['itemid'].isin(TARGET_LAB_IDS))
    ]
    
    # Drop rows where the test result is missing
    filtered_chunk = filtered_chunk.dropna(subset=['valuenum'])
    
    # Keep only necessary columns to keep file size small
    filtered_chunk = filtered_chunk[['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom']]
    
    # Append to output CSV
    filtered_chunk.to_csv(
        OUTPUT_LABS, 
        mode='a', 
        header=first_chunk, 
        index=False
    )
    first_chunk = False
    print(f"Processed a chunk... appending matches to {OUTPUT_LABS}")

print("Extraction complete! Your data is ready.")