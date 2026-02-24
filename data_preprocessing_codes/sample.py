import pandas as pd
import os

# --- 1. Define Paths ---
BASE_DIR = r"E:\ajja\data\mimiciv\3.1\hosp"
DIAGNOSES_FILE = os.path.join(BASE_DIR, "diagnoses_icd.csv.gz")
PATIENTS_FILE = os.path.join(BASE_DIR, "patients.csv.gz")
LABEVENTS_FILE = os.path.join(BASE_DIR, "labevents.csv.gz")

OUTPUT_COHORT = "mm_patient_cohort.csv"
OUTPUT_LABS = "mm_longitudinal_labs.csv"

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


print("Extracting patient demographics...")
df_patients = pd.read_csv(PATIENTS_FILE, compression='gzip')
mm_patients_df = df_patients[df_patients['subject_id'].isin(mm_subject_ids)]

print(mm_patients_df.head())


