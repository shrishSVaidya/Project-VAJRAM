import pandas as pd
import json
import numpy as np
import os
import re

# 1. Define Paths
LABS_FILE = r"data\mm_longitudinal_labs.csv"
NOTE_FILE = r"E:\ajja\data\mimiciv\3.1\mimic-iv-note-deidentified-free-text-clinical-notes-2.2\note\discharge.csv.gz" 
OUTPUT_JSONL = os.path.join(os.getcwd(), "data", "medgemma_real_risk_training.jsonl")

print("Loading Real-World MIMIC-IV Lab Data...")
labs_df = pd.read_csv(LABS_FILE)

# Get the unique hospital admission IDs for our Myeloma patients
myeloma_hadm_ids = set(labs_df['hadm_id'].dropna().unique())
print(f"Tracking {len(myeloma_hadm_ids)} unique hospital admissions.")

# 2. Extract Doctor's Notes (Robust String Searching)
print("Scanning Discharge Summaries for Behavioral Symptoms...")

extracted_notes = []

# Read the massive notes file in chunks
for chunk in pd.read_csv(NOTE_FILE, compression='gzip', chunksize=50000, usecols=['hadm_id', 'text']):
    myeloma_notes = chunk[chunk['hadm_id'].isin(myeloma_hadm_ids)]
    
    for _, row in myeloma_notes.iterrows():
        text = str(row['text'])
        symptoms = ""
        
       
        
        # Failsafe Method 1: Look for History of Present Illness
        hpi_index = text.lower().find("history of present illness:")
        if hpi_index != -1:
            # Grab the next 800 characters after the heading
            start = hpi_index + len("history of present illness:")
            symptoms = text[start : start + 800]
        else:
            # Failsafe Method 2: Look for Chief Complaint
            cc_index = text.lower().find("chief complaint:")
            if cc_index != -1:
                start = cc_index + len("chief complaint:")
                symptoms = text[start : start + 300]
        # if _ ==143:
        #     print(text)
        #     print(symptoms)
        #     x
        
        if symptoms:
            # Clean up the text (remove MIMIC's anonymization underscores and weird spacing)
            symptoms = re.sub(r'_+', '', symptoms) # removes ___
            symptoms = " ".join(symptoms.split())  # fixes spacing
            
            # Truncate at the last full sentence so it doesn't cut off mid-word
            last_period = symptoms.rfind('.')
            if last_period != -1:
                symptoms = symptoms[:last_period + 1]
                
            extracted_notes.append({'hadm_id': row['hadm_id'], 'clinical_symptoms': symptoms.strip()})

notes_df = pd.DataFrame(extracted_notes)
# Drop duplicates, keeping the most recent note
notes_df = notes_df.drop_duplicates(subset=['hadm_id'], keep='last')

print(f"Successfully extracted real symptoms for {len(notes_df)} admissions.")

# 3. Pivot the Lab Data 
lab_map = {
    50912: 'Creatinine', 50893: 'Calcium', 51222: 'Hemoglobin',
    50862: 'Albumin', 53085: 'Albumin', 52022: 'Albumin',
    50881: 'Beta2_Microglobulin', 50954: 'LDH', 50975: 'SPEP',
    51098: 'UPEP', 51625: 'Free_Kappa', 51627: 'Free_Lambda', 51626: 'Kappa_Lambda_Ratio'
}
labs_df['lab_name'] = labs_df['itemid'].map(lab_map)

pivot_df = labs_df.pivot_table(
    index=['subject_id', 'hadm_id'], 
    columns='lab_name', 
    values='valuenum',
    aggfunc='mean'
).reset_index()

# 4. Merge Real Labs with Real Symptoms
final_df = pd.merge(pivot_df, notes_df, on='hadm_id', how='inner')
print(f"Successfully merged {len(final_df)} patient timelines with their clinical notes.")
