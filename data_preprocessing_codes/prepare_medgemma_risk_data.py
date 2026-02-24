import pandas as pd
import json
import numpy as np
import os
import random

# --- 1. Load the Three Raw CSVs ---
print("Loading raw MIMIC-IV datasets...")
labs_df = pd.read_csv(os.path.join("data", "mm_longitudinal_labs.csv"))
cohort_df = pd.read_csv(os.path.join("data", "mm_patient_cohort.csv"))
symptoms_df = pd.read_csv(os.path.join("data", "qwen_extracted_symptoms.csv"))

# --- 2. Map and Pivot the Lab Data ---
print("Pivoting lab data into admission-level timelines...")
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

# --- 3. Execute Multimodal Fusion (The 3-Way Merge) ---
print("Fusing Labs, Demographics, and NLP Symptoms...")
merged_df = pd.merge(pivot_df, symptoms_df, on='hadm_id', how='left')
final_df = pd.merge(merged_df, cohort_df[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')

# Clean up missing symptoms
final_df['clinical_symptoms'] = final_df['clinical_symptoms'].fillna("No acute behavioral complaints documented.")

# Ensure all expected lab columns exist to prevent KeyErrors
expected_cols = ['Creatinine', 'Calcium', 'Hemoglobin', 'Albumin', 'Beta2_Microglobulin', 
                 'LDH', 'SPEP', 'UPEP', 'Free_Kappa', 'Free_Lambda', 'Kappa_Lambda_Ratio']
for col in expected_cols:
    if col not in final_df.columns:
        final_df[col] = np.nan

# Drop rows that don't have at least Hemoglobin or Creatinine for a viable baseline
final_df = final_df.dropna(subset=['Hemoglobin', 'Creatinine'], how='all')
print(f"Total valid patient admissions ready for augmentation: {len(final_df)}")

# --- 4. The Dynamic Data Augmentation Engine ---
def generate_clinical_scenario(row, strategy="baseline"):
    # Copy data into a mutable dictionary
    data = {
        'age': int(row['anchor_age']) if pd.notna(row['anchor_age']) else "Unknown",
        'gender': "Male" if row.get('gender') == 'M' else "Female" if row.get('gender') == 'F' else "Unknown",
        'symptoms': row.get('clinical_symptoms', "No acute behavioral complaints documented."),
        'Creatinine': row['Creatinine'],
        'Calcium': row['Calcium'],
        'Hemoglobin': row['Hemoglobin'],
        'Albumin': row['Albumin'],
        'Beta2_Microglobulin': row['Beta2_Microglobulin'],
        'LDH': row['LDH'],
        'SPEP': row['SPEP'],
        'Kappa_Lambda_Ratio': row['Kappa_Lambda_Ratio']
    }
    
    # APPLY DROPOUT STRATEGIES
    if strategy == "silent_patient":
        data['symptoms'] = "Patient unable to provide history / No symptoms recorded."
        
    elif strategy == "primary_care":
        data['Beta2_Microglobulin'] = np.nan
        data['SPEP'] = np.nan
        data['Kappa_Lambda_Ratio'] = np.nan
        data['LDH'] = np.nan
        
    elif strategy == "messy_ehr":
        labs_to_potentially_drop = ['Creatinine', 'Calcium', 'Hemoglobin', 'Albumin']
        drop_count = random.randint(1, 2)
        labs_to_drop = random.sample(labs_to_potentially_drop, drop_count)
        for lab in labs_to_drop:
            data[lab] = np.nan

    # FORMAT THE PROMPT
    cr = f"{data['Creatinine']:.2f} mg/dL" if pd.notna(data['Creatinine']) else "Not tested"
    ca = f"{data['Calcium']:.2f} mg/dL" if pd.notna(data['Calcium']) else "Not tested"
    hb = f"{data['Hemoglobin']:.2f} g/dL" if pd.notna(data['Hemoglobin']) else "Not tested"
    alb = f"{data['Albumin']:.2f} g/dL" if pd.notna(data['Albumin']) else "Not tested"
    b2m = f"{data['Beta2_Microglobulin']:.2f} mg/L" if pd.notna(data['Beta2_Microglobulin']) else "Not tested"
    ldh = f"{data['LDH']:.2f} U/L" if pd.notna(data['LDH']) else "Not tested"
    spep = f"{data['SPEP']:.2f} g/dL" if pd.notna(data['SPEP']) else "Not tested"
    flc_ratio = f"{data['Kappa_Lambda_Ratio']:.2f}" if pd.notna(data['Kappa_Lambda_Ratio']) else "Not tested"
    
    instruction = (
        f"Assess the Multiple Myeloma risk profile for this {data['age']}-year-old {data['gender']} patient:\n"
        f"- Patient Reported Symptoms: {data['symptoms']}\n"
        f"- CRAB Panel -> Creatinine: {cr}, Calcium: {ca}, Hemoglobin: {hb}\n"
        f"- Tumor/Staging Panel -> Albumin: {alb}, Beta-2 Microglobulin: {b2m}, LDH: {ldh}, M-Spike (SPEP): {spep}, FLC Ratio: {flc_ratio}"
    )
    
    # CALCULATE RE-CALIBRATED GROUND TRUTH
    crab_flags = []
    tumor_flags = []
    
    if pd.notna(data['Hemoglobin']) and data['Hemoglobin'] < 10.0: crab_flags.append("Anemia")
    if pd.notna(data['Creatinine']) and data['Creatinine'] > 2.0: crab_flags.append("Renal impairment")
    if pd.notna(data['Calcium']) and data['Calcium'] > 11.0: crab_flags.append("Hypercalcemia")
        
    if pd.notna(data['Beta2_Microglobulin']) and data['Beta2_Microglobulin'] >= 3.5: tumor_flags.append("Elevated B2M")
    if pd.notna(data['SPEP']) and data['SPEP'] > 0.0: tumor_flags.append("Positive M-Spike")
    if pd.notna(data['Kappa_Lambda_Ratio']) and (data['Kappa_Lambda_Ratio'] > 1.65 or data['Kappa_Lambda_Ratio'] < 0.26): 
        tumor_flags.append("Abnormal FLC Ratio")

    # SYNTHESIZE LLM RESPONSE
    if len(crab_flags) > 0 and len(tumor_flags) > 0:
        response = f"Critical Risk: Active Multiple Myeloma with End-Organ Damage. Patient exhibits CRAB criteria ({', '.join(crab_flags)}) and high tumor burden markers ({', '.join(tumor_flags)}). Immediate intervention required."
    elif len(crab_flags) > 0 and strategy == "primary_care":
        response = f"High Risk: Suspected End-Organ Damage. Patient exhibits CRAB criteria ({', '.join(crab_flags)}). However, specialized tumor markers (SPEP/FLC) are missing. Given the clinical presentation, an urgent hematology referral for a full myeloma workup is required."
    elif len(crab_flags) > 0:
        response = f"High Risk: Suspected End-Organ Damage. Patient exhibits CRAB criteria ({', '.join(crab_flags)}). Correlate with specialized tumor markers to confirm active disease progression."
    elif len(tumor_flags) > 0:
        response = f"Moderate/High Risk: Smoldering or Active Disease. Tumor markers are flagged ({', '.join(tumor_flags)}), but available CRAB markers do not currently show severe downstream organ failure. Monitor closely."
    else:
        response = "Standard/Low Risk based on currently available data. No overt CRAB criteria or tumor markers are met. Continue routine clinical monitoring based on presenting symptoms."
        
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
    }

# --- 5. Execution & Augmentation Loop ---
print("Generating Augmented Training Dataset...")
training_data = []
strategies = ["baseline", "silent_patient", "primary_care", "messy_ehr"]

for _, row in final_df.iterrows():
    # Generate 4 distinct scenarios per patient admission
    for strategy in strategies:
        prompt_json = generate_clinical_scenario(row, strategy)
        training_data.append(prompt_json)

# --- 6. Save to JSONL ---
output_file = os.path.join("data", "medgemma_risk_training_augmented.jsonl")
with open(output_file, 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

print(f"✅ Success! Generated {len(training_data)} augmented training examples and saved to {output_file}.")