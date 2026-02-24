from huggingface_hub import InferenceClient
from tqdm import tqdm
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
MAX_CONCURRENCY = 8  # Adjust based on your GPU VRAM and TGI batch size
TGI_URL = "http://localhost:8080/"

# Define Paths
LABS_FILE = "/home/user/cxr_pretraining/mm_longitudinal_labs.csv"
NOTE_FILE = "/home/user/cxr_pretraining/discharge.csv.gz" 
OUTPUT_SYMPTOMS = "qwen_extracted_symptoms.csv"

# --- Client Setup ---
client = InferenceClient(base_url=TGI_URL)

SYSTEM_PROMPT = """You are a strict oncology triage AI extracting the Chief Complaint from medical records. Your sole objective is to synthesize the patient's physical and behavioral symptoms ON ADMISSION into exactly ONE dense clinical sentence.

CRITICAL RULES:
1. ONLY extract how the patient felt or behaved RIGHT BEFORE arriving at the hospital (e.g., dizziness, weakness, bone pain, lethargy, confusion, hematuria).
2. NEVER mention the hospital course, treatments received, lab values, or discharge instructions.
3. If the text looks like a letter (e.g., "Dear Patient"), ignore the letter formatting and extract only the reason they came to the hospital.
4. You MUST begin your response exactly with the phrase: "Patient presented with "
5. Output in 2-3 sentences that covers all the behavioural markers given in the note.

EXAMPLE INPUT:
"This is a 65yo male with a history of IgG Multiple Myeloma, CAD, and HTN who presents with 3 weeks of worsening lower back pain, acute dizziness, and lethargy. He underwent an MRI which showed a compression fracture. He was given Cefepime."

EXAMPLE OUTPUT:
Patient presented with three weeks of worsening lower back pain, acute dizziness, and lethargy.
"""


# --- Worker Function ---
def process_note(args):
    index, hadm_id, raw_text = args
    
    # OPTIMIZATION: Truncate to first 3500 chars where HPI/Chief Complaint lives
    truncated_text = str(raw_text)[:-10000]
    
    try:
        response = client.chat.completions.create(
            # TGI uses the loaded model automatically, but you can pass a placeholder
            model="tgi", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Synthesize the Chief Complaint and admission symptoms from this note snippet:\n\n{truncated_text}"}
            ],
            temperature=0.1, # Low temp for factual extraction
            max_tokens=200   # We only need one sentence
        )

        symptoms = response.choices[0].message.content.strip()
        
        # We no longer look for JSON, we just return the raw string
        return {"hadm_id": hadm_id, "clinical_symptoms": symptoms, "original_notes": raw_text}

    except Exception as e:
        print(f"[ERROR] ({index}) Failed to process hadm_id {hadm_id}: {e}")
        return {"hadm_id": hadm_id, "clinical_symptoms": "Extraction failed.", "original_notes": raw_text}


if __name__ == "__main__":
    
    # 1. Load target Admission IDs from our lab data
    print("Loading target hospital admissions...")
    labs_df = pd.read_csv(LABS_FILE)
    target_hadms = labs_df['hadm_id'].dropna().unique()
    print(f"Tracking {len(target_hadms)} unique hospital admissions.")

    
    # 2. Extract only the relevant notes from the massive discharge file
    print("Pre-loading and filtering MIMIC-IV discharge summaries...")
    notes_df = pd.read_csv(NOTE_FILE, compression='gzip', usecols=['hadm_id', 'text'])
    notes_df = notes_df[notes_df['hadm_id'].isin(target_hadms)]
    
    # Drop duplicates to ensure we only process the final discharge summary per admission
    notes_df = notes_df.drop_duplicates(subset=['hadm_id'], keep='last')
    
    # Convert to a list of tuples for the thread pool
    inputs = [(i, row['hadm_id'], row['text']) for i, row in notes_df.iterrows()]
    print(f"Found {len(inputs)} matching clinical notes.")

    
    # 3. Execute Multi-threaded Extraction
    results = []
    print(f"Executing extraction with {MAX_CONCURRENCY} concurrent threads...")

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = {executor.submit(process_note, inp): inp for inp in inputs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            # print("==================================================")
            # print(res)
            if res:
                results.append(res)

    print(f"✅ Completed. Successful extractions: {len(results)}")

    # 4. Save Output to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_SYMPTOMS, index=False)
    print(f"Saved real-world symptoms to {OUTPUT_SYMPTOMS}")