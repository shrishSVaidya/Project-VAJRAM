import json
import os
from datasets import Dataset, DatasetDict

def build_and_save_progression_dataset(json_path, output_dir="module4_progression_dataset"):
    print(f"Loading MMRF CoMMpass clinical data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    conversational_pairs = []
    
    # Core biomarkers indicating Myeloma progression
    target_labs = ["M Protein", "Hemoglobin", "Calcium", "Creatinine", "Platelets"]

    for patient in data:
        submitter_id = patient.get("submitter_id", "Unknown")
        follow_ups = patient.get("follow_ups", [])
        
        if len(follow_ups) < 2:
            continue
            
        # 1. Sort visits chronologically
        valid_visits = [f for f in follow_ups if "days_to_follow_up" in f]
        sorted_visits = sorted(valid_visits, key=lambda x: x["days_to_follow_up"])
        
        timeline_text = ""
        m_protein_history = []
        
        # 2. Extract and format the labs
        for visit in sorted_visits:
            day = visit["days_to_follow_up"]
            tests = visit.get("molecular_tests", [])
            
            extracted_labs = {}
            for test in tests:
                lab_name = test.get("laboratory_test")
                if lab_name in target_labs:
                    val = test.get("test_value")
                    unit = test.get("test_units", "")
                    if val is not None:
                        extracted_labs[lab_name] = f"{round(val, 2)} {unit}"
                        if lab_name == "M Protein": 
                            m_protein_history.append(val)
            
            if extracted_labs:
                labs_str = ", ".join([f"{k}: {v}" for k, v in extracted_labs.items()])
                timeline_text += f"- Day {day}: {labs_str}\n"

        if not timeline_text or len(m_protein_history) < 2:
            continue

        # 3. Formulate the Prompt
        user_prompt = (
            f"Review the following longitudinal biomarker history for patient {submitter_id}. "
            "Predict the disease trajectory: Is this patient showing biochemical progression toward Active Myeloma?\n"
            f"Timeline:\n{timeline_text.strip()}"
        )
        
        # 4. Calculate Velocity for Assistant Target
        # IMWG Criteria for Progressive Disease: 
        # >25% increase AND absolute increase of >= 0.5 g/dL
        # 4. Calculate Velocity for Assistant Target (n-visits logic)
        current_m_spike = m_protein_history[-1]
        
        # Find the lowest historical value (nadir) before the current visit
        nadir_m_spike = min(m_protein_history[:-1]) if len(m_protein_history) > 1 else m_protein_history[0]
        
        absolute_change = current_m_spike - nadir_m_spike
        relative_change = (absolute_change / nadir_m_spike) if nadir_m_spike > 0 else 0 
        
        # IMWG Criteria: >= 25% increase AND absolute increase of >= 0.5 g/dL from NADIR
        if absolute_change >= 0.5 and relative_change >= 0.25:
            assistant_target = "Yes. The velocity of the biomarkers indicates clinical disease progression according to IMWG criteria (>= 25% relative increase and >= 0.5 g/dL absolute increase in M-Protein from the nadir). Pre-emptive intervention evaluation is recommended."
        elif current_m_spike < nadir_m_spike:
            # If the current value is a new all-time low
            assistant_target = "No. The biomarkers show a continued favorable response. The M-Protein has reached a new nadir."
        else:
            assistant_target = "No immediate rapid progression detected. The M-Protein is currently stable. Continue routine longitudinal monitoring."
        
        conversational_pairs.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_target}
            ]
        })

    print(f"Successfully serialized {len(conversational_pairs)} patient timelines.")

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(conversational_pairs)
    
    # --- 3-WAY SPLIT: 70 Train / 10 Eval / 20 Test ---
    # Step 1: Split 70% for train, 30% for a temporary test_eval chunk
    split_1 = hf_dataset.train_test_split(test_size=0.30, shuffle=True, seed=42)
    train_ds = split_1["train"]
    temp_test_eval_ds = split_1["test"]
    
    # Step 2: Split the 30% chunk into 10% eval and 20% test 
    # (Since 20 is two-thirds of 30, we use test_size=20/30)
    split_2 = temp_test_eval_ds.train_test_split(test_size=(20/30), shuffle=True, seed=42)
    eval_ds = split_2["train"]
    test_ds = split_2["test"]
    
    # Reconstruct the final dictionary
    hf_dataset_dict = DatasetDict({
        "train": train_ds,
        "eval": eval_ds,
        "test": test_ds
    })

    print("\nDataset split successfully!")
    print(f"Train set size: {len(hf_dataset_dict['train'])} samples (approx 70%)")
    print(f"Eval set size:  {len(hf_dataset_dict['eval'])} samples (approx 10%)")
    print(f"Test set size:  {len(hf_dataset_dict['test'])} samples (approx 20%)")
          
    # --- SAVE TO DISK ---
    print(f"\nSaving dataset to Hugging Face native format in '{output_dir}/'...")
    hf_dataset_dict.save_to_disk(output_dir)

    # Save as JSONL for easy human reading/debugging
    jsonl_dir = f"{output_dir}_jsonl"
    os.makedirs(jsonl_dir, exist_ok=True)
    hf_dataset_dict["train"].to_json(f"{jsonl_dir}/train.jsonl")
    hf_dataset_dict["eval"].to_json(f"{jsonl_dir}/eval.jsonl")
    hf_dataset_dict["test"].to_json(f"{jsonl_dir}/test.jsonl")
    print(f"Saved readable JSONL copies to '{jsonl_dir}/'.")

    return hf_dataset_dict

# Run the builder and save the files
build_and_save_progression_dataset(r'E:\ajja\data\clinical.project-mmrf-commpass.2026-02-14.json', output_dir='module4_data')