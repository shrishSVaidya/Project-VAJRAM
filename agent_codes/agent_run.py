import os

import torch
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- NEW RAG IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import re

def exclude_thinking_component(text: str) -> str:
    """Removes the MedGemma thinking block from a string."""
    # Sometimes MedGemma forgets the closing tag, so we handle both cases
    clean_text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    clean_text = re.sub(r"<unused94>.*", "", clean_text, flags=re.DOTALL) 
    return clean_text.strip()


# ==========================================
# 0. ACTUAL RAG SETUP (FAISS + HuggingFace)
# ==========================================
print("Initializing Embedding Model...")
# Using a tiny, fast embedding model to save your Titan Xp's VRAM
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'} 
)

print("Building FAISS Vector Database...")
# In production, you would load these from a PDF using PyPDFLoader or similar.
# For this script, we are ingesting raw text documents directly into the DB.
guideline_docs = [
    Document(page_content="NCCN Multiple Myeloma Guidelines: For transplant-eligible patients with high-risk cytogenetics and normal renal function, standard VRd is recommended."),
    Document(page_content="NCCN Multiple Myeloma Guidelines: For transplant-eligible patients presenting with severe renal impairment (Creatinine > 2.0), VRd is still preferred, but Bortezomib requires careful dose adjustment (typically 1.3 mg/m2 on day 1, 4, 8, 11) to prevent toxicity."),
    Document(page_content="ESMO Multiple Myeloma Guidelines: Daratumumab plus VTd is a category 1 recommendation for transplant-eligible patients regardless of renal status."),
    Document(page_content="Acute Lymphoblastic Leukemia Guidelines: First-line induction for Philadelphia chromosome-positive ALL includes targeted TKIs like Imatinib combined with Hyper-CVAD.")
]

# Create the vector store and the retriever object
vector_db = FAISS.from_documents(guideline_docs, embeddings)
# k=2 means it will fetch the top 2 most medically relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 2}) 

from peft import PeftModel

from transformers import BitsAndBytesConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. LOAD BASE MODEL & LORA ADAPTERS
# ==========================================
print("Loading Base MedGemma 1.5 in 8-bit mode...")
model_id = "google/medgemma-1.5-4b-it"

# 1. Use Processor instead of Tokenizer
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer
tokenizer.padding_side = 'right' 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Use 4-bit config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # Safe for Titan Xp compute
)

# 3. Use AutoModelForImageTextToText
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16 # Mandatory for Titan Xp
)

TESTING_MODE = True # Set to False when you actually have your LoRA folders ready

if TESTING_MODE:
    print("Testing Mode: No LoRAs loaded. Using base model for all nodes.")
    model = base_model 
else:
    print("Loading LoRA Adapters into memory...")
    model = PeftModel.from_pretrained(base_model, "models/lora_module2/checkpoint-1506", adapter_name="module2")
    model.load_adapter("models/lora_module3/checkpoint-212", adapter_name="module3")
    model.load_adapter("models/lora_module4/checkpoint-201", adapter_name="module4")


# ==========================================
# 2. DEFINE THE INFERENCE HELPER
# ==========================================
def generate_with_adapter(prompt: str, adapter_name: str, max_tokens: int = 150) -> str:
    """Hot-swaps the LoRA adapter, generates text using the Processor, and safely swaps back."""
    
    # ---Apply MedGemma's specific chat template via the Processor ---
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize and move to GPU
    print("Input to model = ", formatted_prompt)
    inputs = processor(text=formatted_prompt, return_tensors="pt", padding=True).to(base_model.device)
    
    # Drop vision-specific keys since we are only doing text
    inputs.pop("token_type_ids", None)
    inputs.pop("pixel_values", None)
    
    print(f"      [LLM Generation Started - Max Tokens: {max_tokens}]")
    
    if adapter_name == "default" or TESTING_MODE:
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter(): 
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens, 
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
        else:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
                
    else:
        model.set_adapter(adapter_name)
        print(f"      [Active LoRA swapped to: {adapter_name}]")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )

    print("      [LLM Generation Complete]")
    
    # --- FIX: Decode only the newly generated tokens using the processor ---
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0, input_length:]
    response = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    
    clean_response = exclude_thinking_component(response)
    
    print(f"      [Cleaned LLM Output]: {response}")
    return clean_response

# ==========================================
# 3. DEFINE LANGGRAPH STATE
# ==========================================
class PatientState(TypedDict):
    patient_id: str
    raw_clinical_text: str
    module2_risk_score: str
    module3_wsi_analysis: str
    module4_progression: str
    module5_guidelines: str  # Holds the ACTUAL retrieved documents
    final_recommendation: str

# ==========================================
# 4. DEFINE LANGGRAPH NODES
# ==========================================
def run_module2_node(state: PatientState):
    print("\n--- NODE: Module 2 (Risk Assessment) ---")
    prompt = f"Assess the Multiple Myeloma risk profile OUTPUT ONE OF THESE: Standard, Low risk, High risk\n based on this clinical data:\n{state['raw_clinical_text']}\nConcise Risk Profile:"
    # FIX: Bumped max_tokens to 400 so it can finish thinking
    return {"module2_risk_score": generate_with_adapter(prompt, adapter_name="module2", max_tokens=200)}

def run_module3_node(state: PatientState):
    print("\n--- NODE: Module 3 (Bone Marrow WSI) ---")
    prompt = f"Provide a 1-sentence summary of the abnormal plasma cell infiltration (WSI) from this data:\n{state['raw_clinical_text']}\nWSI Summary:"
    # FIX: Bumped max_tokens to 300
    return {"module3_wsi_analysis": generate_with_adapter(prompt, adapter_name="module3", max_tokens=300)}

def run_module4_node(state: PatientState):
    print("\n--- NODE: Module 4 (Progression Tracking) ---")
    prompt = f"Identify the M-Spike progression metrics from this data and state if it indicates rapid progression:\n{state['raw_clinical_text']}\nProgression Summary:"
    # FIX: Bumped max_tokens to 300
    return {"module4_progression": generate_with_adapter(prompt, adapter_name="module4", max_tokens=300)}

def run_module5_rag_node(state: PatientState):
    print("\n--- NODE: Module 5 (Actual RAG Retrieval) ---")
    
    # FIX: Feed the raw clinical text directly to the RAG node so it can see the Creatinine levels itself
    prompt = f"""
    Formulate a 4-to-8 word search query to look up NCCN treatment guidelines for this patient.
    Clinical Data: {state['raw_clinical_text']}
    
    Output strictly the search query and nothing else.
    Search Query String:
    """
    search_query = generate_with_adapter(prompt, adapter_name="default", max_tokens=100)
    clean_query = exclude_thinking_component(search_query).replace('"', '').replace('\n', ' ')
    
    print(f"    [Executing Vector Search for: '{clean_query}']")
    retrieved_docs = retriever.invoke(clean_query)
    formatted_context = "\n\n".join([f"Source Document:\n{doc.page_content}" for doc in retrieved_docs])
    
    return {"module5_guidelines": formatted_context}

def orchestrator_synthesis_node(state: PatientState):
    print("\n--- NODE: Orchestrator Synthesis ---")
    
    # FIX: Added a strict directive to use the context provided.
    prompt = f"""
    You are the Master Hematology Orchestrator. Provide a clear, final treatment recommendation based strictly on the retrieved guidelines and patient profile. Do not output your internal thought process.
    
    [PATIENT PROFILE]
    - Clinical Context: {state['raw_clinical_text']}
    - Risk Score: {state['module2_risk_score']}
    - Bone Marrow: {state['module3_wsi_analysis']}
    - Progression: {state['module4_progression']}
    
    [RETRIEVED GUIDELINES (Module 5 RAG)]
    {state['module5_guidelines']}
    
    Final Treatment Recommendation:
    """
    
    result = generate_with_adapter(prompt, adapter_name="default", max_tokens=800)
    return {"final_recommendation": result}

# ==========================================
# 5. BUILD AND COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(PatientState)

workflow.add_node("module2", run_module2_node)
workflow.add_node("module3", run_module3_node)
workflow.add_node("module4", run_module4_node)
workflow.add_node("module5_rag", run_module5_rag_node)
workflow.add_node("orchestrator", orchestrator_synthesis_node)

workflow.set_entry_point("module2")
workflow.add_edge("module2", "module3")
workflow.add_edge("module3", "module4")
workflow.add_edge("module4", "module5_rag")     # Module 4 flows into RAG
workflow.add_edge("module5_rag", "orchestrator") # RAG flows into Orchestrator
workflow.add_edge("orchestrator", END)

full_agent = workflow.compile()

# ==========================================
# 6. RUN THE AGENT
# ==========================================
if __name__ == "__main__":
    print("\n\n=== STARTING MULTI-LORA + ACTUAL RAG INFERENCE ===")
    
    # --- FIX: Provided a complete, rich clinical text so the LoRAs trigger properly ---
    initial_state = {
        "patient_id": "MMRF_2240",
        "raw_clinical_text": """
        65-year-old Male patient. Transplant eligible. 
        - CRAB Panel: Creatinine: 2.5 mg/dL (Severe Renal Impairment), Calcium: 10.5 mg/dL, Hemoglobin: 9.0 g/dL. 
        - Tumor/Staging Panel: Beta-2 Microglobulin: 5.8 mg/L, LDH: 280 U/L. 
        - WSI Pathology: Bone marrow biopsy reveals 45% abnormal plasma cell infiltration.
        - Behaviour: Feeling Dizziness with Chest Pain and Mental Confusion.
        """,
    }
    
    final_state = full_agent.invoke(initial_state)
    
    print("\n\n=== FINAL ORCHESTRATOR OUTPUT ===")
    print(final_state["final_recommendation"])