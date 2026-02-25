import os

os.environ['HF_HOME'] = '/media/shrish/Data/medgemma_finetune/hf_models/'

import gradio as gr
# Import the compiled LangGraph agent from your script
from agent_run import full_agent

def process_patient_data(clinical_text):
    """
    Feeds the input text to the LangGraph agent and extracts the intermediate 
    LoRA states and final orchestrator output for the UI.
    """
    initial_state = {
        "patient_id": "HF_Space_Demo",
        "raw_clinical_text": clinical_text,
    }
    
    # Run the agent
    final_state = full_agent.invoke(initial_state)
    
    # Extract the data from the state dictionary
    risk_score = final_state.get("module2_risk_score", "No data generated.")
    wsi_analysis = final_state.get("module3_wsi_analysis", "No data generated.")
    progression = final_state.get("module4_progression", "No data generated.")
    rag_context = final_state.get("module5_guidelines", "No guidelines retrieved.")
    final_rec = final_state.get("final_recommendation", "No recommendation generated.")
    
    return risk_score, wsi_analysis, progression, rag_context, final_rec

# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 Project VAJRAM")
    gr.Markdown("A Multi-LoRA + RAG Agent powered by MedGemma 1.5. Enter patient clinical data below to trigger specialized diagnostic modules and synthesize a treatment plan.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=10, 
                label="Patient Clinical Profile", 
                placeholder="Enter patient age, CRAB panel, Tumor/Staging panel, WSI findings, and history..."
            )
            submit_btn = gr.Button("Analyze Patient", variant="primary")
            
            gr.Examples(
                examples=[
                    "65-year-old Male. Transplant eligible. CRAB: Creatinine 2.5 mg/dL (Severe Renal Impairment), Calcium 10.5, Hb 9.0. Beta-2 Microglobulin: 5.8 mg/L. WSI reveals 45% abnormal plasma cells. M-Spike increased 1.2 g/dL over 3 months."
                ],
                inputs=input_text
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 LoRA Module Outputs")
            out_risk = gr.Textbox(label="Module 2: Risk Profile", interactive=False)
            out_wsi = gr.Textbox(label="Module 3: Bone Marrow WSI", interactive=False)
            out_prog = gr.Textbox(label="Module 4: Progression", interactive=False)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📚 FAISS RAG Retrieval")
            out_rag = gr.Textbox(label="Module 5: Retrieved Guidelines", lines=4, interactive=False)
            
            gr.Markdown("### ⚖️ Master Orchestrator Synthesis")
            out_final = gr.Textbox(label="Final Treatment Recommendation", lines=8, interactive=False)

    # Wire up the button to the function
    submit_btn.click(
        fn=process_patient_data,
        inputs=[input_text],
        outputs=[out_risk, out_wsi, out_prog, out_rag, out_final]
    )

if __name__ == "__main__":
    demo.launch()