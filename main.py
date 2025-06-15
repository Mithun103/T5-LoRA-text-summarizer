import gradio as gr
import torch  
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = "./full_lora_summarizer"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def summarize(text):
    input_ids = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, padding=True, max_length=2048).input_ids
    output_idse = model.generate(
    input_ids,
    do_sample=True,
   
    top_p=0.95,
    max_length=500,
)
    with torch.no_grad():
        output_ids = output_idse
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

# Gradio interface
interface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, label="Enter Article/News/Content"),
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="📝 Summarization with LoRA + T5",
    description="Powered by HuggingFace Transformers + PEFT (LoRA). Input a long text and get the summarized output instantly."
)

interface.launch(share=True)  # share=True to get a public URL
