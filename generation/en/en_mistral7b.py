import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.nn as nn
access_token = "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"
def answer_trivia(
    input_file,
    output_file,
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    file_name="mistral-7b-instruct-v0.2.Q5_K_S.gguf",
    batch_size=10,
    max_new_tokens=256,
    temperature=0.1,
    top_k=25,
    top_p=1.0,
    repetition_penalty=1.1
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")  
    print("other print")  

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        gguf_file=file_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        token = access_token,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    print("model loaded")

 
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        gguf_file=file_name,
        use_default_system_prompt=False,
        token = access_token
    )

    print("tokenizer loaded")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id
        
    )

    print("generation config loaded")

    # Prompt template
    prompt_template = """<s>[INST] <<SYS>>
       Your task is to answer the given question as concisely as possible.The given answer needs to be precise, concise and contain the information that the question is asking for. For example:
        Question: "What is the capital of Bulgaria?"
        Answer: "Sophia"
        <</SYS>>

        Question: {question} [/INST]"""

    # prompt_template = """<s>[INST] <<SYS>>
    #     Twoim zadaniem jest odpowiadać na podane pytania możliwie najkrócej. Odpowiedź, którą podajesz, musi być dokładna, zwięzła i zawierać tylko informacje odpowiadające na zadane pytanie. Na przykład:
    #     Pytanie: Jaka jest stolica Bułgarii?
    #     Odpowiedź: "Sofia"
    #     <</SYS>>

    #     Pytanie: {question} [/INST]"""

    df = pd.read_csv(input_file)
    results = []

    print('df head: ', df.head())

    def clean_model_response(response):
        if "model" in response:
            parts = response.split("model")
            answer = parts[-1].strip()
            return answer
        return response.strip()

    # Update the processing section:
    for i in range(0, len(df), batch_size):
        print("started ", i)
        batch = df.iloc[i:i+batch_size]
        batch_prompts = [prompt_template.format(question=q) for q in batch['Question']]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, generation_config=generation_config)
        batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        clean_answers = [clean_model_response(ans) for ans in batch_answers]
        
        for q, a in zip(batch['Question'], clean_answers):
            results.append({
                'Question': q,
                'Model Answer': a
            })
        print(f"Processed {i + len(batch)}/{len(df)} questions")

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    answer_trivia(
        input_file='../../data/input/trivia_qa_chosen.csv',
        output_file='../../data/output/en/trivia_answers_english_mistral.csv',
        batch_size=2,
        temperature=0.1,
        max_new_tokens=256
    )