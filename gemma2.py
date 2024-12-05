import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.nn as nn

def answer_trivia(
    input_file,
    output_file,
    model_name="INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0",
    batch_size=10,
    max_new_tokens=256,
    temperature=0.1,
    top_k=25,
    top_p=1.0,
    repetition_penalty=1.1
):

    device = torch.device("cpu")# if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True
    )

 
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False
    )

    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=[1, 107]
    )

    # Prompt template
    prompt_template = """<bos><start_of_turn>user
        Твоята зада1а е да отговаряш възможно най-кратко на даденен въпрос. Отговорът, който даваш, трябва да бъде точен, кратък и да съдържа само информация, която отговаря на зададения въпрос. Например:
        Въпрос: Коя е столицата на България?
        Отговор: "София"
        Сега отговори на следния въпрос:
        Question: {question}<end_of_turn>
        <start_of_turn>model
        """

    df = pd.read_csv(input_file)
    results = []

    def clean_model_response(response):
        if "model" in response:
            parts = response.split("model")
            answer = parts[-1].strip()
            return answer
        return response.strip()

    # Update the processing section:
    for i in range(0, len(df), batch_size):
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
        input_file='trivia_qa_bulgarian.csv',
        output_file='trivia_answers.csv',
        batch_size=2,
        temperature=0.1,
        max_new_tokens=256
    )
