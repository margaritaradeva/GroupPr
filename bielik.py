import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def answer_trivia(
    input_file,
    output_file,
    model_name="speakleash/Bielik-11B-v2.3-Instruct-GPTQ",
    batch_size=10,
    max_new_tokens=256,
    temperature=0.1,
    top_k=25,
    top_p=1.0,
    repetition_penalty=1.1
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False
    )

    # Add padding token if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id
    )

    # Simplified prompt template
    prompt_template = """Twoim zadaniem jest spojrzeć na pytanie z zakresu ciekawostek i odpowiedzieć na nie zwięźle. Interesuje nas wyłącznie odpowiedź, bez żadnych dodatkowych informacji wokół niej.
Najpierw podam przykład pytania i odpowiedzi, a następnie odpowiesz na nowe pytanie.
Oto przykład pytania i prawidłowej odpowiedzi:
‘‘‘
Pytanie: Jaka jest stolica Bułgarii?
Idealna odpowiedź: Sofia
‘‘‘
Ta odpowiedź jest idealna, ponieważ jest zwięzła i poprawnie odpowiada na pytanie, bez żadnych dodatkowych informacji.
Format odpowiedzi powinien wyglądać następująco:
„Sofia” - otocz odpowiedź podwójnymi cudzysłowami."""

    df = pd.read_csv(input_file)
    results = []

    def clean_model_response(response):
        # Remove any stray characters or text beyond the expected answer
        return response.strip().split("\n")[0].strip().replace('"', '')

    # Process in batches
    for i in range(0, len(df), batch_size):
        print(f"Processing batch starting at index {i}")
        batch = df.iloc[i:i+batch_size]
        batch_prompts = [prompt_template.format(question=q) for q in batch['Question']]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, generation_config=generation_config)
        batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        clean_answers = [clean_model_response(ans) for ans in batch_answers]
        print("Batch Prompts:", batch_prompts[:5])
        for q, a in zip(batch['Question'], clean_answers):
            results.append({
                'Question': q,
                'Model Answer': a
            })
        print(f"Processed {i + len(batch)}/{len(df)} questions")

        # Save results to file
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    answer_trivia(
        input_file='trivia_qa_polish.csv',
        output_file='polish_answers_bielik.csv',
        batch_size=2,
        temperature=0.1,
        max_new_tokens=256
    )