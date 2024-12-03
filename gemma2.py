import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def answer_trivia(
    input_file, 
    output_file, 
    model_name="INSAIT-Institute/BgGPT-Gemma-2-27B-IT-v1.0",
    batch_size=10,
    max_new_tokens=179,
    temperature=0.1,
    top_k=25,
    top_p=1.0,
    repetition_penalty=1.1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False
    )
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=[1, 107]
    )
    
    prompt_template = """<bos><start_of_turn>user
Your job is to look at a trivia question and answer it concisely. We are only interested in the answer and not any additional information around it.
First, I will give you an example of a question and its answer and then you will answer a new question.
The following is an example of a question and a correct answer:
'''
Question: What is the capital of Bulgaria?
Ideal Answer: Sofia
'''
This answer is ideal because it is concise and answers the question correctly without any additional information.
The format of the answer should be as follows:
"Sofia" - delimit the answer with double quotes

Now, answer the following question:
Question: {question}<end_of_turn>
<start_of_turn>model
"""
    
    df = pd.read_csv(input_file)
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_prompts = [prompt_template.format(question=q) for q in batch['Question']]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        
        outputs = model.generate(**inputs, generation_config=generation_config)
        
        batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
   
        clean_answers = [
            ans.split('<end_of_turn>')[1].strip() if '<end_of_turn>' in ans else ans.strip()
            for ans in batch_answers
        ]
        
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
        input_file='trivia_questions.csv', 
        output_file='trivia_answers.csv',
        batch_size=10,
        temperature=0.1,
        max_new_tokens=179
    )
