import pandas as pd
import re

answers = pd.read_csv("trivia_answers_bgGPT.csv")
for answer in answers["Model Answer"]:
    start = answer.find('[INST]')
    end = answer.find('[/INST]')
    if start != -1 and end != -1:
        new_answer = answer[:start] + answer[end + len('[/INST]'):]
        answers.loc[answers["Model Answer"] == answer, "Model Answer"] = new_answer.strip()

answers.to_csv("trivia_answers_bgGPT_corrected.csv", index=False)