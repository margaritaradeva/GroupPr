# CS5063 Evaluation of AI Systems Group Project 
## Research Project
The project focuses on evaluating 6 large language models (LLMs) across different languages, specifically Polish, Bulgarian, and English and comparing their hallucination levels.
## Models Compared
### Bulgarian Models
- **BGGPT (1st Version)**: The initial iteration of the Bulgarian language model.
- **BGGPT (2nd Version)**: An enhanced version based on the Gemma2 architecture, incorporating advanced features for improved performance.

### Polish Models
- **Bielik**: A model built on the Mistral architecture, specifically optimized for Polish language tasks.
- **TRURL**: TRURL chatbot is the first Polish language model similar to ChatGPT, which VoiceLab.AI, based on LLAMa2 architecture created.

### English Models
- **Mistral**: With top-tier reasoning capabilities and excels in advanced reasoning, multilingual tasks, math, and code generation. 
- **Gemma2**: A successor to the previous Gemma models with multilanguage capacities, designed to excel in multilingual tasks.

## Pipeline 
![Pipeline Diagram](https://github.com/margaritaradeva/GroupPr/blob/main/data/figures/pipeline.png)
## Repository Structure

### 1. Data
- **`data/input/`**: Contains the source files for questions in both Polish and Bulgarian.
- **`data/output/`**: Organized model responses for each language:
  - **`bg/`**: Bulgarian model responses.
  - **`pl/`**: Polish model responses.
  - **`en/`**: English model responses.
- **`data/output_evaluation/`**: Stores evaluation results and performance metrics.

---

### 2. Core Components

#### Generation Scripts (`generation/`) for MacLaod/ Local Machine Testing 
- **Bulgarian models (`bg/`)**: Scripts for running Bulgarian model evaluations
- **Polish models (`pl/`)**: Scripts for executing Polish model evaluations  
- **English models (`en/`)**: Scripts for executing English model evaluations  

---

### 3. Evaluation Notebooks (`evaluation/`)
- **Language-specific analysis**:
  - `simpleQA_bulgarian.ipynb`: Evaluation and analysis of Bulgarian models.
  - `simpleQA_polish.ipynb`: Evaluation and analysis of Polish models.
  - `simpleQA_english.ipynb`: Evaluation and analysis of English models.
- **Results visualization**:
  - `Visualization.ipynb`: Notebook for creating visual summaries for the initial data
  - `classifications.ipynb`: Notebook for creating visual summaries for the results of SimpleQA.

---
### 4. Hallucination metric
We employed as part of our study, a metric recently developed by OpenAI, SimpleQA, and also translated it into Bulgarian and Polish to work with the 4 fine-tuned Slavic models.
---
### 5. Results 
![Results](https://github.com/margaritaradeva/GroupPr/blob/main/plots/results.png)
