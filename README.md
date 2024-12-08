# CS5063 Evaluation of AI Systems Group Project Repository Structure

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
