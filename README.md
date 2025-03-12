# **SDoH Extraction with Ollama and NLTK**

This script extracts **Social Determinants of Health (SDoH) phrases** from a given text file using **Ollama's LLM** and **NLTK** for text tokenization. The extracted phrases are saved in a structured format.

## **Prerequisites**
Before running the script, ensure you have the following installed:

### **1. Install Python**
This script requires **Python 3.8+**. You can check your version using:

```sh
python --version
```

### **2. Install Required Dependencies**
Run the following command to install the necessary Python libraries:

```sh
pip install ollama nltk
```

### **3. Download NLTK Tokenizer**
The script uses `nltk.tokenize.sent_tokenize`, which requires the **punkt** tokenizer. You can download it manually:

```python
import nltk
nltk.download("punkt")
```

Alternatively, the script downloads it automatically.

---

## **Setup Instructions**

### **Step 1: Install and Set Up Ollama**
Ollama provides local execution for large language models.

1. **Download and Install Ollama**  
   Follow the instructions for your operating system at [Ollama’s website](https://ollama.ai).

2. **Pull the Required Model**  
   This script uses `deepseek-r1`. Download it by running:

   ```sh
   ollama pull deepseek-r1
   ```

---

## **How to Run the Script**

### **Step 1: Prepare the Input File**
Ensure you have a text file (`sdoh_text.txt`) in the same directory as the script. This file should contain the text from which you want to extract SDoH phrases.

### **Step 2: Run the Script**
Execute the script using:

```sh
python sdoh_extraction.py
```

---

## **How the Script Works**
1. **Reads the Input File:**  
   - Loads `sdoh_text.txt`  
   - If the file is missing, it throws an error.

2. **Splits Text into Chunks:**  
   - Uses `nltk.sent_tokenize` to split text into **sentence-based chunks** (max **2000 tokens per chunk**).

3. **Runs the Ollama LLM Extraction Model:**  
   - **Prompts the model** to extract SDoH phrases **directly from text** (without paraphrasing).  
   - **Filters results** to only show categories with matching text.

4. **Saves the Extracted Phrases:**  
   - Outputs **structured results** in a file: `sdoh_llm_raw_responses.txt`.

---

## **Expected Output Format**
Each extracted phrase is categorized based on SDoH attributes. The output file (`sdoh_llm_raw_responses.txt`) will contain results in the following format:

```json
{
  "Age": "47 years old",
  "Gender": "male",
  "Employment": "Works as a delivery driver",
  "Financial Strain": "Struggling to pay rent",
  "Housing Stability": "Facing eviction"
}
```
