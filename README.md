# **SDoH Extraction with Ollama and NLTK**

This script extracts Social Determinants of Health (SDoH) phrases from a given text file.

## **Prerequisites**
Before running the script, ensure you have the following installed:

### **1. Install Python**
This script requires Python 3.8+. Check version :

```sh
python --version
```

### **2. Install Required Dependencies**
Run the following command to install the necessary Python libraries:

```sh
pip install ollama nltk
```

---

## **Setup Instructions**

### **Step 1: Install and Set Up Ollama**
Ollama provides local execution for large language models.

1. **Download and Install Ollama**  
   Follow the instructions for your operating system at [Ollama’s website](https://ollama.ai).

2. **Pull the Required Model**  
   This script supports multiple models. You can use either `deepseek-r1` or `llama3`. Pull the desired model using:

   ```sh
   ollama pull deepseek-r1
   ```

   or

   ```sh
   ollama pull llama3
   ```




---

## **How to Run the Script**

### **Step 1: Prepare the Input File**
Ensure you have a text file (`sdoh_text.txt`) in the same directory as the script. This file should contain the text from which you want to extract SDoH phrases.

### **Step 2: Run the Script**
Execute the script with the desired model using:

```sh
python sdoh_ner.py
```

or

```sh
python sdoh_llama.py
```

Alternatively, for OpenAI API-based extraction, use:

```sh
python sdoh_GPT.py
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

The output will look like:

```json
{
  "Age": "47 years old",
  "Gender": "male",
  "Employment": "Works as a delivery driver",
  "Financial Strain": "Struggling to pay rent",
  "Housing Stability": "Facing eviction"
}
```

For XML format:

```xml
<Age spans="25~37" text="47 years old" id="A0" comment=""/>
```

---

## **Supporting Files**

| File Name              | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `sdoh_text.txt`        | Input text for SDoH extraction.                                         |
| `examples_xml/`        | Folder containing XML examples for few-shot prompting.                 |
| `sdoh_extracted.xml`   | Output XML with extracted entities and spans.                          |
| `gold_sdoh.xml`        | Gold standard annotation used for evaluation.                          |

---

## **Evaluation **

After extraction, the script compares GPT's output with a gold standard (`gold_sdoh.xml`) using character-level span matching and category labels.

### Matching Rules

Each predicted phrase is matched against gold standard phrases:

- **True Positive (Exact Match):**  
  Phrase matches exactly **and** category is correct.

- **Wrong Type:**  
  Phrase matches, but the category is incorrect.

- **Partial Match:**  
  Phrase overlaps but is not an exact match.
  Either the model got part of the phrase, or included too much, but the category may or may not be correct.

- **False Positive:**  
  Model predicted a phrase that doesn’t exist in the gold file.

- **False Negative:**  
  Model missed a phrase that exists in the gold file.

### Metrics Reported

- **Precision:** What % of model predictions were correct? True Positives / (True Positives + False Positives)
- **Recall:** What % of gold annotations did the model find? True Positives / (True Positives + False Negatives)
- **F1 Score:** Balance between precision and recall. 2 * (Precision * Recall) / (Precision + Recall)
- **Per-category breakdown:** Accuracy for each SDoH type.

### Sample Output

```
--- Evaluation (Strict Exact Match) ---
True Positives:       12
Partial Matches:       3
Wrong Category:        1
False Positives:       2
False Negatives:       4
Precision:          0.750
Recall:             0.600
F1 Score:           0.667
```


