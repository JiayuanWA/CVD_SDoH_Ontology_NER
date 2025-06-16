# **SDoH Extraction with OpenAI GPT and NLTK**

This script extracts Social Determinants of Health (SDoH) phrases from clinical or narrative text using OpenAI GPT and rule-based evaluation.

---

## **Prerequisites**

### **1. Install Python**
This script requires Python 3.8+. Check your version using:

```sh
python --version
```

### **2. Install Required Dependencies**
Run the following command to install all required libraries:

```sh
pip install openai nltk
```

---

## **How to Run the Script**

### **Step 1: Prepare Your Input Files**
Ensure you have the following files in the same directory:

- `sdoh_text.txt` – your source text
- `examples_xml/` – few-shot XML examples for in-context prompting
- `gold_sdoh.xml` – gold standard annotations (optional, for evaluation)

### **Step 2: Run the Script**
Execute the extraction script using:

```sh
python sdoh_GPT.py
```

---

## **Methodology**

This pipeline performs SDoH phrase extraction and evaluation in five main stages:

### 1. **Batch-Based Prompting**
- Categories are grouped into 5 batches:
  - **Demographic**
  - **Substance Use**
  - **Psychosocial + Lifestyle**
  - **Economic Stability + Access**
  - **Community + Built Environment**
- Each batch contains 4–10 related SDoH categories to avoid overwhelming the prompt.

### 2. **Few-Shot Prompt Construction**
- XML files in the `examples_xml/` folder are parsed to generate input-output demonstrations.
- These are prepended to GPT prompts for better accuracy.

### 3. **GPT Response Parsing (`parse_response`)**
- GPT is prompted batch-by-batch to extract phrases for relevant SDoH categories. To maximize precision of span matching, each extracted phrase is first deduplicated — only unique phrases per category are retained in the first pass.

### 4. **Span Parsing and Matching, XML Generation**
- In the parse_response() function:
  - Each phrase is matched against all possible spans in the full text.
  - The best non-overlapping span is selected per phrase, ensuring accurate start/end offsets.
  - Although deduplication occurs initially, all distinct occurrences of the phrase are matched later to ensure no information is lost.
  - Extracted results are formatted as XML with span positions, phrase text, and unique tag IDs.

### 5. **Evaluation (Exact and Partial Matching)**
- Predictions are compared to gold-standard annotations using:
  - **Exact match** = same span and category
  - **Partial match** = overlapping span and correct category
  - **Wrong type** = span overlaps, category incorrect
  - **False positive** = predicted but not in gold
  - **False negative** = gold tag missed entirely

Metrics include precision, recall, and F1 score, both strictly (exact match) and relaxed (including partial).

---

## **Expected Output Formats**

### **1. Raw GPT Response**
Appended to: `overall_output.txt`

### **2. Structured XML Output**
Saved to: `sdoh_extracted.xml`

Example:

```xml
<Age spans="25~37" text="47 years old" id="A0" comment=""/>
```

---

## **Supporting Files**

| File Name              | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `sdoh_text.txt`        | Input text for SDoH extraction                                          |
| `examples_xml/`        | XML-formatted few-shot prompts                                          |
| `sdoh_extracted.xml`   | Output file with predicted tags and spans                               |
| `gold_sdoh.xml`        | Gold standard annotations for evaluation                                |

---

## **Evaluation Details**

The script evaluates GPT predictions against gold annotations using character spans and type labels.

### Matching Rules

- **True Positive (Exact):** Span + Category match
- **Partial Match:** Overlapping span with correct category
- **Wrong Type:** Span overlaps but wrong category
- **False Positive:** Prediction not found in gold
- **False Negative:** Gold tag missing from predictions

### Metrics Reported

- **Precision:**  
  `TP / (TP + FP)`
- **Recall:**  
  `TP / (TP + FN)`
- **F1 Score:**  
  `2 * (Precision * Recall) / (Precision + Recall)`
- **Per-category Breakdown:**  
  Separate metrics for each SDoH class (exact + partial)

### Sample Output

```
--- Evaluation (Strict Exact Match Only) ---
True Positives (exact span + correct type):      12
Type Mismatch (span matched, wrong type):         1
False Positives (no overlap with gold):           2
False Negatives (gold missed entirely):           4
Precision (exact only):                           0.750
Recall (exact only):                              0.600
F1 Score (exact only):                            0.667

--- Evaluation (Relaxed: Exact + Partial Match) ---
True Positives (exact + partial span + correct type): 15
Partial Matches (span overlap + correct type):         3
Precision (combined):                                0.833
Recall (combined):                                   0.750
F1 Score (combined):                                 0.789
```
