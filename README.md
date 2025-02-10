# **LLM-Assisted NLP Pipeline for SDoH-CVD Analysis**

## **Project Overview**
This project aims to develop a **Large Language Model (LLM)-assisted NLP pipeline** for **Social Determinants of Health (SDoH) annotation and Named Entity Recognition (NER) in Cardiovascular Disease (CVD)** data. The pipeline leverages **Ollama** to run models locally, ensuring efficient and customizable annotation workflows. 

## **Features**
- **Local LLM Execution**: Uses **Ollama** to run LLMs efficiently on local machines.
- **SDoH Annotation**: Extracts **eight key SDoH categories** from clinical text.
- **Named Entity Recognition (NER)**: Identifies relevant entities to support structured data extraction.
- **Flexible Prompt Engineering**: Supports zero-shot and few-shot learning for annotation experiments.
- **Ontology Development**: Aligns extracted entities with an expanding SDoH-CVD ontology.

## **Installation**
### **Prerequisites**
- **Python 3.8+**
- **Ollama** installed on your machine ([installation guide](https://ollama.com/))
- GPU recommended for faster model inference

### **Setup**
1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   Or follow the [official guide](https://ollama.com/)

2. **Download the required LLM model** (e.g., LLaMA 3.3 or another supported model):
   ```bash
   ollama pull llama3
   ```

3. **Clone the project repository**:
   ```bash
   .......
   ```

4. **Install dependencies**:
   ```bash
   .......
   ```

## **Usage**
### **Running the Chatbot for SDoH Annotation**
1. **Start the Ollama server**:
   ```bash
   ollama serve
   ```
2. **Run the annotation script**:
   ```bash
   python annotate.py --model llama3
   ```

### **Running Experiments**
Four experimental setups for **zero-shot and few-shot** learning:
