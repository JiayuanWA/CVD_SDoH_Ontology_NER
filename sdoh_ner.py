import ollama
import json

# ✅ Load input text from a file
def read_text_file(file_path):
    """Reads a text file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return None

# ✅ Path to your SDoH text file
file_path = "sdoh_text.txt"
text = read_text_file(file_path)

if text:
    print("\n📌 Running Llama 3 SDoH NER Extraction...\n")

    # 🔹 Llama 3 Prompt for Structured SDoH Extraction 🔹
    prompt = f"""
    You are an expert in Social Determinants of Health (SDoH) and healthcare NLP.
    
    Given the following text, extract key SDoH-related terms and classify them into these categories:
    - ECONOMIC (e.g., income, employment, financial strain)
    - HOUSING (e.g., homelessness, rent issues)
    - HEALTHCARE (e.g., insurance, medical access)
    - EDUCATION (e.g., literacy, school access)
    - SOCIAL (e.g., support, community, discrimination)

    **Text:**
    {text}

    Return only a structured JSON list like this:

    [
      {{"term": "financial instability", "category": "ECONOMIC"}},
      {{"term": "homelessness", "category": "HOUSING"}},
      {{"term": "lack of insurance", "category": "HEALTHCARE"}}
    ]
    """

    # ✅ Run the model with Ollama
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])

    # ✅ Extract the response
    generated_text = response['message']['content']

    # ✅ Debugging: Print raw Llama 3 response
    print("\n🛠️ **RAW MODEL RESPONSE:**\n")
    print(generated_text)

    # ✅ Extract JSON portion only
    try:
        response_json = generated_text.split("[", 1)[1].rsplit("]", 1)[0]  # Extract JSON content
        response_json = "[" + response_json + "]"  # Ensure valid JSON
        sdoh_entities = json.loads(response_json)  # Convert to Python list
    except Exception as e:
        print(f"⚠️ Error parsing JSON output: {e}")
        sdoh_entities = []

    # ✅ Print extracted SDoH entities
    print("\n📊 **Extracted SDoH Entities:**\n")
    print(json.dumps(sdoh_entities, indent=2))

    # ✅ Save results to a file
    output_file = "sdoh_llm_results.json"
    if sdoh_entities:
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(sdoh_entities, out, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
    else:
        print("⚠️ No meaningful SDoH terms detected. File not saved.")
else:
    print("⚠️ No text to process. Please check your input file.")
