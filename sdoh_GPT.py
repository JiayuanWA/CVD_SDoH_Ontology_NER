import openai
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download("punkt")  

DEBUG = False

API_KEY = "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"

client = openai.OpenAI(api_key=API_KEY)

file_path = "sdoh_text.txt"
output_file = "sdoh_extracted.xml"

# Define category-to-tag mapping
CATEGORY_TO_XML_TAG = {
    "Age": "Demographics", "Origin": "Demographics", "Ethnicity": "Demographics", "Gender": "Demographics",
    "Marital Status": "Demographics", "Alcohol": "Substance_Use", "Tobacco": "Substance_Use",
    "Illicit Drugs": "Substance_Use", "Caffeine": "Substance_Use", "Stress": "Psychosocial",
    "Psychological Concerns": "Psychosocial", "Social Adversities": "Social_Adversity", "Nutrition": "Health_Behaviors",
    "Physical Activity": "Health_Behaviors", "Sleep": "Health_Behaviors", "Sexual Behavior": "Health_Behaviors",
    "Contraception": "Health_Behaviors", "Treatment Adherence": "Health_Behaviors", "Employment": "Economic_Status",
    "Financial Strain": "Economic_Status", "Food Insecurity": "Food_Security", "Housing Stability": "Housing",
    "Caregiving": "Caregiving", "Living Situation": "Housing", "Social Support": "Social_Connections",
    "Safety": "Safety_and_Environment", "Transportation": "Access", "Utilities": "Access",
    "Medical Access": "Access", "Insurance": "Insurance", "Healthcare Technology": "Healthcare_Tech",
    "Erratic Care": "Erratic_Care", "Maintaining Care": "Maintaining_Care"
}

def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def chunk_text(text, max_tokens=2048):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def chat_with_gpt(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return None

def parse_response(raw_text):
    pattern = r'^(.*?):\s*"(.*?)",\s*Start:\s*(\d+),\s*End:\s*(\d+)'
    results = []
    for line in raw_text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            category, phrase, start, end = match.groups()
            results.append({
                "category": category.strip(),
                "phrase": phrase,
                "start": int(start),
                "end": int(end)
            })
    return results

def generate_xml(text, extractions):
    root = ET.Element("CVD_SDOH")
    text_element = ET.SubElement(root, "TEXT")
    text_element.text = f"<![CDATA[{text}]]>"

    tags = ET.SubElement(root, "TAGS")
    id_counter = {}

    for item in extractions:
        tag_type = CATEGORY_TO_XML_TAG.get(item["category"], "Other")
        base_id = tag_type[:2]
        tag_id_num = id_counter.get(tag_type, 0)
        tag_id = f"{base_id}{tag_id_num}"
        id_counter[tag_type] = tag_id_num + 1

        tag = ET.SubElement(tags, tag_type)
        tag.set("spans", f"{item['start']}~{item['end']}")
        tag.set("text", item["phrase"])
        tag.set("id", tag_id)
        tag.set("comment", "")

    ET.SubElement(root, "META")
    return ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")

# Main Pipeline
text = read_text_file(file_path)

if text:
    text_chunks = chunk_text(text, max_tokens=2000)
    all_extractions = []

    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)}...")

        prompt = f"""
You are an extraction model. Your task is to extract phrases related to 34 SDoH categories.

### Instructions:
- For each category match, return:
  - The exact phrase (copied from the text),
  - The character **start index**,
  - The character **end index**.
- Return nothing for categories with no match.
- Format:
  Category: "Extracted phrase", Start: start_index, End: end_index

Now extract from this text:
{chunk}
"""
        result = chat_with_gpt(prompt)
        if result:
            if DEBUG:
                print(result)
            extracted = parse_response(result)
            all_extractions.extend(extracted)

    # Final output
    xml_output = generate_xml(text, all_extractions)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_output)
    print(f"✅ Extraction complete. XML saved to: {output_file}")
else:
    print("No text to process.")
