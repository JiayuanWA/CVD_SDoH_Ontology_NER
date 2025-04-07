import openai
import nltk
import re
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download("punkt")

DEBUG = False

API_KEY = "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"
client = openai.OpenAI(api_key=API_KEY)

file_path = "sdoh_text.txt"
output_file = "sdoh_extracted.xml"
examples_folder = "examples_xml"

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
    pattern = r'^(.*?):\s*\"(.*?)\",\s*Start:\s*(\d+),\s*End:\s*(\d+)'
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
    id_counter = {}
    tag_lines = []
    for item in extractions:
        tag_type = item["category"].strip().replace(" ", "_")
        base_id = "".join([word[0] for word in tag_type.split("_")])
        count = id_counter.get(tag_type, 0)
        tag_id = f"{base_id}{count}"
        id_counter[tag_type] = count + 1
        tag_line = f'<{tag_type} spans="{item["start"]}~{item["end"]}" text="{item["phrase"]}" id="{tag_id}" comment=""/>'
        tag_lines.append(tag_line)
    xml_content = '<?xml version="1.0" encoding="UTF-8" ?>\n'
    xml_content += "<CVD_SDOH>\n"
    xml_content += "<TEXT><![CDATA[\n" + text + "\n]]></TEXT>\n"
    xml_content += "<TAGS>\n" + "\n".join(tag_lines) + "\n</TAGS>\n"
    xml_content += "<META/>\n</CVD_SDOH>"
    return xml_content

def load_few_shot_examples(folder_path, max_examples=2):
    examples = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")][:max_examples]
    for file in files:
        tree = ET.parse(os.path.join(folder_path, file))
        root = tree.getroot()
        text = root.find("TEXT").text or ""
        tags = root.find("TAGS")
        formatted_tags = []
        for tag in tags:
            category = tag.tag.replace("_", " ")
            phrase = tag.attrib["text"]
            span = tag.attrib["spans"]
            start, end = span.split("~")
            formatted_tags.append(f'{category}: "{phrase}", Start: {start}, End: {end}')
        example = f"""### Example Input:\n{text.strip()}\n\n### Example Output:\n{chr(10).join(formatted_tags)}\n"""
        examples.append(example)
    return "\n".join(examples)

# Main Pipeline
text = read_text_file(file_path)

if text:
    text_chunks = chunk_text(text, max_tokens=2000)
    all_extractions = []
    few_shot_prompt = load_few_shot_examples(examples_folder)

    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)}...")
        prompt = few_shot_prompt + f"""
You are a high-accuracy extraction model for Social Determinants of Health (SDoH). Your task is to scan the input text and extract all exact phrases that match any of the 34 SDoH categories listed below.

### Guidelines:
- ONLY extract phrases that match one of the 34 categories exactly as written in the list below.
- All extracted phrases must be copied exactly from the input text.
- Return the character start and end position of each extracted phrase.
- If a category appears more than once, return all matches.
- Do NOT paraphrase or interpret. Use exact quotes.
- Extract **every single matching phrase** for each category, even if it appears redundant, repeated, or minor.
- Ignore references, citations, metadata, author affiliations, and publication info (e.g., journal titles, DOIs, PMID, author lists).

### Categories:
Age, Origin, Ethnicity, Gender, Marital Status, Alcohol, Tobacco, Illicit Drugs, Caffeine, Stress, Psychological Concerns, Social Adversities, Nutrition, Physical Activity, Sleep, Sexual Behavior, Contraception, Treatment Adherence, Employment, Financial Strain, Food Insecurity, Housing Stability, Caregiving, Living Situation, Social Support, Safety, Transportation, Utilities, Medical Access, Insurance, Healthcare Technology, Erratic Care, Maintaining Care, Outcome

### Output Format:
CategoryName: "Exact phrase from text", Start: starting_index, End: ending_index

Now extract from this text:
{chunk}
"""
        result = chat_with_gpt(prompt)
        if result:
            if DEBUG:
                print(f"\n--- GPT Output ---\n{result}\n")
            extracted = parse_response(result)
            all_extractions.extend(extracted)

    xml_output = generate_xml(text, all_extractions)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_output)
    print(f"✅ Extraction complete. XML saved to: {output_file}")
else:
    print("No text to process.")
