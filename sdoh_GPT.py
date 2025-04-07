import openai
import nltk
import re
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher

# Download NLTK data
nltk.download("punkt")

DEBUG = False

API_KEY = "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"
client = openai.OpenAI(api_key=API_KEY)

file_path = "sdoh_text.txt"
output_file = "sdoh_extracted.xml"
examples_folder = "examples_xml"
gold_path = "gold_sdoh.xml"

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

def chat_with_gpt(prompt, model="gpt-4o"):
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

def evaluate_against_gold(gold_path, predictions):
    tree = ET.parse(gold_path)
    root = tree.getroot()
    gold_tags = set()
    category_errors, span_errors, text_errors = 0, 0, 0

    gold_lookup = []
    for tag in root.find("TAGS"):
        category = tag.tag.replace("_", " ")
        phrase = tag.attrib["text"].strip()
        span = tag.attrib["spans"].strip()
        gold_lookup.append((category, phrase, span))
        gold_tags.add((category, phrase, span))

    predicted_tags = set()
    for item in predictions:
        predicted_tags.add((item["category"], item["phrase"].strip(), f"{item['start']}~{item['end']}"))

    tp = len(predicted_tags & gold_tags)
    fp = len(predicted_tags - gold_tags)
    fn = len(gold_tags - predicted_tags)

    for pred in predicted_tags - gold_tags:
        cat, phr, span = pred
        if not any(g[0] == cat for g in gold_tags):
            category_errors += 1
        elif not any(g[2] == span for g in gold_tags if g[0] == cat):
            span_errors += 1
        elif not any(g[1] == phr for g in gold_tags if g[0] == cat):
            text_errors += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n--- Evaluation ---")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("\n--- Error Breakdown ---")
    print(f"Category mismatches: {category_errors}")
    print(f"Span mismatches:     {span_errors}")
    print(f"Text mismatches:     {text_errors}")

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
        - Go category by category.
        - For each category, extract **complete and self-contained phrases** from the text — not just isolated words. For example, extract "negative social support with respect to both their family and friends" instead of just "social support".
        - The extracted phrase must include **as much surrounding detail as needed to stand alone** and be useful for annotation or labeling.
        - Phrases should capture the **full expression of the concept** as written in the original sentence.
        - ONLY extract phrases that match one of the 34 categories exactly as written in the list below.
        - All extracted phrases must be copied exactly from the input text.
        - Return the character start and end position of each extracted phrase.
        - If a category appears more than once, return all matches.
        - Do NOT paraphrase or interpret. Use exact quotes.
        - Extract at least 20 phrases.
        - Extract **every single matching phrase** for each category, even if it appears redundant, repeated, or minor.
        - Ignore references, citations, metadata, author affiliations, and publication info (e.g., journal titles, DOIs, PMID, author lists).

        ### Categories:
                    - **Age**: Mentions of specific ages, age ranges, or life stages (e.g., child, adolescent, elderly).
                    - **Origin**: Mentions of nationality, country of origin, or geographic location.
                    - **Ethnicity/Race**: Mentions of racial or ethnic identity (e.g., Black, Hispanic).
                    - **Gender**: Mentions of male, female, or gender identity terms.
                    - **Marital Status**: Mentions of relationship status (e.g., married, divorced, widowed).
                    - **Alcohol Use**: Mentions of alcohol consumption, frequency, and effects.
                    - **Tobacco Use**: Mentions of cigarettes, cigars, vaping, or smokeless tobacco.
                    - **Illicit Drug Use**: Mentions of illegal drug use, including marijuana and prescription drug abuse.
                    - **Caffeine Use**: Mentions of coffee, tea, energy drinks, or caffeine-related effects.
                    - **Stress**: Mentions of psychological or emotional stress (e.g., work, family, financial stress).
                    - **Psychological Concern**: Mentions of mental health issues (e.g., anxiety, depression, loneliness).
                    - **Social Adversity**: Mentions of abuse, discrimination, or adverse social experiences.
                    - **Nutrition and Diet**: Mentions of dietary habits, food preferences, and nutrition.
                    - **Physical Activity**: Mentions of exercise, activity levels, or inactivity.
                    - **Sleep**: Mentions of sleep patterns, quality, or disorders.
                    - **Sexual Activities**: Mentions of sexual behavior, history, or safe sex practices.
                    - **Use of Contraception**: Mentions of birth control or contraceptive methods.
                    - **Treatment Adherence**: Mentions of medication adherence, missed doses, or compliance.
                    - **Medical Access**: Mentions of difficulty in accessing medical care, including barriers.
                    - **Dental Access**: Mentions of access to dental care and oral health.
                    - **Insurance Coverage**: Mentions of health insurance, coverage, or lack of coverage.
                    - **Healthcare Technology Access**: Mentions of telehealth, medical apps, or patient portals.
                    - **Erratic Healthcare**: Mentions of irregular medical care, missed appointments, or delays.
                    - **Maintaining Care**: Mentions of regular check-ups, specialist visits, or long-term care.
                    - **Employment**: Mentions of jobs, professions, work conditions, or unemployment.
                    - **Financial Strain**: Mentions of financial struggles, poverty, or inability to pay bills.
                    - **Food Insecurity**: Mentions of hunger, limited food access, or nutritional deprivation.
                    - **Housing Stability**: Mentions of homelessness, frequent moves, or housing issues.
                    - **Caregiver Support**: Mentions of caregiving support from family, friends, or professionals.
                    - **Living Situation**: Mentions of with whom or where the person lives.
                    - **Social Connections**: Mentions of emotional or physical support from family, friends, or others.
                    - **Safety & Environmental Exposure**: Mentions of unsafe environments, toxins, or hazardous conditions.
                    - **Transportation Needs**: Mentions of transportation access, barriers, or challenges.
                    - **Utilities**: Mentions of access to electricity, water, internet, or related issues.
                    - **Outcome**: Mentions of specific **health conditions, diagnoses, or results** related to individual or population health.  
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


    # Run evaluation
    evaluate_against_gold(gold_path, all_extractions)
else:
    print("No text to process.")
