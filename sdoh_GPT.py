
import openai
import nltk
import re
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from collections import defaultdict

nltk.download("punkt")

API_KEY = "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"

client = openai.OpenAI(api_key=API_KEY)

file_path = "sdoh_text.txt"
output_file = "sdoh_extracted.xml"
examples_folder = "examples_xml"
gold_path = "gold_sdoh.xml"

# Define category batches
BATCHES = {
    "Demographic": ["Age", "Origin", "Ethnicity_Race", "Gender", "Marital Status"],
    "Substance Use": ["Alcohol Use", "Tobacco Use", "Illicit Drug Use", "Caffeine Use"],
    "Psychosocial + Lifestyle": [
        "Stress", "Psychological Concern", "Nutrition and Diet", "Physical Activity",
        "Sleep", "Sexual Activities", "Use of Contraception", "Treatment Adherence"
    ],
    "Economic Stability + Access": [
        "Employment", "Financial_Resource", "Food Insecurity", "Housing Stability",
        "Insurance Coverage", "Healthcare Technology Access", "Medical Access",
        "Erratic Healthcare", "Maintaining Care", "Dental Access"
    ],
    "Community + Built Environment": [
        "Caregiver Support", "Living Situation", "Social Connections", "Social Adversity",
        "Education", "Safety & Environmental Exposure", "Transportation Needs", "Utilities"
    ]
}

CATEGORY_DEFINITIONS = {
    "Age": "Mentions of specific ages, age ranges, or life stages (e.g., child, adolescent, elderly).",
    "Origin": "Mentions of nationality, country of origin, or geographic location.",
    "Ethnicity_Race": "Mentions of racial or ethnic identity (e.g., Black, Hispanic).",
    "Gender": "Mentions of male, female, or gender identity terms.",
    "Marital Status": "Mentions of relationship status (e.g., married, divorced, widowed).",
    "Alcohol Use": "Mentions of alcohol consumption, frequency, and effects.",
    "Tobacco Use": "Mentions of cigarettes, cigars, vaping, or smokeless tobacco.",
    "Illicit Drug Use": "Mentions of illegal drug use, including marijuana and prescription drug abuse.",
    "Caffeine Use": "Mentions of coffee, tea, energy drinks, or caffeine-related effects.",
    "Stress": "Mentions of psychological or emotional stress (e.g., work, family, financial stress).",
    "Psychological Concern": "Mentions of mental health issues (e.g., anxiety, depression, loneliness).",
    "Nutrition and Diet": "Mentions of dietary habits, food preferences, and nutrition.",
    "Physical Activity": "Mentions of exercise, activity levels, or inactivity.",
    "Sleep": "Mentions of sleep patterns, quality, or disorders.",
    "Sexual Activities": "Mentions of sexual behavior, history, or safe sex practices.",
    "Use of Contraception": "Mentions of birth control or contraceptive methods.",
    "Treatment Adherence": "Mentions of medication adherence, missed doses, or compliance.",
    "Employment": "Mentions of jobs, professions, work conditions, or unemployment.",
    "Financial_Resource": "Mentions of financial struggles, poverty, or inability to pay bills.",
    "Food Insecurity": "Mentions of hunger, limited food access, or nutritional deprivation.",
    "Housing Stability": "Mentions of homelessness, frequent moves, or housing issues.",
    "Insurance Coverage": "Mentions of health insurance, coverage, or lack of coverage.",
    "Healthcare Technology Access": "Mentions of telehealth, medical apps, or patient portals.",
    "Medical Access": "Mentions of difficulty in accessing medical care, including barriers.",
    "Erratic Healthcare": "Mentions of irregular medical care, missed appointments, or delays.",
    "Maintaining Care": "Mentions of regular check-ups, specialist visits, or long-term care.",
    "Dental Access": "Mentions of access to dental care and oral health.",
    "Caregiver Support": "Mentions of caregiving support from family, friends, or professionals.",
    "Living Situation": "Mentions of with whom or where the person lives.",
    "Social Connections": "Mentions of emotional or physical support from family, friends, or others.",
    "Social Adversity": "Mentions of abuse, discrimination, or adverse social experiences.",
    "Education": "Mentions of educational attainment, school or college attendance, literacy level, or barriers to accessing education.",
    "Safety & Environmental Exposure": "Mentions of unsafe environments, toxins, or hazardous conditions.",
    "Transportation Needs": "Mentions of transportation access, barriers, or challenges.",
    "Utilities": "Mentions of access to electricity, water, internet, or related issues."
}

def get_definitions_for_categories(selected_categories):
    return "\n".join([f"- **{cat}**: {CATEGORY_DEFINITIONS[cat]}" for cat in selected_categories])

def read_text_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

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
        example = f"### Example Input:\n{text.strip()}\n\n### Example Output:\n{chr(10).join(formatted_tags)}\n"
        examples.append(example)
    return "\n".join(examples)

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

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def parse_response(raw_text, full_text):
    pattern = r'^(.+?):\s*"(.+?)",\s*Start:\s*(\d+),\s*End:\s*(\d+)'
    results = []
    VALID_CATEGORIES = set(CATEGORY_DEFINITIONS.keys())
    used_spans = set()

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or not re.match(pattern, line):
            continue

        match = re.match(pattern, line)
        if match:
            raw_cat, phrase, start, end = match.groups()

            # Clean category name
            category = re.sub(r'[^a-zA-Z &]', '', raw_cat).strip()
            if category not in VALID_CATEGORIES:
                print(f"⚠️ Skipping unknown category: {category}")
                continue

            start, end = int(start), int(end)
            span_key = (start, end)
            if span_key in used_spans:
                continue
            used_spans.add(span_key)

            results.append({
                "category": category,
                "phrase": phrase.strip(),
                "start": start,
                "end": end
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

def evaluate_against_gold(gold_path, predictions):
    tree = ET.parse(gold_path)
    root = tree.getroot()
    gold_tags = []
    for tag in root.find("TAGS"):
        category = tag.tag.replace("_", " ")
        phrase = tag.attrib["text"].strip()
        span = tag.attrib["spans"]
        start, end = map(int, span.split("~"))
        gold_tags.append({"category": category, "phrase": phrase, "start": start, "end": end})

    matched_exact = 0
    matched_partial = 0
    matched_wrong_type = 0
    false_positives = []
    matched_ids = set()
    type_counts = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for pred in predictions:
        pred_span = (pred["start"], pred["end"])
        pred_cat = pred["category"]
        pred_text = pred["phrase"]
        matched = False

        for i, gold in enumerate(gold_tags):
            if i in matched_ids:
                continue
            gold_span = (gold["start"], gold["end"])
            gold_cat = gold["category"]
            gold_text = gold["phrase"]

            if pred_text.strip() == gold_text.strip():
                matched_ids.add(i)
                matched = True
                if pred_cat == gold_cat:
                    matched_exact += 1
                    type_counts[pred_cat]["TP"] += 1
                else:
                    matched_wrong_type += 1
                    type_counts[pred_cat]["FP"] += 1
                    type_counts[gold_cat]["FN"] += 1
                break
            elif (pred_text.strip() in gold_text.strip()) or (gold_text.strip() in pred_text.strip()):
                matched_ids.add(i)
                matched = True
                matched_partial += 1
                break

        if not matched:
            false_positives.append(pred)
            type_counts[pred_cat]["FP"] += 1

    false_negatives = [gold for i, gold in enumerate(gold_tags) if i not in matched_ids]
    for fn in false_negatives:
        type_counts[fn["category"]]["FN"] += 1

    precision = matched_exact / len(predictions) if predictions else 0
    recall = matched_exact / len(gold_tags) if gold_tags else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Evaluation (Strict Exact Match) ---")
    print(f"True Positives (exact match + type):        {matched_exact}")
    print(f"Partial Matches (substring overlap):        {matched_partial}")
    print(f"Wrong Category Matches (text ok):           {matched_wrong_type}")
    print(f"False Positives (no match):                 {len(false_positives)}")
    print(f"False Negatives (gold not matched):         {len(false_negatives)}")
    print(f"Precision (exact only):                     {precision:.3f}")
    print(f"Recall (exact only):                        {recall:.3f}")
    print(f"F1 Score (exact only):                      {f1:.3f}")

    print("\n--- Per-Class Breakdown (exact matches only) ---")
    for tag, counts in type_counts.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{tag:25} | P: {prec:.2f}  R: {rec:.2f}  F1: {f1c:.2f}  (TP: {tp}, FP: {fp}, FN: {fn})")

# Main Run
text = read_text_file(file_path)
few_shot_prompt = load_few_shot_examples(examples_folder)
all_extractions = []

if text:
    for batch_name, categories in BATCHES.items():
        print(f"\n--- Processing Batch: {batch_name} ---")
        category_definitions = get_definitions_for_categories(categories)

        prompt = few_shot_prompt + f"""
You are a high-accuracy extraction model for Social Determinants of Health (SDoH).

### Guidelines:
- Go category by category.
- Extract **complete and self-contained phrases**.
- Copy exact text spans. Do not paraphrase.
- Extract every relevant match for the given categories.

### Categories in this batch:
{category_definitions}

### Output Format:
CategoryName: "Exact phrase from text", Start: starting_index, End: ending_index

Now extract from this text:
{text}
"""
        result = chat_with_gpt(prompt)
        if result:
            batch_extractions = parse_response(result, text)
            all_extractions.extend(batch_extractions)

    xml_output = generate_xml(text, all_extractions)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_output)
    print(f"\n✅ Extraction complete. XML saved to: {output_file}")

    evaluate_against_gold(gold_path, all_extractions)
