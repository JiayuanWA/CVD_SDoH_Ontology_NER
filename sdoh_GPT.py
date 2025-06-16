
import openai
import nltk
import re
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from collections import defaultdict
import xml.etree.ElementTree as ET

nltk.download("punkt")

API_KEY = "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"

client = openai.OpenAI(api_key=API_KEY)

file_path = "sdoh_text.txt"
output_file = "sdoh_extracted.xml"
examples_folder = "examples_xml"
gold_path = "gold_sdoh.xml"

# Define category batches
BATCHES = {
    "Demographic": ["Age", "Origin", "Ethnicity_Race", "Gender", "Marital_Status"],
    "Substance Use": ["Alcohol_Use", "Tobacco_Use", "Illicit_Drug_Use", "Caffeine_Use"],
    "Psychosocial + Lifestyle": [
        "Stress", "Psychological_Concern", "Nutrition_and_Diet", "Physical_Activity",
        "Sleep", "Sexual_Activities", "Use_of_Contraception", "Treatment_Adherence"
    ],
    "Economic Stability + Access": [
        "Employment", "Financial_Resource", "Food_Insecurity", "Housing_Stability",
        "Insurance_Coverage", "Healthcare_Technology_Access", "Medical_Transportation_Access",
        "Erratic_Healthcare", "Maintaining_Care", "Dental_Access"
    ],
    "Community + Built Environment": [
        "Caregiver_Support", "Living_Status", "Social_Connections", "Social_Adversity",
        "Education", "Safety_and_Environmental", "Transportation_Needs", "Utilities"
    ]
}

CATEGORY_DEFINITIONS = {
    "Age": "Mentions of specific ages, age ranges, or life stages (e.g., child, adolescent, elderly).",
    "Origin": "Mentions of nationality, country of origin, geographic location, or spatial context such as urban vs. rural, city names, regions, neighborhoods, or other location-based identifiers relevant to an individual's background or environment. Do not extract Origin from author affiliations, institutional addresses, or contact information.",
    "Ethnicity_Race": "Mentions of racial or ethnic identity, including named groups (e.g., Black, Hispanic) as well as general references such as 'racial', 'ethnic', or 'race/ethnicity'.",
    "Gender": "Mentions of male, female, or gender identity terms.",
    "Marital_Status": "Mentions of relationship status (e.g., married, divorced, widowed).",
    "Alcohol_Use": "Mentions of alcohol consumption, frequency, and effects.",
    "Tobacco_Use": "Mentions of smoking, cigarettes, cigars, vaping, or smokeless tobacco.",
    "Illicit_Drug_Use": "Mentions of illegal drug use, including marijuana and prescription drug abuse.",
    "Caffeine_Use": "Mentions of coffee, tea, energy drinks, or caffeine-related effects.",
    "Stress": "Mentions of psychological or emotional stress (e.g., work, family, financial stress).",
    "Psychological_Concern": "Mentions of mental health issues (e.g., anxiety, depression, loneliness).",
    "Nutrition_and_Diet": "Mentions of dietary habits, food preferences, and nutrition.",
    "Physical_Activity": "Mentions of exercise, activity levels, or inactivity.",
    "Sleep": "Mentions of sleep patterns, quality, or disorders.",
    "Sexual_Activities": "Mentions of sexual behavior, history, or safe sex practices.",
    "Use_of_Contraception": "Mentions of birth control or contraceptive methods.",
    "Treatment_Adherence": "Mentions of medication adherence, missed doses, or compliance.",
    "Employment": "Mentions of jobs, professions, work conditions, unemployment.Includes socioeconomic identifiers such as 'working-class', 'blue-collar', or references to the labor force, job types, workplace environments, or occupational categories.",
    "Financial_Resource": "Mentions of financial struggles, poverty, or inability to pay bills. Includes terms like 'low-income', 'high-income', 'fixed income', 'wealthy', 'broke', or general references to personal or household finances, affordability, and economic class.",
    "Food_Insecurity": "Mentions of hunger, limited food access, or nutritional deprivation.",
    "Housing_Stability": "Mentions of homelessness, frequent moves, or housing issues.",
    "Insurance_Coverage": "Mentions of health insurance, coverage, or lack of insurance coverage.",
    "Healthcare_Technology_Access": "Mentions of telehealth, medical apps, or patient portals.",
    "Medical_Transportation_Access": "An individual’s ability to secure reliable transportation specifically for accessing healthcare services, such as appointments, procedures, emergency care, or pharmacies. This includes availability of non-emergency medical transport, ride programs, insurance-covered transit, or personal means to reach healthcare facilities.",
    "Erratic_Healthcare": "Mentions of irregular medical care, missed appointments, or delays.",
    "Maintaining_Care": "Mentions of regular check-ups, specialist visits, or long-term care.",
    "Dental_Access": "Mentions of access to dental care and oral health.",
    "Caregiver_Support": "Mentions of caregiving support from family, friends, or professionals.",
    "Living_Status": "Mentions of with whom or where the person lives.",
    "Social_Connections": "Mentions of emotional or physical support from family, friends, or others.",
    "Social_Adversity": "Mentions of abuse, discrimination, or adverse social experiences.",
    "Education": "Extract any phrase related to education, including training, instruction, literacy, curricula, school, or educational efforts, even if the reference is indirect or brief.",
    "Safety_and_Environmental": "Mentions of physical or environmental risk or danger (e.g., disaster, toxins, poor sanitation, lack of utilities, unsafe infrastructure)",
    "Transportation_Needs": "Mentions of transportation access, barriers, or challenges.",
    "Utilities": "Mentions of access to electricity, water, internet, or related issues."
}

def get_definitions_for_categories(selected_categories):
    return "\n".join([f"- **{cat}**: {CATEGORY_DEFINITIONS[cat]}" for cat in selected_categories])

def read_text_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def load_few_shot_examples(folder_path, max_examples=6):
    examples = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".xml")][:max_examples]
    for file in files:
        tree = ET.parse(os.path.join(folder_path, file))
        root = tree.getroot()
        text = root.find("TEXT").text or ""
        tags = root.find("TAGS")
        formatted_tags = []
        for tag in tags:
            category = tag.tag.replace("_", "_")
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
    results = []
    CATEGORY_LOOKUP = {k.lower().replace(" ", ""): k for k in CATEGORY_DEFINITIONS.keys()}
    used_spans = set()

    for line in raw_text.splitlines():
        line = line.strip()
        if ':' not in line:
            continue
    
        # Split into category and phrases block
        raw_cat, phrases_block = line.split(":", 1)
        phrases = re.findall(r'"(.*?)"', phrases_block)

        category = raw_cat.strip()
        if category not in CATEGORY_DEFINITIONS:
            print(f"Skipping unknown category: {category}")
            continue

        seen_tags = set()

        # Process all extracted phrases under that category
        for phrase in phrases:
            phrase_clean = phrase.strip()
            
            # Collect all potential spans first
            candidate_spans = []
            for m in re.finditer(re.escape(phrase_clean), full_text):
                start, end = m.start(), m.end()
                candidate_spans.append((start, end))

            candidate_spans.sort(key=lambda x: x[1] - x[0], reverse=True)

            # Filter out overlapping matches
            for start, end in candidate_spans:
                if any(not (end <= s or start >= e) for (s, e) in used_spans):
                    continue 

                tag_key = (category, phrase_clean, start, end)
                if tag_key in seen_tags:
                    continue  # exact duplicate

                used_spans.add((start, end))
                seen_tags.add(tag_key)

                results.append({
                    "category": category,
                    "phrase": phrase_clean,
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


def spans_overlap(pred_start, pred_end, gold_start, gold_end):
    return max(pred_start, gold_start) < min(pred_end, gold_end)

def evaluate_against_gold(gold_path, predictions):
    tree = ET.parse(gold_path)
    root = tree.getroot()
    gold_tags = []
    for tag in root.find("TAGS"):
        category = tag.tag.strip()
        phrase = tag.attrib["text"].strip()
        span = tag.attrib["spans"]
        start, end = map(int, span.split("~"))
        gold_tags.append({"category": category, "phrase": phrase, "start": start, "end": end})

    matched_exact = 0
    matched_partial = 0
    matched_wrong_type = 0
    false_positives = []
    matched_ids = set()
    type_counts_exact = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    type_counts_combined = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    print("\n--- Detailed Prediction Debugging ---")
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

            # Exact match
            if pred_span == gold_span and pred_cat == gold_cat:
                matched_exact += 1
                matched_ids.add(i)
                matched = True
                type_counts_exact[pred_cat]["TP"] += 1
                type_counts_combined[pred_cat]["TP"] += 1
                print(f"EXACT MATCH | Category: {pred_cat} | Text: \"{pred_text}\"")
                break

            # Partial match
            elif spans_overlap(*pred_span, *gold_span) and pred_cat == gold_cat:
                matched_partial += 1
                matched_ids.add(i)
                matched = True
                type_counts_combined[pred_cat]["TP"] += 1
                print(f"PARTIAL MATCH | Category: {pred_cat} | Pred: \"{pred_text}\" vs Gold: \"{gold_text}\"")
                break

            # Wrong category
            elif spans_overlap(*pred_span, *gold_span) and pred_cat != gold_cat:
                matched_wrong_type += 1
                matched_ids.add(i)
                matched = True
                type_counts_exact[pred_cat]["FP"] += 1
                type_counts_exact[gold_cat]["FN"] += 1
                type_counts_combined[pred_cat]["FP"] += 1
                type_counts_combined[gold_cat]["FN"] += 1
                print(f"WRONG TYPE | Pred: ({pred_cat}) \"{pred_text}\" vs Gold: ({gold_cat}) \"{gold_text}\"")
                break

        if not matched:
            false_positives.append(pred)
            type_counts_exact[pred_cat]["FP"] += 1
            type_counts_combined[pred_cat]["FP"] += 1
            print(f"NO MATCH     | Category: {pred_cat} | Text: \"{pred_text}\"")

    false_negatives = [gold for i, gold in enumerate(gold_tags) if i not in matched_ids]
    for fn in false_negatives:
        type_counts_exact[fn["category"]]["FN"] += 1
        type_counts_combined[fn["category"]]["FN"] += 1
        print(f"MISSED GOLD | Category: {fn['category']} | Text: \"{fn['phrase']}\"")

    # --- Overall scores ---
    precision_exact = matched_exact / len(predictions) if predictions else 0
    recall_exact = matched_exact / len(gold_tags) if gold_tags else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0

    total_matched_combined = matched_exact + matched_partial
    precision_combined = total_matched_combined / len(predictions) if predictions else 0
    recall_combined = total_matched_combined / len(gold_tags) if gold_tags else 0
    f1_combined = 2 * precision_combined * recall_combined / (precision_combined + recall_combined) if (precision_combined + recall_combined) > 0 else 0

    # --- Print Results ---
    print("\n--- Evaluation (Strict Exact Match Only) ---")
    print(f"True Positives (exact span + correct type):      {matched_exact}")
    print(f"Type Mismatch (span matched, wrong type):        {matched_wrong_type}")
    print(f"False Positives (no overlap with gold):           {len(false_positives)}")
    print(f"False Negatives (gold missed entirely):           {len(false_negatives)}")
    print(f"Precision (exact only):                           {precision_exact:.3f}")
    print(f"Recall (exact only):                              {recall_exact:.3f}")
    print(f"F1 Score (exact only):                            {f1_exact:.3f}")

    print("\n--- Evaluation (Relaxed: Exact + Partial Match) ---")
    print(f"True Positives (exact + partial span + correct type): {total_matched_combined}")
    print(f"Partial Matches (span overlap + correct type):         {matched_partial}")
    print(f"Precision (combined):                                {precision_combined:.3f}")
    print(f"Recall (combined):                                   {recall_combined:.3f}")
    print(f"F1 Score (combined):                                 {f1_combined:.3f}")


    print("\n--- Per-Class Breakdown (Exact Only) ---")
    for tag, counts in type_counts_exact.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{tag:25} | P: {prec:.2f}  R: {rec:.2f}  F1: {f1c:.2f}  (TP: {tp}, FP: {fp}, FN: {fn})")

    print("\n--- Per-Class Breakdown (Exact + Partial) ---")
    for tag, counts in type_counts_combined.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{tag:25} | P: {prec:.2f}  R: {rec:.2f}  F1: {f1c:.2f}  (TP: {tp}, FP: {fp}, FN: {fn})")


# Main Run
text = read_text_file(file_path)
few_shot_prompt = load_few_shot_examples(examples_folder)
all_extractions = []
all_raw_outputs = []

if text:
    overall_output_path = "overall_output.txt"
    with open(overall_output_path, "w", encoding="utf-8") as f:
        f.write("")

    for batch_name, categories in BATCHES.items():
        print(f"\n--- Processing Batch: {batch_name} ---")
        category_definitions = get_definitions_for_categories(categories)

        prompt = few_shot_prompt + f"""
            You are a high-accuracy extraction model for Social Determinants of Health (SDoH).

            ### Guidelines:
            - Go category by category.
            - Extract **complete and self-contained phrases**.
            - Copy exact text. Do not paraphrase.
            - Do not output the category name if there are no matches.
            - Extract every relevant match for the given categories.
            - Only output **unique** phrases per category. If a phrase appears more than once, list it only once.
            - Do NOT extract geographic location terms (e.g., city, country) from author affiliations, department addresses, or contact sections. Only extract locations if mentioned in the main body of the clinical or methods content.
            ### Categories in this batch:
            {category_definitions}

            ### Output Format:
            Use the following format exactly. Do not add any markdown, categories, bullet points, or commentary.

            Correct:
            Age: "adults"
            Origin: "urban"
            Origin: "Boston metropolitan area"
            Ethnicity_Race: "white"

            Incorrect:
            ### Age
            - "adults"


            Now extract from this text:
            {text}
            """
        result = chat_with_gpt(prompt)

        if result:
            with open(overall_output_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- Batch: {batch_name} ---\n{result}\n")
            print(f"✔️ Added batch output to: {overall_output_path}")

            all_raw_outputs.append(result)

# Combine all GPT raw outputs for downstream parsing
combined_output = "\n".join(all_raw_outputs)
all_extractions = parse_response(combined_output, text)

# Generate XML output
xml_output = generate_xml(text, all_extractions)
with open(output_file, "w", encoding="utf-8") as f:
    f.write(xml_output)
print(f"\nExtraction complete. XML saved to: {output_file}")

# Run evaluation
evaluate_against_gold(gold_path, all_extractions)
