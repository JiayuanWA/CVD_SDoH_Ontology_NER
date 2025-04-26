import openai
import nltk
import re
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
from collections import defaultdict

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

def normalize_spaces(text):
    """Collapse multiple spaces into one, strip."""
    return re.sub(r'\s+', ' ', text).strip()

def parse_response(raw_text, full_text):
    pattern = r'^(.+?):\s*"(.+?)",\s*Start:\s*\d+,\s*End:\s*\d+'
    results = []

    # Normalize both sides for matching
    full_text_normalized = normalize_spaces(full_text)

    for line in raw_text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            category, phrase = match.groups()
            phrase_normalized = normalize_spaces(phrase)

            # Try to find match in normalized text
            start_in_normalized = full_text_normalized.find(phrase_normalized)
            if start_in_normalized == -1:
                print(f"Phrase not found after normalization: \"{phrase}\" — skipping.")
                continue

            # Now find original span manually in the original full_text
            # Strategy: Search for the original (non-normalized) phrase in full_text
            start = full_text.find(phrase)
            if start == -1:
                # Fallback: search normalized phrase by allowing flexible spaces
                pattern_phrase = re.sub(r'\s+', r'\\s+', re.escape(phrase_normalized))
                match_original = re.search(pattern_phrase, full_text)
                if match_original:
                    start = match_original.start()
                    end = match_original.end()
                else:
                    print(f"Even regex failed to find: \"{phrase}\" — skipping.")
                    continue
            else:
                end = start + len(phrase)

            results.append({
                "category": category.strip(),
                "phrase": phrase,
                "start": start,
                "end": end
            })
        else:
            print(f"Failed to parse line: {line.strip()}")
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

def similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio() >= threshold

import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from collections import defaultdict

def text_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio() >= threshold

def iou(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    inter = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return inter / union if union > 0 else 0

def evaluate_text_category_then_span(gold_path, predictions):
    tree = ET.parse(gold_path)
    root = tree.getroot()

    gold_tags = []
    for tag in root.find("TAGS"):
        category = tag.tag.replace("_", " ")
        phrase = tag.attrib["text"].strip()
        span = tag.attrib["spans"]
        start, end = map(int, span.split("~"))
        gold_tags.append({
            "category": category,
            "phrase": phrase,
            "start": start,
            "end": end
        })

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

            # EXACT match check (text and category)
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
            # PARTIAL match check (text included but not identical)
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

    return {
        "TP_exact": matched_exact,
        "TP_partial": matched_partial,
        "TP_wrong_type": matched_wrong_type,
        "FP": false_positives,
        "FN": false_negatives,
        "type_counts": dict(type_counts)
    }

def print_error_table(fp_list, fn_list):
    print("\n--- Errors ---")
    print("\nFalse Positives (Predicted but not in Gold):")
    for pred in fp_list:
        print(f"{pred['category']}: \"{pred['phrase']}\" [{pred['start']}~{pred['end']}]")

    print("\nFalse Negatives (Gold but not Predicted):")
    for gold in fn_list:
        print(f"{gold['category']}: \"{gold['phrase']}\" [{gold['start']}~{gold['end']}]")


def evaluate_against_gold(gold_path, predictions):
    tree = ET.parse(gold_path)
    root = tree.getroot()
    gold_tags = []

    for tag in root.find("TAGS"):
        category = tag.tag.replace("_", " ")
        phrase = tag.attrib["text"].strip()
        gold_tags.append((category, phrase))

    matched = 0
    unmatched_preds = []

    for pred in predictions:
        found = False
        for gold_cat, gold_text in gold_tags:
            if pred["category"] == gold_cat and similar(pred["phrase"], gold_text):
                matched += 1
                found = True
                break
        if not found:
            unmatched_preds.append(pred)

    precision = matched / len(predictions) if predictions else 0
    recall = matched / len(gold_tags) if gold_tags else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Evaluation (Category + Phrase Match) ---")
    print(f"True Positives (matched): {matched}")
    print(f"False Positives: {len(unmatched_preds)}")
    print(f"False Negatives: {len(gold_tags) - matched}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

# Main Pipeline
text = read_text_file(file_path)

if text:
    text_chunks = [text]
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
        - If a category appears more than once, return all matches.
        - Do NOT paraphrase or interpret. Use exact quotes.
        - When extracting, always include surrounding words that are necessary to preserve the full meaning (e.g., extract "number of social partners" rather than just "social partners").
        - Do not shorten phrases by dropping adjectives, numbers, quantities, or modifiers.
        - Prefer longer complete expressions over partial phrases if both exist.
        - Extract all complete meaningful expressions, but ignore overly generic words like 'health', 'conditions', or 'outcomes' unless fully specified (e.g., 'high blood pressure').
        - Extract **every single matching phrase** for each category, even if it appears redundant, repeated, or minor.
        - Ignore references, citations, metadata, author affiliations, and publication info (e.g., journal titles, DOIs, PMID, author lists).

        ### Categories:
                    - **Age**: Mentions of specific ages, age ranges, or life stages (e.g., child, adolescent, elderly).
                    - **Origin**: Mentions of nationality, country of origin, or geographic location.
                    - **Ethnicity_Race**: Mentions of racial or ethnic identity (e.g., Black, Hispanic).
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
            extracted = parse_response(result, text)
            all_extractions.extend(extracted)

    xml_output = generate_xml(text, all_extractions)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_output)
    print(f"Extraction complete. XML saved to: {output_file}")

    # Run evaluation
    results = evaluate_text_category_then_span(gold_path, all_extractions)

    print_error_table(results["FP"], results["FN"])

else:
    print("No text to process.")
