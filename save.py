import ollama
import json
import re

# Load input text from a file
def read_text_file(file_path):
    """Reads a text file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

# Function to chunk long text into smaller parts
def chunk_text(text, max_tokens=2048):
    """Splits text into smaller chunks that fit within the model's context limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # Account for spaces

        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add last chunk

    return chunks

# Function to extract the correct JSON format, even if model returns extra text
def extract_valid_json(json_response, chunk_number):
    """Extracts valid JSON SDoH entities, handling different response structures."""
    try:
        # Try loading JSON normally
        data = json.loads(json_response.strip())

        # Case 1: If response is a list of entities, return it
        if isinstance(data, list):
            return data

        # Case 2: If response is a dictionary containing "data" as a list, extract it
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]

        # Case 3: If response is a single dictionary with a category and text, wrap it in a list
        if isinstance(data, dict) and "category" in data and "text" in data:
            return [data]  

    except json.JSONDecodeError:
        # If direct JSON parsing fails, try extracting JSON using regex
        match = re.search(r"\[\s*\{.*?\}\s*\]", json_response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass  # If extraction fails, continue to error message

    print(f"Unexpected format in chunk {chunk_number}. Skipping...")
    return []

# Path to your SDoH text file
file_path = "sdoh_text.txt"
text = read_text_file(file_path)

if text:
    print("\nRunning Llama 3 SDoH Extraction...\n")

    # Chunk text if it's too long
    text_chunks = chunk_text(text, max_tokens=2000)  # Adjust token size as needed

    all_sdoh_entities = []  # Store extracted entities

    for i, chunk in enumerate(text_chunks):
        print(f"\nProcessing chunk {i + 1}/{len(text_chunks)}...\n")

        # 🔹 Llama 3 Prompt for Structured SDoH Extraction 🔹
        prompt = f"""
        Extract Social Determinants of Health (SDoH) terms and categorize them strictly as JSON.
        Identify relevant phrases and map them to one of these 35 categories:
        Outcome, Age, Origin, Ethnicity, Gender, Marital Status, Alcohol, Tobacco, Illicit Drugs, Caffeine, Stress, Psychological Concerns, Social Adversities, Nutrition, Physical Activity, Sleep, Sexual Behavior, Contraception, Treatment Adherence, Employment, Financial Strain, Food Insecurity, Housing Stability, Caregiving, Living Situation, Social Support, Safety, Transportation, Utilities, Medical Access, Insurance, Healthcare Technology, Erratic Care, Maintaining Care.
        ### **SDoH Categories & Annotation Guidelines**
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
            - **Social Adversities**: Mentions of abuse, discrimination, or adverse social experiences.
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
            - **Social Support**: Mentions of emotional or physical support from family, friends, or others.
            - **Safety & Environmental Exposure**: Mentions of unsafe environments, toxins, or hazardous conditions.
            - **Transportation Needs**: Mentions of transportation access, barriers, or challenges.
            - **Utilities**: Mentions of access to electricity, water, internet, or related issues.
            - **Outcome**: Mentions of specific **health conditions, diagnoses, or results** related to individual or population health.  
       

        ### **Expected JSON Output:**
        ```json
        [
            {{"category": "Financial Strain", "text": "income"}},
            {{"category": "Social Adversities", "text": "self-reported exposure to racial discrimination"}}
        ]
        ```

        Now, extract all SDoH terms and phrases from the following text:

        {chunk}
        """

        # Run the model with Ollama enforcing JSON output
        try:
            response = ollama.chat(
                model="deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )

            generated_text = response['message']['content']

            # Debugging: Print raw model response
            print(f"\n DEBUG - Raw Model Response (Chunk {i+1}):\n{generated_text}\n")

            # Extract JSON safely
            extracted_entities = extract_valid_json(generated_text, i + 1)
            all_sdoh_entities.extend(extracted_entities)

        except Exception as e:
            print(f"⚠️ Error processing chunk {i + 1}: {e}")
            continue

    # Print extracted SDoH entities
    print("\n✅ **Extracted SDoH Entities Across All Chunks:**\n")
    print(json.dumps(all_sdoh_entities, indent=2))

    # Save results to a file
    output_file = "sdoh_llm_results.json"
    if all_sdoh_entities:
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(all_sdoh_entities, out, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
    else:
        print("❌ No meaningful SDoH terms detected. File not saved.")
else:
    print("⚠️ No text to process. Please check your input file.")
