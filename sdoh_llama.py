import ollama
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")  

DEBUG = False

def read_text_file(file_path):
    """Reads a text file and returns its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def chunk_text(text, max_tokens=2048):
    """Splits text into smaller chunks while preserving sentence structure."""
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

# Path to your SDoH text file
file_path = "sdoh_text.txt"
text = read_text_file(file_path)

if text:
    print("\nRunning Extraction...\n")

    text_chunks = chunk_text(text, max_tokens=2000)

    all_responses = [] 

    for i, chunk in enumerate(text_chunks):
        print(f"\nProcessing chunk {i + 1}/{len(text_chunks)}...\n")


        prompt = f"""
        You are an extraction model. Your task is to systematically scan the text and extract short phrases that match each of the 34 Social Determinants of Health (SDoH) categories. 

        ### **Instructions:**
        - **Extract text or short phrases from the text.** **Do NOT paraphrase, explain, or interpret.**
        - **Do NOT show categories that have no matching phrases.** **Only return categories that have a match.**
        - **All extracted phrases must be directly copied from the text.** 

        ### **Categories to Scan:**
        - Age, Origin, Ethnicity, Gender, Marital Status, Alcohol, Tobacco, Illicit Drugs, Caffeine, Stress, Psychological Concerns, Social Adversities, Nutrition, Physical Activity, Sleep, Sexual Behavior, Contraception, Treatment Adherence, Employment, Financial Strain, Food Insecurity, Housing Stability, Caregiving, Living Situation, Social Support, Safety, Transportation, Utilities, Medical Access, Insurance, Healthcare Technology, Erratic Care, Maintaining Care.  

        ### **STRICT Output Format (DO NOT DEVIATE)**:
        #  "category name ex:Age" : "exact phrase from text",
        #  "category name" : "exact phrase from text"
            ...
                    
        Now, extract from the following text:

        {chunk}
        """

        try:

            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0} 
            )

            if "message" in response and "content" in response["message"]:
                generated_text = response["message"]["content"]
            else:
                print(f"Unexpected response format in chunk {i + 1}: {response}")
                continue

            if DEBUG:
                print(f"\nDEBUG - Raw Model Response (Chunk {i+1}):\n{generated_text}\n")

            all_responses.append(f"Chunk {i+1}:\n{generated_text}\n")

        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

    print("Extracted Raw Responses Across All Chunks:\n")
    for response in all_responses:
        print(response)

    output_file = "sdoh_llm_raw_responses.txt"
    if all_responses:
        with open(output_file, "w", encoding="utf-8") as out:
            out.writelines(all_responses)
        print(f"Results saved to: {output_file}")
    else:
        print("No meaningful responses detected. File not saved.")
else:
    print("No text to process. Please check your input file.")