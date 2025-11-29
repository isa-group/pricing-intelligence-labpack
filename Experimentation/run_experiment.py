import json
import os
import requests
import time

API_URL = "http://localhost:8086/chat"
INPUT_FILE = "instantiated_questions.json"
OUTPUT_FILE = "experiment_results_gpt_5_nano.json"

def load_results():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {OUTPUT_FILE}. Starting fresh.")
            return []
    return []

def save_results(results):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def run_experiment():
    print(f"Loading questions from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # Load existing results
    existing_results = load_results()
    # Map question text to result entry for easy lookup
    # We use the question text as a unique key for now (assuming unique questions)
    results_map = {entry['input']['question']: entry for entry in existing_results}
    
    # Process only the first 3 questions (as per original request)
    # You can remove the slice [:3] to process all questions
    to_process = questions
    
    print(f"Processing {len(to_process)} questions with checkpointing...")
    
    for i, item in enumerate(to_process):
        question_text = item['question']
        
        # Check if already processed successfully
        if question_text in results_map:
            existing_entry = results_map[question_text]
            # Check if valid: has api_response and no error
            if existing_entry.get('api_response') and not existing_entry.get('error'):
                print(f"[{i+1}/{len(to_process)}] Skipping (already done): {question_text[:50]}...")
                continue
            else:
                print(f"[{i+1}/{len(to_process)}] Retrying (previous error/empty): {question_text[:50]}...")
        else:
            print(f"[{i+1}/{len(to_process)}] Asking: {question_text[:50]}...")
        
        pricing_paths = item.get('pricing_paths', [])
        pricing_yamls = []
        for path in pricing_paths:
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    pricing_yamls.append(content)
            except Exception as e:
                print(f"  Error reading pricing file {path}: {e}")
        
        payload = {
            "question": question_text,
            "pricing_yamls": pricing_yamls
        }
        
        result_entry = None
        try:
            start_time = time.time()
            print("  Sending request (timeout=900s)...")
            response = requests.post(API_URL, json=payload, timeout=900)
            response.raise_for_status()
            data = response.json()
            duration = time.time() - start_time
            
            print(f"  Success ({duration:.2f}s)")
            
            result_entry = {
                "input": item,
                "api_response": data,
                "duration_seconds": duration
            }
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            print(f"  API Request failed after {duration:.2f}s: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"  Response: {e.response.text}")
            
            result_entry = {
                "input": item,
                "error": str(e),
                "duration_seconds": duration
            }
        
        # Update results map and save immediately
        results_map[question_text] = result_entry
        
        # Convert map back to list, preserving order of 'questions' if possible or just values
        # To preserve order of the original questions list:
        current_results_list = []
        for q in questions: # Iterate over all questions to maintain order/completeness
            q_text = q['question']
            if q_text in results_map:
                current_results_list.append(results_map[q_text])
            # If we haven't processed it yet, it's not in the results file.
            # But wait, we only want to save what we have processed.
            # Let's just save the values of the map.
        
        # Actually, let's just save the list of results we have so far.
        # The map is keyed by question text.
        save_results(list(results_map.values()))

    print("Done.")

if __name__ == "__main__":
    run_experiment()
