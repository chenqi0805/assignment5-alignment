import json
import os
from pathlib import Path
import pathlib
from typing import List, Tuple


def math_data_loading(data_dir: str = "data/MATH") -> Tuple[List[str], List[str]]:
    """
    Load MATH dataset from JSON files and extract problems and solutions.
    
    Args:
        data_dir: Path to the MATH dataset directory (default: "data/MATH")
        
    Returns:
        Tuple containing:
        - prompts: List of problem statements
        - output_strs: List of solution strings
    """
    prompts = []
    output_strs = []
    
    # Convert to Path object for easier handling
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Recursively find all JSON files in the directory
    json_files = list(data_path.rglob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract problem and solution fields
            if 'problem' in data and 'solution' in data:
                prompts.append(data['problem'])
                output_strs.append(data['solution'])
            else:
                print(f"Warning: Missing 'problem' or 'solution' field in {json_file}")
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"Loaded {len(prompts)} problem-solution pairs from {len(json_files)} files")
    
    return prompts, output_strs


if __name__ == "__main__":
    # Example usage
    prompts, solutions = math_data_loading((pathlib.Path(__file__).resolve().parent) / "../data/MATH")
    
    print(f"Total problems loaded: {len(prompts)}")
    print(f"Total solutions loaded: {len(solutions)}")
    
    # Display first example
    if prompts and solutions:
        print("\n--- First Example ---")
        print("Problem:")
        print(prompts[0])
        print("\nSolution:")
        print(solutions[0])
