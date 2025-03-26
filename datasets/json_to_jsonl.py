#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import List, Union, Dict

def transform_json_object(obj: Dict) -> Dict:
    """
    Transform a JSON object to match the required format.
    
    Args:
        obj (Dict): Original JSON object
        
    Returns:
        Dict: Transformed JSON object
    """
    # Create new object with required format
    transformed = {
        "name": obj.get("problem_name", ""),
        "split": "valid",
        "informal_prefix": obj.get("informal_statement", ""),
        "formal_statement": "",  # Empty as specified
        "goal": "",  # Empty if not provided
        "header": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    }
    
    return transformed

def process_json_file(file_path: Path) -> List[Dict]:
    """
    Read and parse a JSON file, handling both single objects and arrays.
    
    Args:
        file_path (Path): Path to the JSON file
        
    Returns:
        List[Dict]: List of transformed JSON objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        # Handle both single objects and arrays
        if isinstance(content, list):
            return [transform_json_object(obj) for obj in content]
        else:
            return [transform_json_object(content)]
            
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []

def convert_to_jsonl(input_dir: Union[str, Path], output_file: Union[str, Path]) -> None:
    """
    Convert all JSON files in a directory to a single JSONL file with specific formatting.
    
    Args:
        input_dir (Union[str, Path]): Directory containing JSON files
        output_file (Union[str, Path]): Path to output JSONL file
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.is_dir():
        raise ValueError(f"Input path {input_dir} is not a directory")
        
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for json_file in input_path.glob('*.json'):
            print(f"Processing {json_file}")
            json_objects = process_json_file(json_file)
            
            for obj in json_objects:
                # Write each transformed object as a single line
                out_file.write(json.dumps(obj, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert multiple JSON files to a single JSONL file with specific formatting')
    parser.add_argument('input_dir', help='Directory containing JSON files')
    parser.add_argument('output_file', help='Path to output JSONL file')
    
    args = parser.parse_args()
    
    try:
        convert_to_jsonl(args.input_dir, args.output_file)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()