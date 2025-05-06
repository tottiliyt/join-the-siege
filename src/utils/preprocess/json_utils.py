"""
Utility functions for extracting text from JSON files.
"""
import json
import json5
from werkzeug.datastructures import FileStorage
import io

def extract_text_from_json(file: FileStorage) -> str:
    """
    Extract text from JSON files (.json, .json5) as a flat sequence of keys and values.
    
    Args:
        file (FileStorage): The uploaded JSON file
        
    Returns:
        str: Extracted text as a single string with alternating keys and values (lowercased)
    """
    try:
        file.seek(0)
        file_content = file.read().decode('utf-8')
        
        # Try standard JSON first
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            # If standard JSON fails, try JSON5 which is more lenient
            data = json5.loads(file_content)
        
        # Recursively extract keys and values from the JSON
        keys_and_values = []
        _extract_keys_and_values(data, keys_and_values)
        
        # Join all keys and values into a single string
        formatted_text = ' '.join(str(item) for item in keys_and_values)
        return formatted_text.lower()
    except Exception as e:
        # Optionally log the error here
        print(f"JSON extraction error: {str(e)}")
        return ""

def _extract_keys_and_values(data, result_list):
    """
    Recursively extract keys and values from a JSON object as a flat list.
    Keys and values are added in alternating order.
    
    Args:
        data: The JSON data (can be dict, list, or primitive value)
        result_list: List to collect keys and values
    """
    if isinstance(data, dict):
        for key, value in data.items():
            # Add the key
            result_list.append(key)
            
            if isinstance(value, (str, int, float, bool)):
                # Add the primitive value
                result_list.append(value)
            elif isinstance(value, (dict, list)):
                # Recurse into the complex type
                _extract_keys_and_values(value, result_list)
    
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (str, int, float, bool)):
                # Add primitive values directly
                result_list.append(item)
            elif isinstance(item, (dict, list)):
                # Recurse into the complex type
                _extract_keys_and_values(item, result_list)
    
    elif isinstance(data, (str, int, float, bool)):
        # Add primitive values directly
        result_list.append(data)
