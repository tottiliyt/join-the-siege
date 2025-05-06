"""
Utility functions for extracting text from Excel spreadsheets.
"""
import pandas as pd
import openpyxl
from werkzeug.datastructures import FileStorage
import io

def extract_text_from_excel(file: FileStorage) -> str:
    """
    Extract text from Excel spreadsheets (.xlsx, .xls, .csv).
    
    Args:
        file (FileStorage): The uploaded Excel file
        
    Returns:
        str: Extracted text as a single string (lowercased)
    """
    try:
        file.seek(0)
        filename = file.filename.lower() if file.filename else ''
        
        # Special handling for CSV files
        if filename.endswith('.csv'):
            # Use pandas to read the CSV file
            df = pd.read_csv(file)
        else:
            # Use pandas to read the Excel file
            df = pd.read_excel(file)
        
        # Convert all data to string and join
        text_parts = []
        
        # Add column names
        text_parts.extend(str(col) for col in df.columns)
        
        # Add cell values
        for _, row in df.iterrows():
            for cell in row:
                # Skip NaN values
                if pd.notna(cell):
                    text_parts.append(str(cell))
        
        return ' '.join(text_parts).lower()
    except Exception as e:
        # Try alternative approach with openpyxl for .xlsx files
        try:
            file.seek(0)
            workbook = openpyxl.load_workbook(file)
            
            text_parts = []
            for sheet in workbook:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value:
                            text_parts.append(str(cell.value))
            
            return ' '.join(text_parts).lower()
        except Exception as e:
            # Optionally log the error here
            return ""
