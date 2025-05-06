"""
Utility for generating synthetic document data using OpenAI.
"""
import os
import re
from typing import Dict, List, Optional, Union

import openai
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
if not client:
    print("Warning: OPENAI_API_KEY not found. Synthetic data generation will not work.")

# Document types
DOCUMENT_TYPES = [
    "invoice", "bank_statement", "receipt", "contract", "lease_agreement",
    "drivers_licence", "passport", "resume", "transcript"
]

def generate_documents_batch(doc_type: str, count: int = 5, industry: Optional[str] = None) -> List[Dict[str, Union[str, bool]]]:
    """
    Generate multiple documents of the same type in a single API call.
    
    Args:
        doc_type: Type of document to generate
        count: Number of documents to generate (1-10)
        industry: Specific industry context (optional)
        
    Returns:
        List of dictionaries with generated document data
    """
    # Check for API key and limit batch size
    if not client:
        return [{"success": False, "error": "OpenAI API key not configured"} for _ in range(count)]
    
    count = min(count, 10)  # Limit batch size to 10
    industry_context = f" for the {industry} industry" if industry else ""
    
    # Create prompt for document generation
    prompt = f"""
    Generate {count} different examples of {doc_type}{industry_context} documents.
    
    Each document should:
    - Be realistic but contain fictional data
    - Have clear structural elements typical of a {doc_type}
    - Be separated by a clear delimiter like "---DOCUMENT SEPARATOR---"
    
    Please generate {count} completely different {doc_type} documents, each with unique content:
    """
    
    try:
        # Generate documents using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a document generation assistant that creates realistic synthetic documents for training AI systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        
        generated_text = response.choices[0].message.content.strip()
        documents = []
        
        # Try different separator patterns to split the text
        separator_patterns = [
            "---DOCUMENT SEPARATOR---",
            "---+\\s*DOCUMENT\\s*\\d*\\s*---+",
            "DOCUMENT\\s*\\d+\\s*:",
            "\\n\\n---+\\n\\n",
            "\\n\\n\\*\\*\\*+\\n\\n",
            "\\n\\nDOCUMENT\\s*\\d+\\n\\n"
        ]
        
        # Try each pattern until we get the expected number of documents
        for pattern in separator_patterns:
            if len(documents) == count:
                break
                
            docs = re.split(pattern, generated_text) if pattern != "---DOCUMENT SEPARATOR---" else generated_text.split(pattern)
            docs = [doc.strip() for doc in docs if doc.strip()]
            
            if len(docs) >= count:
                documents = docs[:count]
                break
                
        # Format results
        results = [
            {"success": True, "text": doc, "doc_type": doc_type, "industry": industry or "general"}
            for doc in documents
        ]
        
        # Fill any missing documents with error messages
        while len(results) < count:
            results.append({"success": False, "error": "Failed to generate enough documents", "doc_type": doc_type})
        
        return results
        
    except Exception as e:
        return [{"success": False, "error": f"Error: {str(e)}", "doc_type": doc_type} for _ in range(count)]
