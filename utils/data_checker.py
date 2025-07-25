import os
from pathlib import Path
from typing import Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataChecker:
    @staticmethod
    def check_pdf_directory(data_path: str) -> Dict[str, Any]:
        """Check PDF directory and files for potential issues"""
        path = Path(data_path)
        results = {
            "directory_exists": path.exists(),
            "is_directory": path.is_dir() if path.exists() else False,
            "pdf_files": [],
            "issues": []
        }
        
        if not results["directory_exists"]:
            results["issues"].append(f"Directory {data_path} does not exist")
            return results
            
        if not results["is_directory"]:
            results["issues"].append(f"{data_path} is not a directory")
            return results
            
        # Check PDF files with progress bar
        pdf_files = list(path.glob("*.pdf"))
        for file in tqdm(pdf_files, desc="Checking PDF files"):
            file_info = {
                "name": file.name,
                "size": file.stat().st_size,
                "readable": os.access(file, os.R_OK)
            }
            
            # Check for potential issues
            if file_info["size"] == 0:
                results["issues"].append(f"File {file.name} is empty")
            elif not file_info["readable"]:
                results["issues"].append(f"File {file.name} is not readable")
            
            results["pdf_files"].append(file_info)
            
        if not results["pdf_files"]:
            results["issues"].append(f"No PDF files found in {data_path}")
            
        return results

    @staticmethod
    def print_check_results(results: Dict[str, Any]):
        """Print data check results in a readable format"""
        print("\n=== Data Check Results ===")
        print(f"Directory exists: {results['directory_exists']}")
        print(f"Is directory: {results['is_directory']}")
        print(f"\nPDF Files found: {len(results['pdf_files'])}")
        
        if results["pdf_files"]:
            print("\nPDF Files:")
            for file in results["pdf_files"]:
                print(f"- {file['name']}: {file['size']} bytes, Readable: {file['readable']}")
        
        if results["issues"]:
            print("\nIssues found:")
            for issue in results["issues"]:
                print(f"! {issue}")
        else:
            print("\nNo issues found.")
