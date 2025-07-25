# financial_bot.py
# Comprehensive implementation of a regulatory-compliant financial advisor chatbot
# Combines LayoutLMv3 for document processing, FAISS for vector storage,
# and FinGPT for response generation with RBI compliance checking

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import pdfplumber
import faiss
import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel
)
from PIL import Image
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PDF_DIR = "data\pdfs"
MODEL_DIR = "models/fingpt/"
VECTOR_DB_PATH = "vector_store/financial_data.pkl"
EMBEDDING_DIMENSION = 768  # LayoutLMv3 base embedding dimension

@dataclass
class FinancialData:
    """Structured financial data extracted from documents"""
    interest_rates: List[Dict[str, Union[float, str]]]
    fees: Dict[str, Union[float, Dict]]
    terms: str
    metadata: Dict[str, Any]
    timestamp: datetime = datetime.now()

@dataclass
class FinancialRecommendation:
    """Structured financial recommendation"""
    recommendation: str
    calculation: Dict[str, str]
    risk_analysis: Dict[str, Any]
    compliance: Dict[str, str]
    numerical_breakdown: pd.DataFrame

class VectorStore:
    """Hybrid vector store combining FAISS and structured data"""
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.structured_data = []
        self.embeddings = []
        
    def add_document(self, 
                    embedding: np.ndarray, 
                    financial_data: FinancialData) -> None:
        """Add document embedding and structured data to storage"""
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embedding.shape[1]}")
            
        self.index.add(embedding)
        self.structured_data.append(financial_data)
        self.embeddings.append(embedding)
        
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5) -> List[Tuple[int, float, FinancialData]]:
        """Search for similar documents and return their data"""
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((idx, distance, self.structured_data[idx]))
        return results
    
    def save(self, path: str) -> None:
        """Save vector store to disk"""
        store_data = {
            'index': faiss.serialize_index(self.index),
            'structured_data': self.structured_data,
            'embeddings': self.embeddings
        }
        with open(path, 'wb') as f:
            pickle.dump(store_data, f)
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """Load vector store from disk"""
        with open(path, 'rb') as f:
            store_data = pickle.load(f)
        
        store = cls()
        store.index = faiss.deserialize_index(store_data['index'])
        store.structured_data = store_data['structured_data']
        store.embeddings = store_data['embeddings']
        return store

class DocumentProcessor:
    """Enhanced document processor using LayoutLMv3"""
    
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            torch_dtype=torch.float16
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
        self.embedding_model = AutoModel.from_pretrained(
            "microsoft/layoutlmv3-base",
            torch_dtype=torch.float16
        ).to(device)
        
        self.device = device
        
    def process_directory(self, 
                         directory: str, 
                         vector_store: VectorStore) -> None:
        """Process all PDFs in a directory and store in vector store"""
        pdf_files = list(Path(directory).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        with ThreadPoolExecutor() as executor:
            for pdf_file in pdf_files:
                try:
                    financial_data = self.extract_financial_data(str(pdf_file))
                    embedding = self.generate_embedding(financial_data)
                    vector_store.add_document(embedding, financial_data)
                    logger.info(f"Processed and stored {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Error processing {pdf_file.name}: {str(e)}")
    
    def extract_financial_data(self, pdf_path: str) -> FinancialData:
        """Enhanced financial data extraction from PDFs"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                financial_data = {
                    "interest_rates": [],
                    "fees": {},
                    "terms": "",
                    "metadata": {
                        "source": pdf_path,
                        "pages": len(pdf.pages),
                        "creation_date": pdf.metadata.get('CreationDate', '')
                    }
                }
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}/{len(pdf.pages)}")
                    
                    # Extract and process tables
                    tables = page.extract_tables()
                    for table in tables:
                        self._process_table(table, financial_data)
                    
                    # Extract and process text
                    text = page.extract_text()
                    self._process_text(text, financial_data)
                    
                    # Extract and process images if present
                    if page.images:
                        self._process_images(page.images, financial_data)
                
                return FinancialData(**financial_data)
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def _process_table(self, 
                      table: List[List[str]], 
                      data: Dict) -> None:
        """Enhanced table processing with pattern matching"""
        for row in table:
            # Process interest rates
            if any("interest" in str(cell).lower() for cell in row):
                rate_info = self._extract_rate_information(row)
                if rate_info:
                    data["interest_rates"].append(rate_info)
            
            # Process fees
            elif any("fee" in str(cell).lower() for cell in row):
                fee_info = self._extract_fee_information(row)
                if fee_info:
                    data["fees"].update(fee_info)

    def _extract_rate_information(self, 
                                row: List[str]) -> Optional[Dict]:
        """Extract detailed interest rate information"""
        rate_pattern = r'(\d+\.?\d*)%'
        for cell in row:
            if not isinstance(cell, str):
                continue
                
            rate_match = re.search(rate_pattern, cell)
            if rate_match:
                return {
                    "rate": float(rate_match.group(1)),
                    "type": self._determine_rate_type(cell),
                    "conditions": self._extract_conditions(cell),
                    "applicable_from": self._extract_date(cell)
                }
        return None

    def _determine_rate_type(self, text: str) -> str:
        """Determine the type of interest rate"""
        text = text.lower()
        if "fixed" in text:
            return "fixed"
        elif "floating" in text:
            return "floating"
        elif "base" in text:
            return "base"
        return "unspecified"

    def generate_embedding(self, 
                         financial_data: FinancialData) -> np.ndarray:
        """Generate document embedding using LayoutLMv3"""
        # Combine relevant text for embedding
        text = f"{financial_data.terms} {' '.join(str(rate) for rate in financial_data.interest_rates)}"
        
        # Process text through LayoutLMv3
        inputs = self.processor(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        
        # Use [CLS] token embedding as document embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding

class FinancialCalculator:
    """Enhanced financial calculations engine"""
    
    def calculate_emi(self, 
                     principal: float, 
                     rate: float, 
                     tenure: int) -> Dict[str, float]:
        """
        Calculate detailed EMI breakdown
        Args:
            principal: Loan amount
            rate: Annual interest rate (percentage)
            tenure: Loan tenure in months
        """
        monthly_rate = rate / (12 * 100)
        emi = principal * monthly_rate * pow(1 + monthly_rate, tenure) / (pow(1 + monthly_rate, tenure) - 1)
        
        # Calculate additional metrics
        total_payment = emi * tenure
        total_interest = total_payment - principal
        interest_ratio = (total_interest / principal) * 100
        
        return {
            "monthly_emi": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "interest_ratio": round(interest_ratio, 2)
        }

    def calculate_prepayment_impact(self, 
                                  current_emi: float,
                                  prepayment_amount: float,
                                  remaining_tenure: int,
                                  interest_rate: float) -> Dict[str, float]:
        """Calculate impact of prepayment on loan"""
        remaining_principal = (current_emi * remaining_tenure) / (1 + (interest_rate / (12 * 100)))
        new_principal = remaining_principal - prepayment_amount
        
        new_emi = self.calculate_emi(new_principal, interest_rate, remaining_tenure)["monthly_emi"]
        
        savings = {
            "emi_reduction": round(current_emi - new_emi, 2),
            "total_interest_saved": round((current_emi - new_emi) * remaining_tenure, 2),
            "percentage_saved": round(((current_emi - new_emi) / current_emi) * 100, 2)
        }
        return savings

    def calculate_risk_score(self, 
                           loan_type: str, 
                           income: float,
                           credit_score: Optional[int] = None,
                           existing_loans: Optional[float] = None) -> Dict[str, Union[int, str]]:
        """
        Enhanced risk score calculation
        Returns detailed risk assessment
        """
        base_score = 5
        risk_factors = []
        
        # Loan type adjustment
        loan_type_scores = {
            "home": -1,
            "personal": 2,
            "business": 1,
            "education": -0.5
        }
        base_score += loan_type_scores.get(loan_type.lower(), 0)
        
        # Income adjustment
        income_monthly = income / 12
        if income_monthly < 30000:
            base_score += 2
            risk_factors.append("Low income")
        elif income_monthly > 150000:
            base_score -= 1
            risk_factors.append("High income - favorable")
            
        # Credit score adjustment
        if credit_score:
            if credit_score > 750:
                base_score -= 1
                risk_factors.append("Excellent credit score")
            elif credit_score < 650:
                base_score += 2
                risk_factors.append("Poor credit score")
                
        # Existing loan burden
        if existing_loans:
            debt_to_income = (existing_loans * 12) / income
            if debt_to_income > 0.5:
                base_score += 2
                risk_factors.append("High debt burden")
            
        final_score = max(1, min(10, round(base_score)))
        
        return {
            "score": final_score,
            "risk_level": self._risk_level_description(final_score),
            "risk_factors": risk_factors,
            "recommendations": self._generate_risk_recommendations(final_score, risk_factors)
        }

    def _risk_level_description(self, score: int) -> str:
        """Generate risk level description based on score"""
        if score <= 3:
            return "Low Risk"
        elif score <= 6:
            return "Moderate Risk"
        else:
            return "High Risk"

    def _generate_risk_recommendations(self, 
                                     score: int, 
                                     factors: List[str]) -> List[str]:
        """Generate specific recommendations based on risk assessment"""
        recommendations = []
        if score > 6:
            recommendations.append("Consider additional collateral or guarantor")
            recommendations.append("Higher interest rate may be applicable")
        if "Low income" in factors:
            recommendations.append("Explore government subsidy schemes")
        if "High debt burden" in factors:
            recommendations.append("Debt consolidation may be beneficial")
        return recommendations

class ComplianceChecker:
    """Enhanced RBI compliance checker"""
    
    def __init__(self):
        self.rbi_guidelines = self._load_guidelines()
        self.last_update = datetime.now()
        
    def _load_guidelines(self) -> Dict:
        """Load comprehensive RBI guidelines"""
        return {
            "loan_to_value_ratio": {
                "home": 0.80,  # 80% for home loans
                "personal": 0.70,
                "business": 0.75
            },
            "income_ratio_limits": {
                "min_income": 25000,  # Monthly minimum income 
                "max_emi_ratio": 0.50,  # Maximum 50% of income as EMI
                "total_exposure_ratio": 0.70  # Total debt exposure limit
            },
            "tenure_limits": {
                "home": 360,  # 30 years
                "personal": 60,
                "business": 84
            },
            "documentation_requirements": {
                "home": [
                    "income_proof",
                    "property_documents", 
                    "identity_proof",
                    "bank_statements"
                ],
                "personal": [
                    "income_proof",
                    "identity_proof", 
                    "bank_statements"
                ]
            },
            "interest_rate_guidelines": {
                "base_rate_linked": True,
                "transparency_required": True,
                "max_spread": 5.0  # Maximum spread over base rate
            },
            "special_categories": {
                "women": dict(additional_ltv=0.05),
                "senior_citizens": dict(rate_concession=0.25),
                "government_employees": dict(documentation_relaxation=True)
            }
        }

    def validate_loan_application(self,
                                loan_type: str,
                                amount: float,
                                income: float,
                                tenure: int,
                                existing_emi: float = 0,
                                property_value: Optional[float] = None,
                                category: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of loan application against RBI guidelines
        Returns detailed compliance report
        """
        compliance_report = {
            "is_compliant": True,
            "checks_performed": [],
            "violations": [],
            "warnings": [],
            "recommendations": []
        }

        # Check loan-to-value ratio for secured loans
        if property_value and loan_type == "home":
            base_ltv = self.rbi_guidelines["loan_to_value_ratio"][loan_type]
            if category in self.rbi_guidelines["special_categories"]:
                base_ltv += self.rbi_guidelines["special_categories"][category].get("additional_ltv", 0)
            
            ltv_ratio = amount / property_value
            compliance_report["checks_performed"].append({
                "check": "Loan to Value Ratio",
                "actual": f"{ltv_ratio:.2%}",
                "limit": f"{base_ltv:.2%}"
            })
            
            if ltv_ratio > base_ltv:
                compliance_report["is_compliant"] = False
                compliance_report["violations"].append(
                    f"LTV ratio {ltv_ratio:.2%} exceeds maximum allowed {base_ltv:.2%}"
                )

        # Check income and EMI ratios
        monthly_income = income / 12
        total_emi_ratio = (existing_emi + self._calculate_approximate_emi(amount, tenure)) / monthly_income
        
        compliance_report["checks_performed"].append({
            "check": "EMI to Income Ratio",
            "actual": f"{total_emi_ratio:.2%}",
            "limit": f"{self.rbi_guidelines['income_ratio_limits']['max_emi_ratio']:.2%}"
        })

        if total_emi_ratio > self.rbi_guidelines["income_ratio_limits"]["max_emi_ratio"]:
            compliance_report["is_compliant"] = False
            compliance_report["violations"].append(
                "Total EMI burden exceeds maximum allowed percentage of income"
            )

        # Check tenure limits
        max_tenure = self.rbi_guidelines["tenure_limits"].get(loan_type, 0)
        if tenure > max_tenure:
            compliance_report["is_compliant"] = False
            compliance_report["violations"].append(
                f"Requested tenure {tenure} months exceeds maximum allowed {max_tenure} months"
            )

        # Add recommendations based on special categories
        if category in self.rbi_guidelines["special_categories"]:
            benefits = self.rbi_guidelines["special_categories"][category]
            compliance_report["recommendations"].append({
                "category": category,
                "benefits": benefits
            })

        return compliance_report

    def _calculate_approximate_emi(self,
                                 amount: float,
                                 tenure: int,
                                 rate: float = 0.10) -> float:
        """Calculate approximate EMI for compliance checking"""
        monthly_rate = rate / 12
        return amount * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)

class ChatbotEngine:
    """Enhanced chatbot engine with sophisticated response generation"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.calculator = FinancialCalculator()
        self.compliance_checker = ComplianceChecker()
        self.vector_store = self._initialize_vector_store()
        self.model = self._initialize_language_model()
        self.tokenizer = AutoTokenizer.from_pretrained("FinGPT/FingPT-Mistral-7B-LoRA")

    def _initialize_vector_store(self) -> VectorStore:
        """Initialize or load existing vector store"""
        if os.path.exists(VECTOR_DB_PATH):
            try:
                return VectorStore.load(VECTOR_DB_PATH)
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                return VectorStore()
        return VectorStore()

    def _initialize_language_model(self) -> AutoModelForCausalLM:
        """Initialize FinGPT model with optimizations"""
        try:
            return AutoModelForCausalLM.from_pretrained(
                "FinGPT/FingPT-Mistral-7B-LoRA",
                load_in_4bit=True,
                device_map="auto"
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def process_query(self,
                     query: str,
                     user_data: Dict[str, Any]) -> FinancialRecommendation:
        """
        Process user query and generate comprehensive recommendation
        Args:
            query: User's financial query
            user_data: Dictionary containing user's financial information
        Returns:
            Structured financial recommendation
        """
        # Extract query embedding for relevant document retrieval
        query_embedding = self.document_processor.generate_embedding(
            FinancialData([], {}, query, {})
        )
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(query_embedding, k=3)
        
        # Extract loan details from query
        loan_details = self._extract_loan_details(query)
        
        # Perform financial calculations
        calculations = self._perform_calculations(loan_details, user_data)
        
        # Check compliance
        compliance_report = self.compliance_checker.validate_loan_application(
            loan_type=loan_details["type"],
            amount=loan_details["amount"],
            income=user_data["monthly_income"],
            tenure=loan_details["tenure"],
            existing_emi=user_data.get("existing_emi", 0),
            property_value=user_data.get("property_value")
        )
        
        # Generate risk analysis
        risk_analysis = self.calculator.calculate_risk_score(
            loan_type=loan_details["type"],
            income=user_data["monthly_income"],
            credit_score=user_data.get("credit_score"),
            existing_loans=user_data.get("existing_emi", 0) * 12
        )
        
        # Create numerical breakdown
        numerical_breakdown = self._create_numerical_breakdown(calculations, loan_details)
        
        # Generate final recommendation
        recommendation = self._generate_recommendation(
            loan_details,
            calculations,
            compliance_report,
            risk_analysis,
            relevant_docs
        )
        
        return FinancialRecommendation(
            recommendation=recommendation,
            calculation=calculations,
            risk_analysis=risk_analysis,
            compliance=compliance_report,
            numerical_breakdown=numerical_breakdown
        )

    def _create_numerical_breakdown(self,
                                  calculations: Dict,
                                  loan_details: Dict) -> pd.DataFrame:
        """Create detailed numerical breakdown of the loan analysis"""
        data = {
            'Metric': [
                'Monthly EMI',
                'Total Interest',
                'Total Payment',
                'Interest Ratio',
                'Loan Amount',
                'Tenure (Years)'
            ],
            'Value': [
                f"₹{calculations['monthly_emi']:,.2f}",
                f"₹{calculations['total_interest']:,.2f}",
                f"₹{calculations['total_payment']:,.2f}",
                f"{calculations['interest_ratio']:.2f}%",
                f"₹{loan_details['amount']:,.2f}",
                f"{loan_details['tenure']/12:.1f}"
            ]
        }
        return pd.DataFrame(data)

    def _generate_recommendation(self,
                               loan_details: Dict,
                               calculations: Dict,
                               compliance_report: Dict,
                               risk_analysis: Dict,
                               relevant_docs: List) -> str:
        """Generate comprehensive recommendation text"""
        if not compliance_report["is_compliant"]:
            return self._generate_non_compliant_response(compliance_report)
        
        # Build recommendation using list for cleaner formatting
        parts = []
        
        # Main recommendation
        parts.append(
            f"Based on your profile and current market conditions, "
            f"I recommend proceeding with the {loan_details['type']} loan "
            f"with the following considerations:"
        )
        
        # Financial insights
        monthly_income = loan_details.get("monthly_income", 0)
        if monthly_income:
            emi_ratio = calculations["monthly_emi"] / monthly_income
            parts.append(
                f"\nThe monthly EMI of ₹{calculations['monthly_emi']:,.2f} "
                f"would constitute {emi_ratio:.1%} of your monthly income."
            )
        
        # Risk assessment
        parts.append(
            f"\nRisk Assessment: {risk_analysis['risk_level']} "
            f"(Score: {risk_analysis['score']}/10)"
        )
        
        # Add risk recommendations
        if risk_analysis['recommendations']:
            parts.append("\nRisk Mitigation Suggestions:")
            parts.extend(f"- {rec}" for rec in risk_analysis['recommendations'])
        
        # Add relevant policy information
        if relevant_docs:
            parts.append("\nRelevant Policy Information:")
            for _, _, doc in relevant_docs[:2]:
                if doc.terms:
                    parts.append(f"- {doc.terms[:200]}...")
        
        return "\n".join(parts)

    def _generate_non_compliant_response(self,
                                       compliance_report: Dict) -> str:
        """Generate response for non-compliant loan applications"""
        response_parts = [
            "I regret to inform you that the loan application does not meet all regulatory requirements:",
            "\nSpecific Issues:"
        ]
        
        for violation in compliance_report["violations"]:
            response_parts.append(f"- {violation}")
            
        if compliance_report["recommendations"]:
            response_parts.append("\nRecommended Actions:")
            for rec in compliance_report["recommendations"]:
                response_parts.append(f"- {rec}")
                
        return "\n".join(response_parts)

# Example usage demonstration
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = ChatbotEngine()
    
    # Example query and user data
    query = "As a ₹25L home loan seeker, should I choose 8.5% fixed or 7.9% floating rate?"
    user_data = {
        "monthly_income": 150000,  # ₹1.5L per month
        "credit_score": 750,
        "existing_emi": 20000,
        "property_value": 3000000
    }
    
    try:
        # Process query and get recommendation
        result = chatbot.process_query(query, user_data)
        
        # Display results
        print("\nRecommendation Summary:")
        print("=" * 50)
        print(result.recommendation)
        
        print("\nNumerical Breakdown:")
        print("=" * 50)
        print(result.numerical_breakdown.to_string(index=False))
        
        print("\nRisk Analysis:")
        print("=" * 50)
        for key, value in result.risk_analysis.items():
            print(f"{key}: {value}")
        
        print("\nCompliance Status:")
        print("=" * 50)
        print(f"Compliant: {result.compliance['is_compliant']}")
        if not result.compliance['is_compliant']:
            print("Violations:")
            for violation in result.compliance['violations']:
                print(f"- {violation}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")