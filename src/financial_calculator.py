from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FinancialMetrics:
    debt_to_income: float
    savings_ratio: float
    risk_score: float
    affordability_index: float

class FinancialCalculator:
    def calculate_loan_metrics(self, amount: float, interest: float, tenure: int, income: float) -> Dict[str, float]:
        """Calculate comprehensive loan metrics"""
        monthly_rate = interest / (12 * 100)
        emi = amount * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)
        
        return {
            'emi': round(emi, 2),
            'total_interest': round(emi * tenure - amount, 2),
            'loan_to_income': round(amount / (income * tenure), 2),
            'monthly_burden': round(emi / income * 100, 2)
        }
    
    def calculate_insurance_needs(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate insurance coverage requirements"""
        annual_income = profile.get('annual_income', 0)
        age = profile.get('age', 0)
        dependents = profile.get('dependents', 0)
        existing_coverage = profile.get('existing_coverage', 0)
        
        # Human Capital Value approach
        working_years = min(60 - age, 35)  # Consider working till 60 or next 35 years
        income_growth = 0.06  # 6% annual income growth
        inflation = 0.04  # 4% inflation
        real_growth = (1 + income_growth) / (1 + inflation) - 1
        
        future_income = annual_income * (1 - (1 + real_growth)**working_years) / (1 - (1 + real_growth))
        
        # Add dependent-specific needs
        dependent_needs = dependents * annual_income * 5  # 5 years of income per dependent
        
        recommended_coverage = future_income + dependent_needs - existing_coverage
        
        return {
            'recommended_coverage': round(recommended_coverage, -5),  # Round to nearest lakh
            'income_replacement_value': round(future_income, -5),
            'dependent_needs': round(dependent_needs, -5),
            'coverage_gap': round(recommended_coverage - existing_coverage, -5)
        }