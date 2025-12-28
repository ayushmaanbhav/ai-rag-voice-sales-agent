"""
Customer profile management and segmentation.

Handles mock customer data for demo and segment classification (P1-P4).
"""
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CustomerProfile:
    """Customer profile data model."""

    id: str
    name: str
    language: str  # hi, ta, te, kn, ml, en
    segment: str  # high_value, trust_seeker, shakti, young_professional
    current_provider: str  # muthoot, manappuram, iifl, other, none
    estimated_outstanding: int  # Amount in Rs (0 if no active loan)
    estimated_rate: float  # Current interest rate % (0 if no active loan)
    kotak_products: list[str]  # ['savings', '811', 'credit_card', 'fd']
    city: str
    tier: int  # 1 or 2
    occupation: str  # msme, salaried, professional, homemaker, farmer
    age_group: str  # 21-30, 31-40, 41-50, 50+
    gender: str  # male, female

    # Loan status attributes
    has_active_loan: bool = True  # Whether customer has an active gold loan
    loan_status: str = "active"  # active, closed, never
    previous_providers: list[str] = None  # List of providers they've used before
    loan_tenure_months: int = 0  # How many months into current loan

    def to_dict(self) -> dict:
        return asdict(self)

    def calculate_savings(self, kotak_rate: float = 10.0) -> int:
        """Calculate annual savings by switching to Kotak."""
        current_interest = self.estimated_outstanding * (self.estimated_rate / 100)
        kotak_interest = self.estimated_outstanding * (kotak_rate / 100)
        return int(current_interest - kotak_interest)

    def get_segment_display(self) -> str:
        """Get human-readable segment name."""
        segments = {
            "high_value": "High-Value MSME",
            "trust_seeker": "Trust-Seeker",
            "shakti": "Shakti (Women)",
            "young_professional": "Young Professional",
        }
        return segments.get(self.segment, self.segment)

    def get_conversation_approach(self) -> str:
        """
        Get the recommended conversation approach based on loan status.

        Returns:
            approach: "balance_transfer" | "new_loan" | "reactivation"
        """
        if self.has_active_loan and self.loan_status == "active":
            return "balance_transfer"
        elif self.loan_status == "closed":
            return "reactivation"  # Previously had loan, may need again
        else:
            return "new_loan"  # Never had gold loan

    def get_loan_context(self) -> str:
        """
        Get context string about customer's loan status for agent guidance.

        IMPORTANT: These are probability based attributes from our data.
        Agent should NOT presume or directly state knowledge of customer's loan.
        Instead, use smart questions to confirm and guide conversation.

        NOTE: Avoid symbols like hyphen, use words. TTS reads symbols literally.
        """
        if self.has_active_loan and self.loan_status == "active":
            return (
                f"LIKELIHOOD: High probability of active gold loan (data suggests {self.current_provider.title()}).\n"
                f"APPROACH: Ask smartly. Example: Kya aapne gold loan liya hai? or Aapka gold loan kahan se hai?\n"
                f"If confirmed: Discuss balance transfer benefits, bridge loan option, 40 to 50 percent savings.\n"
                f"If denied: Pivot to new gold loan offer with competitive rates."
            )
        elif self.loan_status == "closed":
            prev = ", ".join(self.previous_providers) if self.previous_providers else "NBFCs"
            return (
                f"LIKELIHOOD: Previously had gold loan (possibly with {prev}), currently closed.\n"
                f"APPROACH: Ask about current financial needs. Example: Kya aapko abhi kisi fund ki zaroorat hai?\n"
                f"If yes: Offer Kotak gold loan with better rates than their previous experience.\n"
                f"If no: Thank them, mention Kotak for future needs."
            )
        else:
            return (
                f"LIKELIHOOD: No prior gold loan history detected.\n"
                f"APPROACH: Ask if they have gold jewelry and explore if they need funds.\n"
                f"Explain gold loan concept briefly. Quick funds against gold at low interest.\n"
                f"Highlight: 9 to 12 percent rate, same day disbursement, bank level security."
            )


class CustomerDB:
    """In-memory customer database for demo."""

    def __init__(self):
        self.customers: dict[str, CustomerProfile] = {}
        self._load_mock_data()

    def _load_mock_data(self):
        """Load mock customer data."""
        mock_customers = [
            # P1: High-Value Switchers (MSME owners)
            CustomerProfile(
                id="C001",
                name="Rajesh Kumar",
                language="hi",
                segment="high_value",
                current_provider="muthoot",
                estimated_outstanding=800000,
                estimated_rate=18,
                kotak_products=["savings", "credit_card"],
                city="Mumbai",
                tier=1,
                occupation="msme",
                age_group="41-50",
                gender="male",
                has_active_loan=True,
                loan_status="active",
                previous_providers=["iifl"],
                loan_tenure_months=8,
            ),
            CustomerProfile(
                id="C002",
                name="Suresh Patel",
                language="hi",
                segment="high_value",
                current_provider="manappuram",
                estimated_outstanding=1200000,
                estimated_rate=20,
                kotak_products=["savings", "fd", "credit_card"],
                city="Ahmedabad",
                tier=1,
                occupation="msme",
                age_group="31-40",
                gender="male",
                has_active_loan=True,
                loan_status="active",
                previous_providers=["muthoot", "iifl"],
                loan_tenure_months=14,
            ),
            # P2: Trust-Seekers (Safety-focused)
            CustomerProfile(
                id="C003",
                name="Venkat Rao",
                language="te",
                segment="trust_seeker",
                current_provider="iifl",
                estimated_outstanding=500000,
                estimated_rate=20,
                kotak_products=["savings", "fd"],
                city="Hyderabad",
                tier=1,
                occupation="professional",
                age_group="41-50",
                gender="male",
                has_active_loan=True,
                loan_status="active",
                previous_providers=["muthoot"],
                loan_tenure_months=6,
            ),
            CustomerProfile(
                id="C004",
                name="Ramamurthy Iyer",
                language="ta",
                segment="trust_seeker",
                current_provider="none",
                estimated_outstanding=0,
                estimated_rate=0,
                kotak_products=["savings"],
                city="Chennai",
                tier=1,
                occupation="salaried",
                age_group="50+",
                gender="male",
                has_active_loan=False,
                loan_status="closed",  # Previously had loan, now closed
                previous_providers=["muthoot"],
                loan_tenure_months=0,
            ),
            # P3: Shakti - Women Entrepreneurs
            CustomerProfile(
                id="C005",
                name="Lakshmi Devi",
                language="ta",
                segment="shakti",
                current_provider="manappuram",
                estimated_outstanding=200000,
                estimated_rate=22,
                kotak_products=["811"],
                city="Chennai",
                tier=1,
                occupation="homemaker",
                age_group="31-40",
                gender="female",
                has_active_loan=True,
                loan_status="active",
                previous_providers=[],
                loan_tenure_months=3,
            ),
            CustomerProfile(
                id="C006",
                name="Anitha Reddy",
                language="te",
                segment="shakti",
                current_provider="muthoot",
                estimated_outstanding=350000,
                estimated_rate=19,
                kotak_products=["savings", "811"],
                city="Hyderabad",
                tier=1,
                occupation="msme",
                age_group="31-40",
                gender="female",
                has_active_loan=True,
                loan_status="active",
                previous_providers=["manappuram"],
                loan_tenure_months=10,
            ),
            CustomerProfile(
                id="C007",
                name="Kavitha Nair",
                language="ml",
                segment="shakti",
                current_provider="none",
                estimated_outstanding=0,
                estimated_rate=0,
                kotak_products=["811"],
                city="Kochi",
                tier=2,
                occupation="homemaker",
                age_group="41-50",
                gender="female",
                has_active_loan=False,
                loan_status="never",  # Never had gold loan
                previous_providers=[],
                loan_tenure_months=0,
            ),
            # P4: Young Professionals
            CustomerProfile(
                id="C008",
                name="Arjun Sharma",
                language="hi",
                segment="young_professional",
                current_provider="iifl",
                estimated_outstanding=150000,
                estimated_rate=24,
                kotak_products=["811", "credit_card"],
                city="Bangalore",
                tier=1,
                occupation="salaried",
                age_group="21-30",
                gender="male",
                has_active_loan=True,
                loan_status="active",
                previous_providers=[],
                loan_tenure_months=2,
            ),
            CustomerProfile(
                id="C009",
                name="Priya Menon",
                language="en",
                segment="young_professional",
                current_provider="none",
                estimated_outstanding=0,
                estimated_rate=0,
                kotak_products=["811"],
                city="Pune",
                tier=1,
                occupation="professional",
                age_group="21-30",
                gender="female",
                has_active_loan=False,
                loan_status="never",  # Never had gold loan - potential new customer
                previous_providers=[],
                loan_tenure_months=0,
            ),
            # Additional diverse profiles
            CustomerProfile(
                id="C010",
                name="Manjunath Gowda",
                language="kn",
                segment="high_value",
                current_provider="iifl",
                estimated_outstanding=600000,
                estimated_rate=19,
                kotak_products=["savings", "credit_card"],
                city="Bangalore",
                tier=1,
                occupation="msme",
                age_group="41-50",
                gender="male",
                has_active_loan=True,
                loan_status="active",
                previous_providers=["muthoot"],
                loan_tenure_months=12,
            ),
            CustomerProfile(
                id="C011",
                name="Deepa Krishnan",
                language="ml",
                segment="trust_seeker",
                current_provider="muthoot",
                estimated_outstanding=450000,
                estimated_rate=18,
                kotak_products=["savings", "fd"],
                city="Trivandrum",
                tier=2,
                occupation="professional",
                age_group="41-50",
                gender="female",
                has_active_loan=True,
                loan_status="active",
                previous_providers=[],
                loan_tenure_months=5,
            ),
            CustomerProfile(
                id="C012",
                name="Rahul Verma",
                language="en",
                segment="high_value",
                current_provider="none",
                estimated_outstanding=0,
                estimated_rate=0,
                kotak_products=["savings", "fd", "credit_card", "demat"],
                city="Delhi",
                tier=1,
                occupation="msme",
                age_group="31-40",
                gender="male",
                has_active_loan=False,
                loan_status="closed",  # Had loan before, now closed
                previous_providers=["manappuram", "iifl"],
                loan_tenure_months=0,
            ),
        ]

        for customer in mock_customers:
            self.customers[customer.id] = customer

        logger.info(f"Loaded {len(self.customers)} mock customer profiles")

    def get_customer(self, customer_id: str) -> Optional[CustomerProfile]:
        """Get customer by ID."""
        return self.customers.get(customer_id)

    def get_all_customers(self) -> list[dict]:
        """Get all customers as list of dicts."""
        return [c.to_dict() for c in self.customers.values()]

    def get_customers_by_segment(self, segment: str) -> list[CustomerProfile]:
        """Get all customers in a segment."""
        return [c for c in self.customers.values() if c.segment == segment]

    def get_customers_by_language(self, language: str) -> list[CustomerProfile]:
        """Get all customers by preferred language."""
        return [c for c in self.customers.values() if c.language == language]
