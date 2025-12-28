"""Application-wide constants."""

# Kotak Gold Loan rates and product info
KOTAK_GOLD_LOAN = {
    "interest_rate_min": 9.0,
    "interest_rate_max": 10.0,
    "processing_fee_percent": 0.5,
    "ltv_ratio": 75,  # Loan-to-Value ratio
    "min_loan_amount": 25000,
    "max_loan_amount": 5000000,
    "tenure_months_min": 6,
    "tenure_months_max": 36,
}

# Competitor rates for comparison
COMPETITOR_RATES = {
    "muthoot": {"rate_min": 15, "rate_max": 24},
    "manappuram": {"rate_min": 16, "rate_max": 26},
    "iifl": {"rate_min": 14, "rate_max": 24},
    "other_nbfc": {"rate_min": 18, "rate_max": 27},
}

# Customer segment definitions
SEGMENTS = {
    "p1": "high_value",      # High-value MSME with large loans
    "p2": "trust_seeker",    # Safety-conscious, need trust building
    "p3": "shakti",          # Women entrepreneurs
    "p4": "young_professional",  # Digital-savvy young customers
}

# Conversation state thresholds
CONVERSATION_LIMITS = {
    "max_turns": 20,
    "objection_retry_limit": 3,
    "urgency_trigger_after_turns": 5,
    "slow_response_threshold_ms": 3000,
}

# RAG retrieval settings
RAG_CONFIG = {
    "default_top_k": 5,
    "max_context_chars": 500,
    "similarity_threshold": 0.7,
    "hybrid_semantic_weight": 0.7,
}
