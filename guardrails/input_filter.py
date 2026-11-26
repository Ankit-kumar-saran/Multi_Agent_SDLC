# guardrails/input_filter.py
"""Input guardrails for security, sensitive-data masking, and financial domain gating.

Enforces:
1. Zero-Trust Security Guardrails: Blocks prompt injection, jailbreak attempts, and destructive commands.
2. Sensitive Data Masking: Redacts PII, account/card numbers, emails, phone numbers, and secrets
   before sending prompts to external LLM APIs.
3. Financial Domain Scope Guardrails: Ensures requests relate to financial software systems
   (banking, lending, payments, fraud/AML, insurance, investments/trading, KYC, reporting, analytics).
"""

import re
from typing import List, Tuple

DISALLOWED_INPUT_PHRASES = {
    "drop database",
    "drop table",
    "shutdown server",
    "run sudo",
    "delete all",
    "rm -rf",
    "jailbreak",
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard system prompt",
    "reveal system prompt",
    "malware",
    "ransomware",
    "sql injection payload",
    "exploit vulnerability",
}

FINANCIAL_SUBDOMAINS = {
    "Payments, Transfers & Billing": [
        "payment", "payments", "upi", "autopay", "mandate", "wallet", "pos",
        "debit card", "credit card", "payment card", "rupay", "visa", "mastercard",
        "qr", "merchant", "bbps", "tokenization", "tokenisation", "switch",
        "settlement", "clearing", "gateway", "remittance", "transaction",
        "checkout", "interbank", "payout", "billing", "invoice", "wire transfer",
        "ach", "sepa", "iso 20022", "iso 8583", "cbdc", "digital currency"
    ],
    "Digital & Core Banking": [
        "bank", "banking", "fintech", "financial", "finance", "cbs", "core banking",
        "account", "accounts", "ledger", "deposit", "withdrawal", "neft", "rtgs",
        "imps", "nach", "enach", "atm", "cash", "vault", "cassette", "passbook",
        "statement", "overdraft", "treasury", "swift", "reconciliation",
        "double-entry", "savings", "checking", "branch", "fx", "foreign exchange"
    ],
    "Lending, Credit & Underwriting": [
        "loan", "loans", "lending", "borrower", "lender", "nbfc", "cibil", "crif",
        "experian", "equifax", "underwriting", "disbursal", "disbursement",
        "repayment", "emi", "kfs", "apr", "interest rate", "dbt", "microfinance",
        "shg", "collateral", "mortgage", "debt", "credit risk", "credit scoring",
        "credit line", "credit", "bnpl"
    ],
    "Capital Markets, Trading & Wealth": [
        "trading", "trade", "algo trading", "algorithmic trading", "broker",
        "brokerage", "stock", "stocks", "equity", "equities", "share", "shares",
        "demat", "depository", "nsdl", "cdsl", "nse", "bse", "sebi", "derivative",
        "derivatives", "futures", "options", "fix protocol", "colocation",
        "order routing", "rms", "circuit breaker", "portfolio", "mutual fund",
        "sip", "amc", "nav", "wealth", "investment", "asset management", "robo advisor"
    ],
    "Insurance & Actuarial Systems": [
        "insurance", "insurtech", "policy", "premium", "claim", "claims",
        "underwrite", "tpa", "irdai", "annuity", "coverage", "actuarial",
        "cashless", "indemnity", "reimbursement", "solvency", "actuary"
    ],
    "Fraud, AML, KYC & Regulatory Reporting": [
        "kyc", "e-kyc", "ekyc", "v-cip", "aadhaar", "uidai", "pan", "ckyc",
        "ckcy", "aml", "pmla", "str", "ctr", "fiu", "cscrf", "fraud",
        "anti-money laundering", "sanction", "sanctions", "audit trail",
        "regulatory reporting", "compliance", "rbi", "npci", "dpdp", "pci dss",
        "pci-dss", "sox", "basel", "gdpr", "fatf", "risk management", "tax",
        "accounting", "payroll", "regtech", "open banking", "account aggregator"
    ],
}

OUT_OF_DOMAIN_CATEGORIES = [
    {
        "category": "Healthcare & Clinical Management",
        "patterns": [
            r"\bhealth\s*sector\b", r"\bhealth\s*care\b", r"\bhealthcare\b",
            r"\bhospital\b", r"\bclinic\b", r"\bclinical\b", r"\bpatient\b",
            r"\bdoctor\b", r"\bnurse\b", r"\bward\b", r"\bradiology\b",
            r"\bpathology\b", r"\behr\b", r"\bemr\b", r"\bprescription\b",
            r"\bsurgery\b", r"\bpharmacy\b", r"\bmedical\b", r"\btelemedicine\b",
        ],
        "allowed_exceptions": [
            "insurance", "claim", "claims", "tpa", "billing", "payment",
            "payments", "reimbursement", "premium", "policy", "gateway", "loan"
        ],
        "clarification": "If you are building a health insurance claims, medical billing, or hospital payment system, please include those financial requirements in your prompt.",
    },
    {
        "category": "Gaming & Entertainment",
        "patterns": [
            r"\bvideo\s*game\b", r"\bgame\s*engine\b", r"\brpg\b",
            r"\bmultiplayer\s*game\b", r"\besports\b", r"\bmovie\s*streaming\b",
            r"\bmusic\s*streaming\b", r"\bpodcast\s*player\b",
        ],
        "allowed_exceptions": ["in-game purchase", "payment gateway", "wallet", "checkout", "microtransaction", "payout"],
        "clarification": "General gaming and media streaming apps are outside the financial domain unless focused on payments, wallets, or monetization.",
    },
    {
        "category": "Food Delivery & Culinary",
        "patterns": [
            r"\brecipe\b", r"\bcooking\b", r"\bmeal\s*planner\b", r"\brestaurant\b",
            r"\bchef\b", r"\bingredient\b", r"\bdiet\s*tracker\b", r"\bpizza\b",
            r"\bfood\s*deliver(y|ing)?\b", r"\bfood\s*order(ing)?\b", r"\bkitchen\b",
        ],
        "allowed_exceptions": ["payment", "billing", "pos", "checkout", "settlement", "refund", "wallet", "upi", "accounting"],
        "clarification": "General restaurant/food ordering features are outside the financial domain unless designing the payment, POS, or settlement subsystem.",
    },
    {
        "category": "Social Networking & Dating",
        "patterns": [
            r"\bdating\s*app\b", r"\bmatchmaking\b", r"\btinder\b",
            r"\bphoto\s*sharing\b", r"\bsocial\s*media\s*feed\b",
        ],
        "allowed_exceptions": ["creator payout", "tip payment", "wallet", "payment", "fraud"],
        "clarification": "Social networking and dating features are outside the financial domain.",
    },
]


def is_safe(text: str) -> bool:
    """Return True if text does not contain disallowed security/injection phrases."""
    lowered = text.lower()
    return not any(phrase in lowered for phrase in DISALLOWED_INPUT_PHRASES)


def mask_sensitive_data(text: str) -> Tuple[str, List[str]]:
    """Mask sensitive PII/financial secrets before sending text to an external LLM API.

    Returns:
        (masked_text, redaction_types)
    """
    masked = text
    redacted: List[str] = []

    patterns = [
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[MASKED_EMAIL]"),
        ("SECRET", r"(?i)\b(password|passwd|pwd|api[_-]?key|secret|token)\s*[:=]\s*\S+", r"\1=[MASKED_SECRET]"),
        ("CARD_OR_ACCOUNT", r"\b(?:\d[ -]*?){13,19}\b", "[MASKED_CARD_OR_ACCOUNT]"),
        ("NATIONAL_ID", r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b", "[MASKED_ID]"),
        ("TAX_ID", r"\b[A-Z]{5}\d{4}[A-Z]\b", "[MASKED_TAX_ID]"),
        ("PHONE", r"(?:\+?\d{1,3}[\s-]?)?\b[6-9]\d{9}\b", "[MASKED_PHONE]"),
    ]

    for label, pattern, replacement in patterns:
        new_masked, count = re.subn(pattern, replacement, masked)
        if count > 0:
            masked = new_masked
            redacted.append(f"{label} ({count})")

    return masked, redacted


def classify_financial_domain(text: str) -> List[str]:
    """Return all matching financial sub-domain tags for the given input."""
    lowered = text.lower()
    matched_tags: List[str] = []
    for domain, kws in FINANCIAL_SUBDOMAINS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", lowered) for kw in kws):
            matched_tags.append(domain)
    return matched_tags


def check_domain_scope(text: str) -> Tuple[bool, str, str]:
    """Verify the prompt pertains to financial software engineering."""
    lowered = text.lower()
    matched_tags = classify_financial_domain(text)

    for cat in OUT_OF_DOMAIN_CATEGORIES:
        has_cat_match = any(re.search(pat, lowered) for pat in cat["patterns"])
        if has_cat_match:
            has_exception = any(exc in lowered for exc in cat["allowed_exceptions"])
            if has_exception or matched_tags:
                primary = matched_tags[0] if matched_tags else "Financial Subsystem Integration"
                return True, f"{primary} ({cat['category']})", ""
            return False, cat["category"], cat["clarification"]

    if matched_tags:
        return True, matched_tags[0], ""

    clarification = (
        "No financial domain context was detected in this request. "
        "This assistant is scoped to financial software systems (digital banking, payments, "
        "lending, fraud/AML, insurance, investments/trading, KYC, regulatory reporting, or financial analytics). "
        "Please describe the financial functionality or compliance requirements of your system."
    )
    return False, "Non-Financial / General Domain", clarification
