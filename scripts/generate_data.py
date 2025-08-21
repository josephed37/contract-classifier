"""
Synthetic Contract Document Generation Script.

This script is responsible for creating a synthetic dataset for the
contract classification project. It generates a specified number of documents
for several predefined contract categories (e.g., NDA, SLA).

The generation process works by combining:
1.  Category-specific "signal" keywords to ensure a document is identifiable.
2.  A pool of generic "boilerplate" legal phrases that act as noise.

The final output is a shuffled CSV file containing the document text and its
corresponding category, ready for use in the data preprocessing stage.

To run:
    python scripts/generate_data.py
"""

import pandas as pd
import random
from pathlib import Path

# --- Configuration ---
# Define the number of documents to generate per category.
NUM_DOCS_PER_CATEGORY = 200

# Define the output path for the generated dataset.
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "contracts.csv"

# --- Data Definitions ---
KEYWORDS = {
    "NDA": [
        "Confidential Information", "Disclosing Party", "Receiving Party",
        "proprietary information", "trade secrets", "non-disclosure",
        "duty of confidentiality"
    ],
    "SLA": [
        "Service-Level Agreement", "service uptime", "performance metrics",
        "response time", "service credits", "availability", "resolution time",
        "service commitment"
    ],
    "Employment": [
        "Employment Contract", "employee", "employer", "job title", "salary",
        "annual leave", "start date", "termination clause", "scope of employment",
        "probation period"
    ],
    "Vendor": [
        "Vendor Agreement", "supplier", "goods and services", "purchase order",
        "payment terms", "delivery schedule", "invoice", "quality standards",
        "statement of work"
    ],
    "Partnership": [
        "Partnership Agreement", "partners", "partnership", "capital contribution",
        "profit and loss distribution", "business purpose", "management and voting",
        "dissolution of partnership"
    ]
}

# A list of boilerplate phrases for more realistic "noise".
BOILERPLATE = [
    "This Agreement is made and entered into as of the Effective Date.",
    "This Agreement shall be governed by and construed in accordance with the laws of the State of [STATE].",
    "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.",
    "Any notice or other communication given hereunder shall be in writing and shall be deemed given.",
    "This document constitutes the entire agreement between the parties.",
    "Neither party may assign its rights or obligations under this Agreement without the prior written consent of the other party.",
    "If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions will remain in full force and effect.",
    "The relationship of the parties is that of independent contractors.",
    "No waiver of any provision of this Agreement shall be effective unless in writing and signed by the waiving party.",
    "All disputes arising out of or in connection with this Agreement shall be finally settled under the Rules of Arbitration.",
    "This Agreement may be executed in counterparts, each of which shall be deemed an original.",
    "The headings in this Agreement are for convenience only and shall not affect its interpretation.",
    "Force Majeure. Neither party shall be liable for any failure to perform due to causes beyond its reasonable control.",
    "The parties agree to indemnify and hold harmless each other against any and all claims.",
    "This Agreement shall terminate upon the completion of the services described herein.",
    "Confidentiality obligations shall survive the termination of this Agreement.",
    "All intellectual property rights developed hereunder shall be owned by [PARTY NAME].",
    "Time is of the essence in the performance of the obligations in this Agreement.",
    "This Agreement supersedes all prior oral or written agreements between the parties.",
    "Any amendments to this Agreement must be in writing and signed by both parties."
]

def generate_document(category: str) -> str:
    """
    Generates a single synthetic contract document for a given category.

    Args:
        category (str): The contract category for which to generate the document.

    Returns:
        str: A string representing the synthetic contract text.
    """
    num_keywords = random.randint(2, 4)
    signal_phrases = random.sample(KEYWORDS[category], num_keywords)

    num_boilerplate = random.randint(3, 5)
    noise_phrases = random.sample(BOILERPLATE, num_boilerplate)

    document_parts = signal_phrases + noise_phrases
    random.shuffle(document_parts)

    return ". ".join(document_parts) + "."

def main():
    """
    Main function to generate the entire dataset and save it to a CSV file.
    """
    print("Starting synthetic data generation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_contracts = []
    for category in KEYWORDS.keys():
        print(f"Generating {NUM_DOCS_PER_CATEGORY} documents for category: {category}")
        for _ in range(NUM_DOCS_PER_CATEGORY):
            text = generate_document(category)
            all_contracts.append({"text": text, "category": category})

    df = pd.DataFrame(all_contracts)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully generated {len(df)} documents.")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
