"""
Advanced Synthetic Contract Document Generation Script (Hard Version).

This script creates a more realistic and challenging dataset for the
contract classification project. It introduces several real-world complexities:

1.  **Class Imbalance:** The number of documents per category is uneven,
    simulating a real-world scenario where some contract types are
    far more common than others.
2.  **Keyword Overlap (Ambiguity):** The script intentionally injects
    keywords from one category into another (e.g., adding NDA-style
    confidentiality clauses to Vendor agreements). This forces models
    to learn context rather than just memorizing keywords.
3.  **Longer Documents:** A much larger pool of boilerplate text is used
    to generate longer documents, testing the models' ability to handle
    more noise and potentially requiring truncation.

To run:
    python scripts/generate_hard_data.py
"""

import pandas as pd
import random
from pathlib import Path

# --- Configuration ---
# Define an imbalanced distribution for 5,000 total documents.
DOC_DISTRIBUTION = {
    "NDA": 2000,
    "SLA": 1500,
    "Vendor": 800,
    "Employment": 500,
    "Partnership": 200,
}

# Define the output path for the generated dataset.
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "contracts_hard.csv"

# --- Data Definitions ---
KEYWORDS = {
    "NDA": ["Confidential Information", "Disclosing Party", "Receiving Party", "proprietary information", "trade secrets", "non-disclosure"],
    "SLA": ["Service-Level Agreement", "service uptime", "performance metrics", "response time", "service credits", "availability"],
    "Employment": ["Employment Contract", "employee", "employer", "job title", "salary", "annual leave", "scope of employment"],
    "Vendor": ["Vendor Agreement", "supplier", "goods and services", "purchase order", "payment terms", "delivery schedule", "invoice"],
    "Partnership": ["Partnership Agreement", "partners", "capital contribution", "profit and loss distribution", "business purpose", "dissolution"]
}

# Expanded list of boilerplate phrases for longer, more complex documents.
BOILERPLATE = [
    "This Agreement is made and entered into as of the Effective Date by and between the parties.",
    "This Agreement shall be governed by and construed in accordance with the laws of the State of [STATE], without regard to its conflict of law principles.",
    "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.",
    "Any notice or other communication given hereunder shall be in writing and shall be deemed given when delivered personally or sent by certified mail.",
    "This document, including any attachments, constitutes the entire agreement between the parties.",
    "Neither party may assign its rights or obligations under this Agreement without the prior written consent of the other party.",
    "If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions will remain in full force and effect.",
    "The relationship of the parties is that of independent contractors, and nothing in this Agreement shall be construed to create a partnership, joint venture, or agency relationship.",
    "No waiver of any provision of this Agreement shall be effective unless in writing and signed by the waiving party.",
    "All disputes arising out of or in connection with this Agreement shall be finally settled under the Rules of Arbitration of the International Chamber of Commerce.",
    "This Agreement may be executed in one or more counterparts, each of which shall be deemed an original, but all of which together shall constitute one and the same instrument.",
    "The headings in this Agreement are for convenience only and shall not affect its interpretation.",
    "Force Majeure. Neither party shall be liable for any failure to perform its obligations hereunder where such failure results from any cause beyond such party's reasonable control.",
    "Each party agrees to indemnify and hold harmless the other party against any and all claims, losses, damages, liabilities, penalties, punitive damages, expenses, and costs.",
    "This Agreement shall terminate upon thirty (30) days written notice by either party.",
    "The confidentiality obligations set forth herein shall survive the termination of this Agreement for a period of five (5) years.",
    "All intellectual property rights, including copyrights, patents, patent disclosures and inventions (whether patentable or not), that are created or conceived by a party hereunder shall be owned by that party.",
    "Time is of the essence in the performance of the obligations in this Agreement.",
    "This Agreement supersedes all prior oral or written agreements, commitments, or understandings between the parties.",
    "Any amendments or modifications to this Agreement must be in writing and signed by authorized representatives of both parties.",
    "This Agreement is for the sole benefit of the parties hereto and their respective successors and permitted assigns and nothing herein, express or implied, is intended to or shall confer upon any other person or entity any legal or equitable right, benefit or remedy of any nature whatsoever under or by reason of this Agreement.",
    "The language in all parts of this Agreement shall be in all cases construed as a whole, according to its fair meaning, and not strictly for or against any party.",
    "Each party shall be responsible for its own costs and expenses incurred in connection with the negotiation and execution of this Agreement.",
    "This Agreement does not create any third-party beneficiary rights in any individual or entity that is not a party to this Agreement.",
    "The failure of either party to enforce any provision of this Agreement shall not be construed as a waiver or limitation of that party's right to subsequently enforce and compel strict compliance with every provision of this Agreement.",
    "The parties acknowledge that they have had an adequate opportunity to consult with legal counsel of their choice.",
    "This Agreement shall be binding upon and inure to the benefit of the parties hereto and their respective heirs, successors, and assigns.",
    "All remedies, either under this Agreement or by law or otherwise afforded to any party, shall be cumulative and not alternative.",
    "The descriptive headings of the sections and subsections of this Agreement are for convenience only and do not affect this Agreement’s construction or interpretation.",
    "Each party represents and warrants that it has the full power and authority to enter into and perform its obligations under this Agreement."
] * 4 # Repeat the list to get over 100 items for more variety.

def generate_ambiguous_document(category: str) -> str:
    """
    Generates a single, more complex synthetic document.

    This function introduces ambiguity by potentially injecting keywords
    from a different, randomly chosen category.

    Args:
        category (str): The primary contract category for the document.

    Returns:
        str: A string representing the complex synthetic contract text.
    """
    # --- Primary Signal ---
    num_keywords = random.randint(4, 6)
    signal_phrases = random.sample(KEYWORDS[category], num_keywords)

    # --- Inject Ambiguity (30% chance) ---
    if random.random() < 0.3:
        other_categories = list(KEYWORDS.keys())
        other_categories.remove(category)
        ambiguity_category = random.choice(other_categories)
        
        num_ambiguity_keywords = random.randint(1, 2)
        ambiguity_phrases = random.sample(KEYWORDS[ambiguity_category], num_ambiguity_keywords)
        signal_phrases.extend(ambiguity_phrases)

    # --- Noise ---
    num_boilerplate = random.randint(15, 25) # Increased number for longer docs
    noise_phrases = random.sample(BOILERPLATE, num_boilerplate)

    document_parts = signal_phrases + noise_phrases
    random.shuffle(document_parts)
    return ". ".join(document_parts) + "."

def main():
    """Main function to generate the dataset."""
    print("Starting generation of 'hard' synthetic dataset...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_contracts = []
    for category, num_docs in DOC_DISTRIBUTION.items():
        print(f"Generating {num_docs} documents for category: {category}")
        for _ in range(num_docs):
            text = generate_ambiguous_document(category)
            all_contracts.append({"text": text, "category": category})

    df = pd.DataFrame(all_contracts)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    total_docs = sum(DOC_DISTRIBUTION.values())
    print(f"\nSuccessfully generated {total_docs} documents.")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
