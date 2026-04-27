"""
data/argilla_setup.py

Creates the Argilla workspace and configures the preference ranking
dataset for domain-expert annotation.

Usage:
    python data/argilla_setup.py --workspace alignr --domain insurance
"""

import argparse
import sys

import argilla as rg


DOMAIN_GUIDELINES = {
    "insurance": """
You are reviewing responses from an AI assistant used in member services for a health insurer.

Prefer the response that:
- Uses precise, non-committal coverage language (avoid "covered", "not covered" — use "typically applies", "subject to review")
- Correctly identifies when to escalate to a human claims adjuster
- Maintains an empathetic but professionally neutral tone on claim denials
- Avoids legally loaded language that could create liability

Reject the response that:
- Makes definitive coverage statements the model cannot guarantee
- Fails to escalate complex or high-stakes queries
- Uses dismissive or cold language on emotionally difficult queries
- Contains unqualified medical or legal advice
""",
    "banking": """
You are reviewing responses from an AI assistant used in lending decision support for a bank.

Prefer the response that:
- Uses fair lending language compliant with ECOA and DORA obligations
- Includes appropriate disclaimers without making credit outcome promises
- Correctly routes lending queries to human advisors when required
- Avoids language that could constitute unlawful credit discrimination

Reject the response that:
- Makes or implies credit approval/denial decisions
- Omits required regulatory disclaimers
- Uses language that could be construed as discriminatory under ECOA
""",
    "healthcare": """
You are reviewing responses from an AI assistant used for clinical admin at an NHS trust.

Prefer the response that:
- Contains zero clinical advice (symptoms, diagnoses, treatment recommendations)
- Correctly escalates all clinical queries to qualified staff
- Is DSPT-compliant in how it handles patient data references
- Uses clear, plain language appropriate for patients and admin staff

Reject the response that:
- Contains any clinical advice, even hedged advice
- Fails to escalate clinical queries
- References patient data in ways that could breach DSPT
""",
    "legal": """
You are reviewing responses from an AI assistant used for contract review at an EU law firm.

Prefer the response that:
- Clearly declines to give legal advice while explaining why
- Uses jurisdiction-aware disclaimers (EU law vs. member state law)
- Correctly identifies privilege boundaries
- Assists with document organisation without interpreting legal obligations

Reject the response that:
- Gives legal advice or interprets legal obligations
- Omits jurisdiction disclaimers
- Crosses privilege boundaries
""",
}


def main():
    parser = argparse.ArgumentParser(description="Set up Argilla annotation workspace")
    parser.add_argument("--workspace", required=True, help="Argilla workspace name")
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_GUIDELINES.keys()),
                        help="Annotation domain")
    parser.add_argument("--argilla-url", default="http://localhost:6900")
    parser.add_argument("--api-key", default="argilla.apikey")
    args = parser.parse_args()

    print(f"Connecting to Argilla at {args.argilla_url}...")
    try:
        rg.init(api_url=args.argilla_url, api_key=args.api_key)
    except Exception as e:
        print(f"ERROR: Cannot connect to Argilla. Is the Docker stack running?\n  {e}")
        print("\nStart it with: docker compose -f docker/argilla/docker-compose.yml up -d")
        sys.exit(1)

    # Create workspace
    try:
        workspace = rg.Workspace.create(args.workspace)
        print(f"Created workspace: {args.workspace}")
    except Exception:
        workspace = rg.Workspace.from_name(args.workspace)
        print(f"Workspace already exists: {args.workspace}")

    # Define preference ranking dataset
    dataset_name = f"alignr-{args.domain}-preferences"
    guidelines = DOMAIN_GUIDELINES[args.domain]

    fields = [
        rg.TextField(name="prompt", title="Prompt", required=True),
        rg.TextField(name="response_a", title="Response A", required=True),
        rg.TextField(name="response_b", title="Response B", required=True),
    ]

    questions = [
        rg.RatingQuestion(
            name="preference",
            title="Which response is better for this domain?",
            description="1 = strongly prefer A, 3 = no preference, 5 = strongly prefer B",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.TextQuestion(
            name="rationale",
            title="Briefly explain your preference (optional but recommended for ART-001)",
            required=False,
        ),
        rg.RatingQuestion(
            name="confidence",
            title="How confident are you?",
            description="1 = unsure, 5 = very confident",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
    ]

    metadata = [
        rg.TermsMetadataProperty(name="domain", title="Domain"),
        rg.TermsMetadataProperty(name="query_category", title="Query category"),
    ]

    try:
        ds = rg.FeedbackDataset(
            fields=fields,
            questions=questions,
            metadata_properties=metadata,
            guidelines=guidelines,
        )
        ds.push_to_argilla(name=dataset_name, workspace=args.workspace)
        print(f"\nDataset '{dataset_name}' created in workspace '{args.workspace}'")
        print(f"Annotators can log in at {args.argilla_url}")
        print(f"Username: argilla  |  Password: 12345678")
        print(f"\nNext step: run data/export_preferences.py after annotation is complete.")
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
