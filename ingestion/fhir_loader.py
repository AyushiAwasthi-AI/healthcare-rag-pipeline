"""
ingestion/fhir_loader.py
Loads FHIR R4 Patient and Observation resources.
In production, this connects to Azure Health Data Services FHIR API.
For this project, loads from mock FHIR JSON fixtures.

FHIR R4 is the healthcare interoperability standard used by:
- Epic, Cerner (EHR systems)
- Azure Health Data Services
- CMS Interoperability Rule (mandatory for US payers since 2021)
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class FHIRPatientLoader:
    """
    Loads FHIR R4 Patient and Observation resources.
    Extracts clinically relevant context for RAG personalization.
    """

    def load_patient_context(self, patient_id: str) -> Optional[str]:
        """
        Load patient context from FHIR fixture.
        Returns formatted clinical context string for RAG prompt injection.
        """
        fixture_path = BASE_DIR / "data" / "fhir" / f"{patient_id}.json"

        if not fixture_path.exists():
            logger.info(f"No FHIR fixture for patient {patient_id}")
            return None

        with open(fixture_path) as f:
            bundle = json.load(f)

        return self._extract_clinical_context(bundle)

    def _extract_clinical_context(self, bundle: dict) -> str:
        """
        Extract key clinical facts from FHIR Bundle.
        Formats for injection into RAG generation prompt.
        """
        context_lines = ["PATIENT CLINICAL CONTEXT (from EHR):"]

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            r_type   = resource.get("resourceType")

            if r_type == "Patient":
                name   = resource.get("name", [{}])[0]
                given  = " ".join(name.get("given", []))
                family = name.get("family", "")
                dob    = resource.get("birthDate", "unknown")
                gender = resource.get("gender", "unknown")
                context_lines.append(f"- Patient: {given} {family}")
                context_lines.append(f"- DOB: {dob} | Gender: {gender}")

            elif r_type == "Condition":
                code    = resource.get("code", {})
                display = code.get("text") or (
                    code.get("coding", [{}])[0].get("display", "unknown")
                )
                icd     = code.get("coding", [{}])[0].get("code", "")
                context_lines.append(f"- Active condition: {display} (ICD-10: {icd})")

            elif r_type == "Observation":
                code    = resource.get("code", {}).get("text", "")
                value   = resource.get("valueQuantity", {})
                val_str = f"{value.get('value', '')} {value.get('unit', '')}"
                date    = resource.get("effectiveDateTime", "")[:10]
                context_lines.append(f"- {code}: {val_str} (as of {date})")

            elif r_type == "MedicationRequest":
                med = resource.get("medicationCodeableConcept", {}).get("text", "unknown")
                context_lines.append(f"- Current medication: {med}")

        return "\n".join(context_lines)