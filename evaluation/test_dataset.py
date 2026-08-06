"""
evaluation/test_dataset.py
5 hand-crafted question-answer pairs from the diabetes PDFs.
Ground truths are written from verified document content.
This is your RAGAS test set — small but representative.
"""

TEST_DATASET = [
    {
        "question": "What are the classic symptoms of Type 2 diabetes?",
        "ground_truth": (
            "Classic symptoms of Type 2 diabetes include polyuria, thirst, "
            "blurred vision, paresthesia, and fatigue, which are manifestations "
            "of hyperglycemia and osmotic diuresis. Chronic skin infections, "
            "generalized pruritus, and symptoms of vaginitis are also common."
        )
    },
    {
        "question": "What lifestyle modifications are recommended for Type 2 diabetes?",
        "ground_truth": (
            "Lifestyle modifications for Type 2 diabetes management include "
            "dietary changes to reduce caloric intake, regular physical activity, "
            "weight reduction in overweight patients, and smoking cessation."
        )
    },
    {
        "question": "What is HbA1c and what does it measure?",
        "ground_truth": (
            "HbA1c is glycated hemoglobin that reflects average blood glucose "
            "levels over the preceding 2 to 3 months. It is used to assess "
            "long-term glycemic control in diabetes management."
        )
    },
    {
        "question": "What are the risk factors for developing Type 2 diabetes?",
        "ground_truth": (
            "Risk factors for Type 2 diabetes include obesity, strong family "
            "history of diabetes, physical inactivity, age over 45, prior "
            "gestational diabetes, and impaired fasting glucose."
        )
    },
    {
        "question": "What oral medications are used to treat Type 2 diabetes?",
        "ground_truth": (
            "Oral medications for Type 2 diabetes include metformin as first-line "
            "therapy, sulfonylureas, and other agents. Insulin therapy may be "
            "required when oral medications cannot achieve adequate glycemic control."
        )
    },
]