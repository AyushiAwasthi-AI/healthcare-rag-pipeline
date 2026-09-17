"""
evaluation/test_dataset.py
10 hand-crafted question-answer pairs from the diabetes PDFs.
Ground truths are written from verified document content.
This is your RAGAS test set — small but representative.
"""

TEST_DATASET = [
    {
        "question": "What are the classic symptoms of Type 2 diabetes?",
        "ground_truth": (
            "Classic symptoms of Type 2 diabetes include polyuria, thirst, "
            "blurred vision, paresthesia, and fatigue from hyperglycemia. "
            "Chronic skin infections, generalized pruritus, and vaginitis "
            "are also common presenting complaints."
        )
    },
    {
        "question": "What lifestyle modifications are recommended for Type 2 diabetes management?",
        "ground_truth": (
            "Lifestyle modifications include dietary changes to reduce caloric intake, "
            "regular physical activity, weight reduction in overweight patients, "
            "and smoking cessation to reduce cardiovascular risk."
        )
    },
    {
        "question": "What is HbA1c and what does it measure?",
        "ground_truth": (
            "HbA1c is glycated hemoglobin reflecting average blood glucose over "
            "2 to 3 months. It is used to assess long-term glycemic control "
            "and guide treatment decisions in diabetes management."
        )
    },
    {
        "question": "What are the diagnostic criteria for Type 2 diabetes?",
        "ground_truth": (
            "Diagnosis is established when classic symptoms accompany a random plasma "
            "glucose greater than 200 mg/dl, or when fasting plasma glucose exceeds "
            "126 mg/dl on two occasions, or when the 2-hour glucose tolerance test "
            "value exceeds 200 mg/dl."
        )
    },
    {
        "question": "What are the risk factors for developing Type 2 diabetes?",
        "ground_truth": (
            "Risk factors include obesity, strong family history of diabetes, "
            "physical inactivity, age over 45, prior gestational diabetes, "
            "recurrent skin or urinary tract infections, and birth weight "
            "greater than 4 kg."
        )
    },
    {
        "question": "What oral medications are used to treat Type 2 diabetes?",
        "ground_truth": (
            "Oral medications include metformin as first-line therapy and "
            "sulfonylureas as additional agents. Insulin therapy may be required "
            "when oral medications cannot achieve adequate glycemic control, "
            "particularly in patients with significant hyperglycemia."
        )
    },
    {
        "question": "What are the complications of poorly controlled diabetes?",
        "ground_truth": (
            "Poorly controlled diabetes leads to microvascular complications "
            "including retinopathy, nephropathy, and neuropathy. Macrovascular "
            "complications include cardiovascular disorders. Type 2 diabetes is "
            "responsible for initiating a cluster of degenerative diseases."
        )
    },
    {
        "question": "How does obesity relate to Type 2 diabetes?",
        "ground_truth": (
            "Obesity is a major risk factor for Type 2 diabetes. Many obese patients "
            "have an insidious onset of hyperglycemia and may be relatively "
            "asymptomatic initially. Weight reduction is a key component of "
            "diabetes management in overweight patients."
        )
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "ground_truth": (
            "Type 1 diabetes has an acute onset with classic symptoms of polydipsia, "
            "polyuria, polyphagia, and significant weight loss lasting 2-3 weeks. "
            "Type 2 diabetes has a more insidious onset, is generally perceived as "
            "less serious initially, but causes profound pathological features "
            "including cardiovascular disorders."
        )
    },
    {
        "question": "What monitoring is required for diabetes management?",
        "ground_truth": (
            "Diabetes management requires regular monitoring of blood glucose levels "
            "and HbA1c to assess glycemic control. Screening is important for "
            "patients with family history of diabetes, significant obesity, or "
            "recurrent infections to detect diabetes early."
        )
    },
]