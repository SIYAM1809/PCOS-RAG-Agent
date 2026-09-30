"""
Evaluation dataset definitions for RAGAS quantitative assessment of PCOS Care Navigator.
Contains the 7 benchmark guideline queries, categories, and reference ground truth answers.
"""

from typing import List, Dict, Any

eval_cases: List[Dict[str, Any]] = [
    {
        "id": 1,
        "category": "Diagnostic Criteria Definition",
        "question": "What are the 3 core diagnostic criteria established in the Rotterdam 2003 consensus for PCOS?",
        "ground_truth": (
            "According to the Rotterdam 2003 consensus, PCOS diagnosis requires the presence of at least 2 of 3 core features. "
            "The first core feature is oligo- or anovulation (ovulatory dysfunction). "
            "The second core feature is clinical and/or biochemical signs of hyperandrogenism. "
            "The third core feature is polycystic ovaries on ultrasound examination. "
            "Other etiologies such as congenital adrenal hyperplasia, thyroid disease, or hyperprolactinemia must be excluded."
        )
    },
    {
        "id": 2,
        "category": "Ultrasound Thresholds",
        "question": "What are the specific ultrasound threshold criteria (follicle count or ovarian volume) for defining polycystic ovarian morphology?",
        "ground_truth": (
            "Follicle number per ovary (FNPO) ≥ 20 (or 20 or more follicles) in at least one ovary defines polycystic ovarian morphology. "
            "Ovarian volume (OV) ≥ 10 ml (or 10 mL or more) in at least one ovary defines polycystic ovarian morphology. "
            "Follicle number per section (FNPS) ≥ 10 in at least one ovary can be used when FNPO cannot be reliably assessed. "
            "These ultrasound threshold criteria apply to adult women in the absence of a corpus luteum or dominant follicle."
        )
    },
    {
        "id": 3,
        "category": "Biochemical / Hormonal Ratios",
        "question": "Why would an elevated LH to FSH ratio be observed in a patient suspected of PCOS?",
        "ground_truth": (
            "An elevated LH to FSH ratio (often 2:1 or 3:1) can be observed in patients with PCOS due to altered "
            "hypothalamic GnRH pulse frequency and pituitary sensitivity, which increases LH secretion relative to FSH. "
            "Excess LH stimulates ovarian theca cells to produce androgens. While a common supportive biochemical finding, "
            "guidelines state that LH:FSH ratio is not a mandatory standalone diagnostic criterion due to high variability."
        )
    },
    {
        "id": 4,
        "category": "First-Line Clinical Management",
        "question": "What lifestyle and behavioral interventions are recommended as the first-line management approach for PCOS?",
        "ground_truth": (
            "Multicomponent lifestyle interventions combining diet, exercise, and behavioral strategies are recommended as the first-line management for all women with PCOS. "
            "Lifestyle interventions aim to optimize general health, quality of life, and emotional wellbeing. "
            "Lifestyle interventions aim to improve metabolic health including central adiposity and lipid profile. "
            "Behavioral strategies such as SMART goal-setting, self-monitoring, and problem-solving should be included to support lifestyle change. "
            "No single dietary composition has been shown to be superior for outcomes in women with PCOS."
        )
    },
    {
        "id": 5,
        "category": "Adolescent vs Adult Nuance",
        "question": "How should diagnostic assessment for PCOS differ between adolescents and adult women?",
        "ground_truth": (
            "In adolescents, diagnostic criteria are more stringent: both ovulatory dysfunction and clinical or biochemical hyperandrogenism are required for diagnosis. "
            "Ultrasound examination for polycystic ovarian morphology (PCOM) is not recommended in adolescents within 8 years of menarche. "
            "This exclusion is due to the high prevalence of multi-follicular ovaries during normal pubertal maturation."
        )
    },
    {
        "id": 6,
        "category": "Lay Terminology (CRAG Rewrite)",
        "question": "why do I have weird skipped periods and extra dark facial hair growing?",
        "ground_truth": (
            "Irregular or skipped menstrual periods and abnormal dark facial hair growth are classic manifestations of PCOS. "
            "Skipped periods indicate ovulatory dysfunction such as oligo-amenorrhea. "
            "Excess dark facial hair indicates clinical hyperandrogenism or hirsutism. "
            "Clinical guidelines recommend seeking medical evaluation to determine if diagnostic criteria are met and rule out other causes."
        )
    },
    {
        "id": 7,
        "category": "Out of Scope (Hard Stop / No-Hallucination)",
        "question": "What is the recommended dosing titration for semaglutide in pediatric differentiated thyroid carcinoma?",
        "ground_truth": (
            "This information is not available in the loaded PCOS clinical guidelines. Semaglutide dosing titration for "
            "pediatric differentiated thyroid carcinoma is outside the scope of PCOS diagnostic and management protocols."
        )
    },
]
