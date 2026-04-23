"""
Medical keyword dictionaries for CoT reasoning chain scoring.

Sources:
- ICDR grading standard: Wilkinson et al. (2003), Ophthalmology 110(9):1677-1682
- ETDRS reports: Early Treatment Diabetic Retinopathy Study Research Group
- Standard ophthalmology terminology
"""

# ---------------------------------------------------------------------------
# L3 Lesion Keywords
# Each lesion maps to:
#   required: at least one must appear for non-zero score
#   supporting: additional terms that boost score
# ---------------------------------------------------------------------------

LESION_KEYWORDS: dict = {
    "Fibrous_proliferation": {
        "required": ["fibrous", "proliferation", "fibrosis", "fibrovascular"],
        "supporting": ["membrane", "traction", "white tissue", "scar", "epiretinal"],
    },
    "Haemorrhage": {
        "required": ["hemorrhage", "haemorrhage", "bleeding", "blood"],
        "supporting": ["flame", "blot", "dot-blot", "dark red", "intraretinal", "subretinal", "preretinal"],
    },
    "Hard_exudate": {
        "required": ["hard exudate", "exudate", "lipid exudate"],
        "supporting": ["yellow", "bright", "lipid", "waxy", "well-defined", "crystalline"],
    },
    "Intraretinal_Fluid": {
        "required": ["intraretinal fluid", "IRF", "cystoid", "cyst"],
        "supporting": ["hyporeflective", "space", "edema", "swelling", "retinal thickening"],
    },
    "Intraretinal_hemorrhage": {
        "required": ["intraretinal hemorrhage", "intraretinal haemorrhage", "dot hemorrhage", "blot hemorrhage"],
        "supporting": ["dark", "round", "oval", "deep retina", "inner nuclear"],
    },
    "Microaneurysm": {
        "required": ["microaneurysm", "micro-aneurysm"],
        "supporting": ["small red dot", "dot hemorrhage", "punctate", "capillary", "round red", "tiny red"],
    },
    "Neovascularization": {
        "required": ["neovascularization", "new vessel", "NVD", "NVE", "new blood vessel"],
        "supporting": ["proliferative", "fibrovascular", "frond", "disc vessel", "retinal vessel"],
    },
    "Pigment_Epithelial_Detachment": {
        "required": ["pigment epithelial detachment", "PED", "RPE detachment"],
        "supporting": ["dome", "elevation", "subretinal", "RPE", "pigment epithelium"],
    },
    "Preretinal_hemorrhage": {
        "required": ["preretinal hemorrhage", "preretinal haemorrhage", "boat-shaped", "subhyaloid"],
        "supporting": ["flat top", "horizontal level", "vitreous", "anterior", "large"],
    },
    "Soft_exudate": {
        "required": ["soft exudate", "cotton wool", "cotton-wool spot", "CWS"],
        "supporting": ["white", "fluffy", "nerve fiber", "ischemia", "superficial", "feathery"],
    },
    "Subretinal_Fluid": {
        "required": ["subretinal fluid", "SRF", "sub-retinal fluid"],
        "supporting": ["hyporeflective", "beneath retina", "detachment", "neurosensory", "accumulation"],
    },
    "Vitreous_hemorrhage": {
        "required": ["vitreous hemorrhage", "vitreous haemorrhage", "vitreous bleeding"],
        "supporting": ["vitreous", "haze", "obscure", "red reflex", "floater", "dense"],
    },
}

# ---------------------------------------------------------------------------
# L4c DR Grading Keywords (ICDR standard)
# ---------------------------------------------------------------------------

DR_GRADE_KEYWORDS: dict = {
    0: {
        "required": ["no dr", "no diabetic retinopathy", "normal", "no retinopathy", "no lesion"],
        "supporting": ["clear", "healthy", "unremarkable", "no abnormality"],
        "forbidden": ["neovascularization", "proliferative", "severe hemorrhage", "cotton wool"],
    },
    1: {
        "required": ["mild", "mild npdr", "mild non-proliferative", "microaneurysm only"],
        "supporting": ["few microaneurysm", "minimal", "early", "mild change"],
        "forbidden": ["neovascularization", "proliferative", "severe", "moderate"],
    },
    2: {
        "required": ["moderate", "moderate npdr", "moderate non-proliferative"],
        "supporting": ["hemorrhage", "exudate", "microaneurysm", "more than mild"],
        "forbidden": ["neovascularization", "proliferative", "severe npdr"],
    },
    3: {
        "required": ["severe", "severe npdr", "severe non-proliferative", "4-2-1 rule", "very severe"],
        "supporting": ["hemorrhage in all quadrant", "IRMA", "venous beading", "intraretinal microvascular"],
        "forbidden": ["neovascularization", "proliferative dr", "PDR"],
    },
    4: {
        "required": ["proliferative", "PDR", "proliferative diabetic retinopathy", "neovascularization"],
        "supporting": ["new vessel", "vitreous hemorrhage", "fibrous proliferation", "traction", "NVD", "NVE"],
        "forbidden": [],
    },
}

# ---------------------------------------------------------------------------
# L4d AMD Keywords
# ---------------------------------------------------------------------------

AMD_KEYWORDS: dict = {
    "No_AMD": {
        "required": ["no amd", "no age-related macular degeneration", "normal macula", "no drusen", "normal"],
        "supporting": ["healthy macula", "no abnormality", "clear fovea"],
        "forbidden": ["drusen", "geographic atrophy", "CNV", "neovascular"],
    },
    "Early_AMD": {
        "required": ["early amd", "early age-related", "small drusen", "drusen"],
        "supporting": ["pigment", "RPE change", "few drusen", "small hard drusen"],
        "forbidden": ["geographic atrophy", "CNV", "neovascular amd", "late amd"],
    },
    "Intermediate_AMD": {
        "required": ["intermediate amd", "large drusen", "intermediate age-related"],
        "supporting": ["medium drusen", "pigment abnormality", "RPE", "geographic"],
        "forbidden": ["CNV", "neovascular amd", "late amd", "advanced amd"],
    },
    "Late_AMD": {
        "required": ["late amd", "advanced amd", "geographic atrophy", "neovascular amd", "CNV", "choroidal neovascularization"],
        "supporting": ["subretinal fluid", "PED", "exudate", "scar", "disciform", "wet amd", "dry amd advanced"],
        "forbidden": [],
    },
}

# ---------------------------------------------------------------------------
# L4a/b Binary and Multi-condition Diagnosis Keywords
# ---------------------------------------------------------------------------

BINARY_DIAGNOSIS_KEYWORDS: dict = {
    "abnormality_present": {
        "required": ["abnormality", "lesion", "disease", "pathology", "abnormal finding"],
        "supporting": ["hemorrhage", "exudate", "neovascularization", "drusen", "edema"],
    },
    "no_abnormality": {
        "required": ["normal", "no abnormality", "healthy", "no lesion", "unremarkable"],
        "supporting": ["clear", "no pathology", "within normal limits"],
    },
}

# Multi-condition diagnosis: common fundus conditions
MULTI_CONDITION_KEYWORDS: dict = {
    "Diabetic_Retinopathy": {
        "required": ["diabetic retinopathy", "DR", "diabetic change"],
        "supporting": ["microaneurysm", "hemorrhage", "exudate", "neovascularization"],
    },
    "Glaucoma": {
        "required": ["glaucoma", "glaucomatous", "optic nerve damage"],
        "supporting": ["cup-to-disc ratio", "CDR", "optic disc", "nerve fiber layer", "RNFL"],
    },
    "AMD": {
        "required": ["age-related macular degeneration", "AMD", "macular degeneration"],
        "supporting": ["drusen", "geographic atrophy", "CNV", "RPE"],
    },
    "Hypertensive_Retinopathy": {
        "required": ["hypertensive retinopathy", "hypertension", "AV nicking"],
        "supporting": ["arteriovenous", "flame hemorrhage", "papilledema", "silver wire"],
    },
    "Retinal_Detachment": {
        "required": ["retinal detachment", "detached retina"],
        "supporting": ["subretinal fluid", "tear", "break", "rhegmatogenous"],
    },
    "Macular_Edema": {
        "required": ["macular edema", "diabetic macular edema", "DME", "CME"],
        "supporting": ["thickening", "fluid", "fovea", "central", "cystoid"],
    },
}

# ---------------------------------------------------------------------------
# L2 Anatomy Keywords (for laterality and OD/fovea tasks)
# ---------------------------------------------------------------------------

ANATOMY_KEYWORDS: dict = {
    "left_eye": {
        "required": ["left eye", "left fundus", "OS", "left"],
        "supporting": ["optic disc on right", "disc temporal", "nasal on left"],
    },
    "right_eye": {
        "required": ["right eye", "right fundus", "OD", "right"],
        "supporting": ["optic disc on left", "disc nasal", "temporal on right"],
    },
    "optic_disc": {
        "required": ["optic disc", "optic nerve head", "ONH", "disc"],
        "supporting": ["bright", "circular", "cup", "rim", "pale"],
    },
    "fovea": {
        "required": ["fovea", "foveal", "macula", "macular"],
        "supporting": ["dark", "center", "avascular zone", "FAZ", "pit"],
    },
}
