"""
Sentence-level cleaning of MIMIC-IV-Note discharge-summary HPI, for ED-MERGE's
external validation cohort.

Adapted from a sibling project's chest-pain-only HPI cleaner (ED-Agent/data_processing/
hpi_clean.py). The discharge summary is written after the whole stay (ED + any
inpatient course), so its HPI mixes what the patient reported on arrival with
the ED work-up, outside-hospital findings and the hospital course -- using the
raw HPI as "notes available during the ED stay" would leak the outcome. Each
HPI sentence is labelled by an LLM:

  K  patient-reportable history (symptoms, timeline, past history, home meds)
  P  pre-arrival work-up of this episode (outside hospital / clinic / EMS findings)
  D  anything at or after arrival at this ED (tests, results, treatments, impressions)

Only K sentences are kept, verbatim. The cleaned text therefore contains no
model-written words.

Structural regexes (HPI header, section end, sentence splitting, and the
generic encounter/finding/negation/past-episode cues in rule_veto) are
unchanged from the source project -- verified against 500 real MIMIC-IV-Note
discharge.csv rows: 97.8% HPI-header match rate, 0 fallbacks to
the 4000-char cap, median HPI length 1292 chars. The two content tables below
(_STUDY_DX, _TEST_RESULT) and the new _ICU_DEATH block ARE changed -- the
source project only covers the chest-pain differential (ACS/PE/AHF/pneumonia/
COPD/aortic dissection/pneumothorax/pericarditis); ED-MERGE's external
validation covers its own 13 diagnosis outcomes (same ICD-9/10 lists as the
outcome labels) plus the "Critical" outcome (ICU admission or death), which
has no diagnosis name at all and needed a dedicated keyword block.

One fix made here but not present upstream: _NEGATED only caught negation
stated BEFORE a test/ICU-death match ("no troponin elevation"); added
_NEGATED_AFTER for negation stated in the same clause AFTER the match ("CT
head showed no acute hemorrhage"), which the source project's sentence set
apparently never exercised (caught by an 11-case smoke test, 10/11 before
the fix).

Known gap: ED-MERGE's "Bacterial pneumonia" and "Viral pneumonia" outcomes
share one combined all-cause-pneumonia code list (MIMIC codes have no
bacterial/viral split) -- _STUDY_DX cannot distinguish them, both
outcomes rely on the same pneumonia terms.
"""

import json
import re
from typing import List, Optional

_HPI_START = re.compile(r"(?:History of Present Illness|^\s*HPI)\s*:", re.I | re.M)
_SECTION_END = re.compile(
    r"^\s*(past medical history|pmh|past surgical history|psh|social history|family history|"
    r"review of systems|ros|medications on admission|allergies|physical exam\w*|"
    r"pertinent results|brief hospital course|major surgical)\s*:", re.I | re.M)
_PARA_SPLIT = re.compile(r"\n\s*\n|\n\s*\.\s*\n|\n(?=\s*[-*•]\s+)|\n(?=\s*\d+[.)]\s)")
# No sentence break after common abbreviations ("Mr. ___", "Dr. ___", "approx. 3 days").
_ABBREV = re.compile(r"\b(mr|mrs|ms|dr|st|vs|approx|e\.g|i\.e|pt|no|fig)\.$", re.I)


def raw_hpi(note: str) -> str:
    m = _HPI_START.search(note or "")
    if not m:
        return ""
    body = note[m.end():]
    e = _SECTION_END.search(body)
    return body[:e.start()] if e else body[:4000]


def split_sentences(text: str) -> List[dict]:
    """[{"para": i, "text": s}] -- paragraphs on blank lines/bullets, sentences on . ! ?"""
    out = []
    for pi, para in enumerate(p for p in _PARA_SPLIT.split(text or "") if p and p.strip()):
        para = re.sub(r"\s+", " ", para).strip()
        if not re.sub(r"[\W_]", "", para):
            continue
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z_(\"'])", para)
        buf = ""
        for piece in pieces:
            buf = f"{buf} {piece}".strip() if buf else piece
            if _ABBREV.search(buf):
                continue
            out.append({"para": pi, "text": buf})
            buf = ""
        if buf:
            out.append({"para": pi, "text": buf})
    return out


SYSTEM_PROMPT = """You label the sentences of the History of Present Illness (HPI) from a hospital discharge summary.
Goal: keep only what the patient could tell a triage nurse on arrival at THIS emergency department (ED), before any evaluation there.

Labels:
K = patient-reportable history: symptoms and their onset, duration, timing, character, location, radiation, severity, triggers and relieving factors; associated symptoms and symptoms the patient denies; what the patient did before coming (e.g. took a medication at home, called 911); past medical history, prior procedures and diagnoses that existed BEFORE this illness; home medications and adherence; recent events such as surgery, travel, immobility, sick contacts.
P = work-up of the current illness done BEFORE arrival at this ED by an outside hospital, clinic, primary care provider or EMS: their vital signs, examination, tests, test results, diagnoses, treatments, and the reason for referral or transfer.
D = anything at or after arrival at this ED: triage or ED vital signs, examination, labs, imaging and their results, consultations, treatments, diagnoses or impressions of the current illness, ICU admission, death, the admission decision, the hospital or floor course, and plans.

Rules:
- A sentence that only states the presenting complaint (e.g. "presented to the ED with chest pain") is K.
- A sentence that names the diagnosis of the CURRENT illness (not a pre-existing condition) is P or D.
- A sentence that mixes K content with P or D content gets the P or D label.
- A sentence reporting that a test was done or what it showed ("a chest x-ray revealing ...", "was found to have ...", "was sent in for ...") is P or D, even inside a sentence about symptoms.
- A sentence reporting ICU transfer, intubation, or death is D.
- Pre-existing conditions, prior episodes and denials ("history of DVT", "no prior pneumothorax") are K.
- When unsure, choose D.

Return ONLY a JSON object that maps every sentence number to its label, e.g. {"1": "K", "2": "D", "3": "P"}."""


def user_prompt(sentences: List[dict]) -> str:
    lines = [f"[{i + 1}] {s['text']}" for i, s in enumerate(sentences)]
    return (f"HPI sentences ({len(sentences)}):\n" + "\n".join(lines)
            + f"\n\nReturn labels for sentences 1 to {len(sentences)}.")


def parse_labels(raw: str, n: int) -> Optional[List[str]]:
    """Labels from the model output, or None if they are missing or the count is wrong."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {str(i) for i in range(1, n + 1)}:
        return None
    labels = [str(obj[str(i)]).strip().upper()[:1] for i in range(1, n + 1)]
    return labels if all(x in ("K", "P", "D") for x in labels) else None


# Deterministic second layer over the model's K sentences: a sentence that names
# one of ED-MERGE's 13 diagnosis outcomes next to a test/result phrase is
# dropped, unless the name is marked as pre-existing history. Terms below
# are the clinical-synonym expansion of the outcome ICD-9/10 code lists, not the codes themselves (codes don't appear in prose).
_STUDY_DX = re.compile(
    r"\bsepsis\b|septic shock|bacteremia|"
    r"pneumonia|\bpna\b|consolidation|infiltrate|"
    r"\bards\b|acute respiratory distress syndrome|"
    r"\bcopd\b exacerbation|chronic obstructive pulmonary disease exacerbation|"
    r"asthma exacerbation|status asthmaticus|"
    r"acute coronary syndrome|\bacs\b|stemi|nstemi|myocardial infarct|\bmi\b|unstable angina|"
    r"acute (decompensated )?heart failure|\bchf\b exacerbation|decompensated heart failure|"
    r"\bstroke\b|\bcva\b|cerebrovascular accident|ischemic stroke|hemorrhagic stroke|intrac(erebral|ranial) hemorrhage|"
    r"acute kidney injury|\baki\b|acute renal failure|"
    r"pulmonary embol\w*|\bpes?\b", re.I)
_TEST_RESULT = re.compile(
    r"\b(reveal(?!\s+(monitor|linq|loop|device))\w*|found to (have|be)|was found|were found|show(s|ed|ing)|demonstrat\w+|"
    r"consistent with|concerning for|suspicious for|sent (in )?for|"
    r"(evaluation|work-?up|w/u|rule out|r/o) (of|for))\b", re.I)
_HISTORY_NEAR = re.compile(
    r"(history of|h/o|\bhx\b|s/p|prior|previous|known|chronic|recurrent|\bpmh\b|past medical history|"
    r"no|denies|without)\W+(\w+\W+){0,4}$", re.I)


_ENCOUNTER = re.compile(
    r"\b(presents?|presented|presenting|p/w|admitted|transferred|transfer|referred|sent)\b", re.I)
_FINDING = re.compile(
    r"\b(reveal(?!\s+(monitor|linq|loop|device))\w*|found to (have|be)|was found|were found|"
    r"show(s|ed|ing)|demonstrat\w+|consistent with|concerning for|suspicious for)\b", re.I)
_MED_LIST = re.compile(r"\b(was|were)?\s*given\s*:|^\W*given\s*:", re.I)
# A test result of this episode, keyed to ED-MERGE's 13 diagnosis outcomes
# (a heuristic best-effort layer, not expected to be exhaustive).
_TEST_RESULT_WORD = re.compile(
    r"\b(lactate|blood cultures?|procalcitonin|\bsirs\b|\bqsofa\b|\bsofa\b score|"                   # sepsis
    r"chest x-?ray|\bcxr\b|ct chest|sputum culture|respiratory viral panel|flu pcr|covid pcr|rsv pcr|"  # pneumonia
    r"p/f ratio|pao2/fio2|bilateral infiltrates|\babg\b|\bpeep\b|berlin criteria|"                      # ARDS
    r"peak flow|spirometry|\bfev1\b|nebulizer|bronchodilator response|"                                 # COPD/asthma
    r"troponin|\btrop\w*|st[- ](segment )?(elevation|depression)s?|(ekg|ecg) changes|catheterization|"  # ACS/MI
    r"cardiac cath\w*|stress test|wall motion|"
    r"\bbnp\b|nt-proBNP|ejection fraction|pulmonary edema|diuresis|"                                    # AHF
    r"\bnihss\b|ct head|mri brain|\btpa\b|thrombolysis|thrombectomy|cta head|cta neck|"                 # stroke
    r"creatinine|\begfr\b|\bbun\b|urine output|dialysis|renal replacement|"                             # AKI
    r"cta chest|d-?dimer|v/q scan|wells score|heparin|anticoagulation)\b", re.I)                        # PE
_NEGATED = re.compile(r"\b(no|not|never|denies|without)\b[^.;]{0,30}$", re.I)
# Negation stated AFTER the match, in the same clause ("CT head showed no acute
# hemorrhage") -- _NEGATED alone only catches negation stated BEFORE it.
_NEGATED_AFTER = re.compile(r"^[^.,;]{0,20}\b(no|not|never|denies|without)\b", re.I)
# The ED stay itself: what happened in the ED or since presenting there.
_ED_COURSE = re.compile(
    r"\bin (the )?(ed|er|emergency (department|room))\b|\bsince (his|her|their) presentation\b|"
    r"\bfrom when (he|she|they) presented\b", re.I)
# "in the ED" as the destination or time point of the history, not the ED stay.
_ED_ARRIVAL = re.compile(
    r"\b(arrival|arriving|arrive|prior to|before|care|evaluated|evaluation|seen|present\w*|came|come|went|go|"
    r"brought|transported|sent|referred|reported|called)\b[^.;]{0,15}$", re.I)
# Marks a test as part of an earlier episode ("prior", "in ___", "ago", "last admission").
_PAST_ADMISSION = re.compile(r"\b(had been|was|were|recently) (admitted|hospitalized)\b|\bwhen (he|she|they) (was|were)\b", re.I)
_PAST_EPISODE = re.compile(
    r"\b(prior|previous|previously|history of|h/o|hx|s/p|ago|last (year|month|admission)|"
    r"recent admission|recently admitted|in the past|in ___|at that time)\b", re.I)


def _clause(sentence: str, start: int, end: int) -> str:
    """The text around sentence[start:end], not crossing a , . ; boundary --
    so a past-episode/negation cue in an adjacent, unrelated clause ("X was
    positive, with no prior Y") doesn't get attributed to this match."""
    lo = max((sentence.rfind(c, 0, start) for c in ".,;"), default=-1) + 1
    ends = [i for i in (sentence.find(c, end) for c in ".,;") if i != -1]
    hi = min(ends) if ends else len(sentence)
    return sentence[lo:hi]
# ED-MERGE's "Critical" outcome (ICU admission or death)
# has no diagnosis name, so it needs its own leakage cue instead of _STUDY_DX.
# DNR/DNI is deliberately excluded -- it can be a pre-existing code status on
# a chronically ill patient and isn't itself evidence of ICU admission or
# death during THIS encounter.
_ICU_DEATH = re.compile(
    r"\b(icu|micu|sicu|ccu|cvicu|nicu|intubat\w*|mechanical(ly)? ventilat\w*|vasopressors?|pressors?|"
    r"code blue|rapid response|expired|deceased|passed away|comfort care|hospice)\b", re.I)


def rule_veto(sentence: str) -> bool:
    """True if a kept sentence still reports a result, treatment, ICU stay or
    death of this encounter: a test/result phrase after the encounter verb
    ("presents with ..., found to have ..."), a study diagnosis named as the
    presentation ("p/w ... sepsis"), a study diagnosis named shortly after a
    result phrase with no history cue in between, a test result of this
    episode (troponin, lactate, creatinine, ...) not marked as past, an ICU/
    death cue not marked as a pre-existing code status, what happened in the
    ED, or a list of medications given."""
    if _MED_LIST.search(sentence):
        return True
    for m in _ED_COURSE.finditer(sentence):
        if not _ED_ARRIVAL.search(sentence[:m.start()]):
            return True
    for m in _TEST_RESULT_WORD.finditer(sentence):
        if not _PAST_EPISODE.search(_clause(sentence, m.start(), m.end())) \
                and not _NEGATED.search(sentence[:m.start()]) \
                and not _NEGATED_AFTER.search(sentence[m.end():m.end() + 40]):
            return True
    for m in _ICU_DEATH.finditer(sentence):
        if not _PAST_EPISODE.search(_clause(sentence, m.start(), m.end())) \
                and not _NEGATED.search(sentence[:m.start()]) \
                and not _NEGATED_AFTER.search(sentence[m.end():m.end() + 40]):
            return True
    enc = _ENCOUNTER.search(sentence)
    if enc and _FINDING.search(sentence, enc.end()):
        return True
    if enc and not _PAST_ADMISSION.search(sentence, 0, enc.end()):
        # a study diagnosis named as what the patient presents with; not when the
        # encounter is an earlier one ("recently admitted ... now presents")
        for m in _STUDY_DX.finditer(sentence, enc.end()):
            if _HISTORY_NEAR.search(sentence[:m.start()]) or \
                    _PAST_EPISODE.search(sentence, enc.end(), m.start()) or \
                    _ENCOUNTER.search(sentence, m.end()):
                continue
            return True
    for m in _STUDY_DX.finditer(sentence):
        res = None
        for res in _TEST_RESULT.finditer(sentence, max(0, m.start() - 100), m.start()):
            pass
        if res is None:
            continue
        hist = _HISTORY_NEAR.search(sentence[:m.start()])
        if hist is None or hist.start() < res.start():
            return True
    return False


def final_labels(sentences: List[dict], labels: List[str]) -> List[str]:
    """Model labels with the rule layer applied (vetoed K sentences become D)."""
    return ["D" if lab == "K" and rule_veto(s["text"]) else lab for s, lab in zip(sentences, labels)]


def cleaned_text(sentences: List[dict], labels: List[str]) -> str:
    """Kept (K) sentences after the rule layer, verbatim, with the original paragraph breaks."""
    paras: dict = {}
    for s, lab in zip(sentences, final_labels(sentences, labels)):
        if lab == "K":
            paras.setdefault(s["para"], []).append(s["text"])
    return "\n".join(" ".join(v) for _, v in sorted(paras.items()))


# -- Use in a case-input / external-validation build step ---------------------

def label_input(hadm_id: int, note: str) -> Optional[dict]:
    """The labelling item for one note (input line for hpi_clean_label.py), or None if it has no HPI."""
    sentences = split_sentences(raw_hpi(note))
    if not sentences:
        return None
    return {"id": f"c{hadm_id}", "set": "cohort", "hadm_id": hadm_id, "sentences": sentences}


def load_labels(path: str) -> dict:
    """hadm_id -> labelled cohort record, for records that parsed."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("set") == "cohort" and r.get("labels"):
                    out[r["hadm_id"]] = r
    except FileNotFoundError:
        pass
    return out


def clean_hpi(note: str, record: Optional[dict]) -> Optional[str]:
    """Cleaned HPI of a note, or None if the note's labels are missing or were made
    from different sentences (the note changed since labelling)."""
    sentences = split_sentences(raw_hpi(note))
    if record is None or record["sentences"] != sentences:
        return None
    return cleaned_text(sentences, record["labels"])
