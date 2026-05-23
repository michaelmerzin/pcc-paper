"""Task registry for all 8 paper benchmarks.

Tasks (paper §4):
    sst2          — binary sentiment (GLUE)
    mrpc          — paraphrase (GLUE)
    ag_news       — 4-class topic
    dbpedia_14    — 14-class topic
    boolq         — yes/no reading comprehension (SuperGLUE)
    arc_challenge — multiple-choice science (4 choices)
    gsm8k         — grade-school math word problems (canonical 8-shot CoT)
    mmlu          — multitask language understanding (5-shot, n=500)

Each task is a Task dataclass with:
  - load_data(cfg)              → (demo_pool, calib_pool, eval_set)
  - build_prompt(ex, shots, fs) → str
  - parse_pred(text)            → parsed prediction (str/None)
  - gold(ex)                    → reference answer string
  - score(pred, ex)             → bool
  - sample_demos(pool, k, rng)  → list of demos (balanced for classification)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Any
import random
import re


@dataclass
class Task:
    name: str
    load_data: Callable
    build_prompt: Callable
    parse_pred: Callable
    gold: Callable
    score: Callable
    sample_demos: Callable
    default_k_shots: int = 4
    canonical_demos: Optional[List[Dict[str, Any]]] = None
    max_new_tokens: int = 8
    recommended_max_seq_len: int = 1024
    num_classes: Optional[int] = None      # None = generative


# =========================================================================== #
#   Generic class-balanced demo sampler                                       #
# =========================================================================== #
def make_balanced_sampler(label_key: str, num_classes: int):
    """Pick k shots balanced across classes, rotating order by example label."""
    def sampler(pool, k, rng, example=None):
        by_label: Dict[int, List] = {}
        for ex in pool:
            by_label.setdefault(int(ex[label_key]), []).append(ex)

        labels = list(range(num_classes))
        if example is not None and label_key in example:
            pos = int(example[label_key]) % num_classes
            labels = labels[pos:] + labels[:pos]

        base, rem = k // num_classes, k % num_classes
        shots = []
        for i, y in enumerate(labels):
            take = base + (1 if i < rem else 0)
            avail = [x for x in by_label.get(y, []) if x is not example]
            if avail and take > 0:
                shots.extend(rng.sample(avail, k=min(take, len(avail))))
        return shots
    return sampler


def random_sampler(pool, k, rng, example=None):
    avail = [x for x in pool if x is not example]
    return rng.sample(avail, k=min(k, len(avail)))


def canonical_first_sampler(pool, k, rng, example=None):
    return list(pool[:k])


# =========================================================================== #
#   SST-2 — binary sentiment (GLUE)                                           #
# =========================================================================== #
SST2_VERBALIZER = {0: "Negative", 1: "Positive"}


def _sst2_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")
    full = [{"text": ex["sentence"].strip(), "label": int(ex["label"])} for ex in ds["train"]]
    by_c = {0: [], 1: []}
    for ex in full:
        by_c[ex["label"]].append(ex)

    per_class = max(1, cfg.n_train_pool // 2)
    demo_pool = by_c[0][:per_class] + by_c[1][:per_class]

    n_calib_per_class = max(10, cfg.n_calib_pool // 2)
    calib_pool = (by_c[0][per_class:per_class + n_calib_per_class]
                  + by_c[1][per_class:per_class + n_calib_per_class])
    rng = random.Random(42); rng.shuffle(calib_pool)

    eval_set = [{"text": ex["sentence"].strip(), "label": int(ex["label"])}
                for ex in ds["validation"] if ex["label"] != -1]
    eval_set = eval_set[:min(cfg.n_eval, len(eval_set))]
    return demo_pool, calib_pool, eval_set


_SST2_INSTRUCTION = (
    "Classify the sentiment of the following sentence as Positive or Negative.\n"
    "Answer with only the sentiment label.\n\n"
)


def _sst2_prompt(ex, shots, with_fs):
    body = ""
    if with_fs:
        for s in shots:
            body += f"Sentence: {s['text']}\nSentiment: {SST2_VERBALIZER[s['label']]}\n\n"
    return _SST2_INSTRUCTION + body + f"Sentence: {ex['text']}\nSentiment:"


def _sst2_parse(text):
    if text is None:
        return None
    first = text.strip().split("\n")[0].lower()
    if "positive" in first: return "Positive"
    if "negative" in first: return "Negative"
    return None


SST2 = Task(
    name="sst2",
    load_data=_sst2_load,
    build_prompt=_sst2_prompt,
    parse_pred=_sst2_parse,
    gold=lambda ex: SST2_VERBALIZER[ex["label"]],
    score=lambda p, ex: p == SST2_VERBALIZER[ex["label"]],
    sample_demos=make_balanced_sampler("label", 2),
    default_k_shots=4,
    max_new_tokens=4,
    recommended_max_seq_len=512,
    num_classes=2,
)


# =========================================================================== #
#   MRPC — paraphrase (GLUE)                                                  #
# =========================================================================== #
MRPC_VERBALIZER = {0: "Not Equivalent", 1: "Equivalent"}


def _mrpc_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("glue", "mrpc")
    full = [{"sentence1": ex["sentence1"], "sentence2": ex["sentence2"],
             "label": int(ex["label"])} for ex in ds["train"]]
    by_c = {0: [], 1: []}
    for ex in full:
        by_c[ex["label"]].append(ex)
    per_class = max(1, cfg.n_train_pool // 2)
    demo_pool = by_c[0][:per_class] + by_c[1][:per_class]

    n_calib_per_class = max(10, cfg.n_calib_pool // 2)
    calib_pool = (by_c[0][per_class:per_class + n_calib_per_class]
                  + by_c[1][per_class:per_class + n_calib_per_class])
    rng = random.Random(42); rng.shuffle(calib_pool)

    eval_set = [{"sentence1": ex["sentence1"], "sentence2": ex["sentence2"],
                 "label": int(ex["label"])} for ex in ds["validation"]]
    eval_set = eval_set[:min(cfg.n_eval, len(eval_set))]
    return demo_pool, calib_pool, eval_set


_MRPC_INSTRUCTION = (
    "Decide whether the two sentences are paraphrases (Equivalent) or not (Not Equivalent).\n"
    "Answer with only the relationship label.\n\n"
)


def _mrpc_prompt(ex, shots, with_fs):
    body = ""
    if with_fs:
        for s in shots:
            body += (f"Sentence 1: {s['sentence1']}\nSentence 2: {s['sentence2']}\n"
                     f"Relationship: {MRPC_VERBALIZER[s['label']]}\n\n")
    return (_MRPC_INSTRUCTION + body
            + f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nRelationship:")


def _mrpc_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0].lower()
    if "not equivalent" in first or "not_equivalent" in first:
        return "Not Equivalent"
    if "equivalent" in first:
        return "Equivalent"
    return None


MRPC = Task(
    name="mrpc",
    load_data=_mrpc_load,
    build_prompt=_mrpc_prompt,
    parse_pred=_mrpc_parse,
    gold=lambda ex: MRPC_VERBALIZER[ex["label"]],
    score=lambda p, ex: p == MRPC_VERBALIZER[ex["label"]],
    sample_demos=make_balanced_sampler("label", 2),
    default_k_shots=4,
    max_new_tokens=6,
    recommended_max_seq_len=768,
    num_classes=2,
)


# =========================================================================== #
#   AG News — 4-class topic                                                   #
# =========================================================================== #
AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def _ag_news_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("ag_news")
    full = [{"text": ex["text"].strip(), "label": int(ex["label"])} for ex in ds["train"]]
    by_c = {i: [] for i in range(4)}
    for ex in full:
        by_c[ex["label"]].append(ex)
    per_class = max(1, cfg.n_train_pool // 4)
    demo_pool = []
    for c in range(4):
        demo_pool.extend(by_c[c][:per_class])

    n_calib_per_class = max(10, cfg.n_calib_pool // 4)
    calib_pool = []
    for c in range(4):
        calib_pool.extend(by_c[c][per_class:per_class + n_calib_per_class])
    rng = random.Random(42); rng.shuffle(calib_pool)

    eval_set = [{"text": ex["text"].strip(), "label": int(ex["label"])} for ex in ds["test"]]
    eval_set = eval_set[:min(cfg.n_eval, len(eval_set))]
    return demo_pool, calib_pool, eval_set


_AG_INSTRUCTION = (
    "Classify the following news article into one of these topics: World, Sports, Business, Sci/Tech.\n"
    "Answer with only the topic name.\n\n"
)


def _ag_prompt(ex, shots, with_fs):
    body = ""
    if with_fs:
        for s in shots:
            body += f"Article: {s['text']}\nTopic: {AG_NEWS_LABELS[s['label']]}\n\n"
    return _AG_INSTRUCTION + body + f"Article: {ex['text']}\nTopic:"


_AG_SYNONYMS = {
    "World": ["world", "international", "politics"],
    "Sports": ["sports", "sport"],
    "Business": ["business", "finance", "economy"],
    "Sci/Tech": ["sci/tech", "sci-tech", "science", "technology", "tech"],
}


def _ag_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0].lower()
    for label, syns in _AG_SYNONYMS.items():
        for s in syns:
            if s in first:
                return label
    return None


AG_NEWS = Task(
    name="ag_news",
    load_data=_ag_news_load,
    build_prompt=_ag_prompt,
    parse_pred=_ag_parse,
    gold=lambda ex: AG_NEWS_LABELS[ex["label"]],
    score=lambda p, ex: p == AG_NEWS_LABELS[ex["label"]],
    sample_demos=make_balanced_sampler("label", 4),
    default_k_shots=4,
    max_new_tokens=6,
    recommended_max_seq_len=1024,
    num_classes=4,
)


# =========================================================================== #
#   DBPedia-14 — 14-class topic                                               #
# =========================================================================== #
DBPEDIA_LABELS = [
    "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
    "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
    "Plant", "Album", "Film", "WrittenWork",
]


def _dbp_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("dbpedia_14")
    full = [{"content": ex["content"].strip(), "label": int(ex["label"])} for ex in ds["train"]]
    by_c = {i: [] for i in range(14)}
    for ex in full:
        by_c[ex["label"]].append(ex)
    per_class = max(1, cfg.n_train_pool // 14)
    demo_pool = []
    for c in range(14):
        demo_pool.extend(by_c[c][:per_class])

    n_calib_per_class = max(5, cfg.n_calib_pool // 14)
    calib_pool = []
    for c in range(14):
        calib_pool.extend(by_c[c][per_class:per_class + n_calib_per_class])
    rng = random.Random(42); rng.shuffle(calib_pool)

    eval_set = [{"content": ex["content"].strip(), "label": int(ex["label"])} for ex in ds["test"]]
    eval_set = eval_set[:min(cfg.n_eval, len(eval_set))]
    return demo_pool, calib_pool, eval_set


_DBP_INSTRUCTION = (
    "Classify the following text into one of these 14 DBPedia categories:\n"
    "Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation,\n"
    "Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.\n"
    "Answer with only the category name.\n\n"
)


def _dbp_prompt(ex, shots, with_fs):
    body = ""
    if with_fs:
        for s in shots:
            body += f"Text: {s['content']}\nCategory: {DBPEDIA_LABELS[s['label']]}\n\n"
    return _DBP_INSTRUCTION + body + f"Text: {ex['content']}\nCategory:"


def _dbp_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0]
    # Greedy: try exact match first, then case-insensitive substring
    for label in DBPEDIA_LABELS:
        if label.lower() in first.lower():
            return label
    return None


DBPEDIA = Task(
    name="dbpedia_14",
    load_data=_dbp_load,
    build_prompt=_dbp_prompt,
    parse_pred=_dbp_parse,
    gold=lambda ex: DBPEDIA_LABELS[ex["label"]],
    score=lambda p, ex: p == DBPEDIA_LABELS[ex["label"]],
    sample_demos=make_balanced_sampler("label", 14),
    default_k_shots=4,
    max_new_tokens=12,
    recommended_max_seq_len=1024,
    num_classes=14,
)


# =========================================================================== #
#   BoolQ — yes/no                                                            #
# =========================================================================== #
BOOLQ_VERBALIZER = {False: "No", True: "Yes"}


def _boolq_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("boolq")
    full = [{"question": ex["question"], "passage": ex["passage"],
             "label": bool(ex["answer"])} for ex in ds["train"]]
    by_c = {True: [], False: []}
    for ex in full:
        by_c[ex["label"]].append(ex)
    per_class = max(1, cfg.n_train_pool // 2)
    demo_pool = by_c[True][:per_class] + by_c[False][:per_class]

    n_calib_per_class = max(10, cfg.n_calib_pool // 2)
    calib_pool = (by_c[True][per_class:per_class + n_calib_per_class]
                  + by_c[False][per_class:per_class + n_calib_per_class])
    rng = random.Random(42); rng.shuffle(calib_pool)

    eval_set = [{"question": ex["question"], "passage": ex["passage"],
                 "label": bool(ex["answer"])} for ex in ds["validation"]]
    eval_set = eval_set[:min(cfg.n_eval, len(eval_set))]
    return demo_pool, calib_pool, eval_set


def _boolq_prompt(ex, shots, with_fs):
    instr = "Answer the question Yes or No based on the passage.\n\n"
    body = ""
    if with_fs:
        for s in shots:
            body += (f"Passage: {s['passage']}\nQuestion: {s['question']}\n"
                     f"Answer: {BOOLQ_VERBALIZER[s['label']]}\n\n")
    return instr + body + f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"


def _boolq_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0].lower()
    if "yes" in first.split()[:3]: return "Yes"
    if "no" in first.split()[:3]: return "No"
    return None


BOOLQ = Task(
    name="boolq",
    load_data=_boolq_load,
    build_prompt=_boolq_prompt,
    parse_pred=_boolq_parse,
    gold=lambda ex: BOOLQ_VERBALIZER[ex["label"]],
    score=lambda p, ex: p == BOOLQ_VERBALIZER[ex["label"]],
    sample_demos=lambda pool, k, rng, example=None: make_balanced_sampler(
        "label", 2)([{**e, "label": int(e["label"])} for e in pool], k, rng, example),
    default_k_shots=4,
    max_new_tokens=4,
    recommended_max_seq_len=1536,
    num_classes=2,
)


# =========================================================================== #
#   ARC-Challenge — 4-way multiple choice                                     #
# =========================================================================== #
def _arc_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    def _norm(ex):
        return {
            "question": ex["question"],
            "choices": ex["choices"]["text"],
            "labels": ex["choices"]["label"],
            "answer": ex["answerKey"],
        }
    full_train = [_norm(ex) for ex in ds["train"]]
    demo_pool = full_train[:cfg.n_train_pool]
    calib_pool = full_train[cfg.n_train_pool:cfg.n_train_pool + cfg.n_calib_pool]
    eval_set = [_norm(ex) for ex in ds["validation"]]
    eval_set = eval_set[:min(cfg.n_eval, len(eval_set))]
    return demo_pool, calib_pool, eval_set


def _arc_format_choices(ex):
    return "\n".join(f"{l}. {t}" for l, t in zip(ex["labels"], ex["choices"]))


def _arc_prompt(ex, shots, with_fs):
    instr = "Answer the multiple-choice science question with the letter of the correct choice.\n\n"
    body = ""
    if with_fs:
        for s in shots:
            body += (f"Question: {s['question']}\n{_arc_format_choices(s)}\n"
                     f"Answer: {s['answer']}\n\n")
    return instr + body + f"Question: {ex['question']}\n{_arc_format_choices(ex)}\nAnswer:"


_ARC_LETTER_RE = re.compile(r"\b([A-E1-5])\b")


def _arc_parse(text):
    if text is None: return None
    m = _ARC_LETTER_RE.search(text.strip())
    return m.group(1) if m else None


ARC_CHALLENGE = Task(
    name="arc_challenge",
    load_data=_arc_load,
    build_prompt=_arc_prompt,
    parse_pred=_arc_parse,
    gold=lambda ex: ex["answer"],
    score=lambda p, ex: p == ex["answer"],
    sample_demos=random_sampler,
    default_k_shots=4,
    max_new_tokens=4,
    recommended_max_seq_len=1024,
    num_classes=None,
)


# =========================================================================== #
#   GSM8K — grade-school math (CoT)                                           #
# =========================================================================== #
_GSM_NUM_RE = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")


def _gsm_extract_number(text):
    if text is None: return None
    matches = _GSM_NUM_RE.findall(text.replace("$", ""))
    return matches[-1].replace(",", "") if matches else None


# Canonical 8 GSM8K CoT exemplars from Wei et al. 2022, Appendix G
GSM_CANONICAL_8 = [
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6"},
    {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"},
    {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"},
    {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8"},
    {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9"},
    {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29"},
    {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"},
    {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8"},
]


def _gsm_load(cfg):
    from datasets import load_dataset
    test = list(load_dataset("gsm8k", "main", split=f"test[:{cfg.n_eval}]"))
    n_calib = cfg.n_calib_pool
    if getattr(cfg, "use_canonical_demos", True):
        demo_pool = list(GSM_CANONICAL_8)
        calib_pool = list(load_dataset("gsm8k", "main", split=f"train[:{n_calib}]"))
    else:
        n_demo = cfg.n_train_pool
        full = list(load_dataset("gsm8k", "main", split=f"train[:{n_demo+n_calib}]"))
        demo_pool = sorted(full[:n_demo],
                           key=lambda ex: (len(ex["answer"]), len(ex["question"])))
        calib_pool = full[n_demo:]
    return demo_pool, calib_pool, test


def _gsm_prompt(ex, shots, with_fs):
    """CoT chat-style prompt; works for Mistral-Instruct, Phi-3, Gemma-it."""
    def fmt_shot(s):
        return f"Question: {s['question']}\nSolution: {s['answer']}\n\n"
    instr = "Solve the math problem step by step. End with '#### <answer>'.\n\n"
    body = ""
    if with_fs:
        for s in shots:
            body += fmt_shot(s)
    # Use [INST] wrapper for Mistral-Instruct compatibility — Phi/Gemma accept it
    return f"[INST] {instr}{body}Question: {ex['question']}\nSolution: [/INST]"


def _gsm_parse(text):
    if text is None: return None
    return _gsm_extract_number(text)


def _gsm_score(pred, ex):
    gold = _gsm_extract_number(ex["answer"])
    if pred is None or gold is None: return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


GSM8K = Task(
    name="gsm8k",
    load_data=_gsm_load,
    build_prompt=_gsm_prompt,
    parse_pred=_gsm_parse,
    gold=lambda ex: _gsm_extract_number(ex["answer"]) or "",
    score=_gsm_score,
    sample_demos=canonical_first_sampler,
    default_k_shots=8,
    canonical_demos=GSM_CANONICAL_8,
    max_new_tokens=256,
    recommended_max_seq_len=1536,
    num_classes=None,
)


# =========================================================================== #
#   MMLU — 5-shot, n=500 subject-matched (Hendrycks protocol)                #
# =========================================================================== #
def _mmlu_load(cfg):
    from datasets import load_dataset
    # 'all' config gives the standard joint MMLU.
    ds = load_dataset("cais/mmlu", "all")
    def _norm(ex):
        return {
            "question": ex["question"],
            "choices": ex["choices"],
            "subject": ex["subject"],
            "answer": int(ex["answer"]),
        }
    dev = [_norm(ex) for ex in ds["dev"]]    # used as demos (5-shot in paper)
    val = [_norm(ex) for ex in ds["validation"]]
    test = [_norm(ex) for ex in ds["test"]]
    # demo_pool = dev set (Hendrycks's 5-shot demo source)
    demo_pool = dev
    calib_pool = val[:cfg.n_calib_pool]
    eval_set = test[:min(cfg.n_eval, len(test))]
    return demo_pool, calib_pool, eval_set


def _mmlu_format_choices(ex):
    letters = ["A", "B", "C", "D"]
    return "\n".join(f"{l}. {c}" for l, c in zip(letters, ex["choices"]))


def _mmlu_subject_matched_shots(demo_pool, k, rng, example=None):
    """Hendrycks 5-shot subject-matched: pick k demos with the same subject."""
    if example is not None and "subject" in example:
        same = [d for d in demo_pool if d["subject"] == example["subject"]]
        if len(same) >= k:
            return rng.sample(same, k)
    return rng.sample(demo_pool, k=min(k, len(demo_pool)))


def _mmlu_prompt(ex, shots, with_fs):
    instr = (f"The following are multiple choice questions (with answers) about "
             f"{ex.get('subject', 'various subjects')}.\n\n")
    body = ""
    if with_fs:
        for s in shots:
            body += (f"{s['question']}\n{_mmlu_format_choices(s)}\n"
                     f"Answer: {chr(ord('A') + s['answer'])}\n\n")
    return instr + body + f"{ex['question']}\n{_mmlu_format_choices(ex)}\nAnswer:"


def _mmlu_parse(text):
    if text is None: return None
    m = re.search(r"\b([A-D])\b", text.strip())
    return m.group(1) if m else None


MMLU = Task(
    name="mmlu",
    load_data=_mmlu_load,
    build_prompt=_mmlu_prompt,
    parse_pred=_mmlu_parse,
    gold=lambda ex: chr(ord("A") + ex["answer"]),
    score=lambda p, ex: p == chr(ord("A") + ex["answer"]),
    sample_demos=_mmlu_subject_matched_shots,
    default_k_shots=5,
    max_new_tokens=4,
    recommended_max_seq_len=1024,
    num_classes=None,
)


# =========================================================================== #
#   Registry                                                                  #
# =========================================================================== #
TASKS: Dict[str, Task] = {
    "sst2": SST2,
    "mrpc": MRPC,
    "ag_news": AG_NEWS,
    "dbpedia_14": DBPEDIA,
    "boolq": BOOLQ,
    "arc_challenge": ARC_CHALLENGE,
    "gsm8k": GSM8K,
    "mmlu": MMLU,
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {list(TASKS)}")
    return TASKS[name]


def effective_k_shots(cfg, task: Task) -> int:
    """Return cfg.k_shots if explicitly set, else the task's default."""
    if cfg.k_shots is not None:
        return cfg.k_shots
    return task.default_k_shots
