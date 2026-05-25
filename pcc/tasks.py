"""Task definitions: prompt builders, parsers, samplers, dataset loaders.

A task is a Task dataclass with eight fields; everything downstream is generic.
"""
import random
import re
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Any


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
    num_classes: Optional[int] = None


def effective_k_shots(cfg, task):
    if getattr(cfg, "k_shots", None) is not None:
        return cfg.k_shots
    return task.default_k_shots


# Class-balanced sampler. Rotates the label order based on the query example's
# label to make the FS demos look "natural" (the query's label shows up first).
def make_balanced_sampler(label_key, num_classes):
    def sampler(pool, k, rng, example=None):
        by_label = {}
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


# --- SST-2 -------------------------------------------------------------------

SST2_V = {0: "Negative", 1: "Positive"}


def _sst2_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")
    full = [{"text": ex["sentence"].strip(), "label": int(ex["label"])}
            for ex in ds["train"]]
    by_c = {0: [], 1: []}
    for ex in full:
        by_c[ex["label"]].append(ex)
    per_class = max(1, cfg.n_train_pool // 2)
    demo_pool = by_c[0][:per_class] + by_c[1][:per_class]
    n_cal_pc = max(10, cfg.n_calib_pool // 2)
    calib_pool = by_c[0][per_class:per_class + n_cal_pc] + by_c[1][per_class:per_class + n_cal_pc]
    random.Random(42).shuffle(calib_pool)
    eval_set = [{"text": ex["sentence"].strip(), "label": int(ex["label"])}
                for ex in ds["validation"] if ex["label"] != -1]
    return demo_pool, calib_pool, eval_set[:cfg.n_eval]


_SST2_INSTR = ("Classify the sentiment of the following sentence as Positive or Negative.\n"
               "Answer with only the sentiment label.\n\n")


def _sst2_prompt(ex, shots, with_fs):
    fmt = lambda x: f"Sentence: {x['text']}\nSentiment: {SST2_V[x['label']]}"
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    return _SST2_INSTR + body + f"Sentence: {ex['text']}\nSentiment:"


def _sst2_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0].lower()
    if "positive" in first: return "Positive"
    if "negative" in first: return "Negative"
    return None


SST2 = Task(
    name="sst2",
    load_data=_sst2_load,
    build_prompt=_sst2_prompt,
    parse_pred=_sst2_parse,
    gold=lambda ex: SST2_V[ex["label"]],
    score=lambda p, ex: p == SST2_V[ex["label"]],
    sample_demos=make_balanced_sampler("label", 2),
    default_k_shots=4,
    max_new_tokens=5,
    recommended_max_seq_len=512,
    num_classes=2,
)


# --- MRPC --------------------------------------------------------------------

MRPC_V = {0: "Not Equivalent", 1: "Equivalent"}


def _mrpc_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("glue", "mrpc")
    full = [{"sentence1": ex["sentence1"], "sentence2": ex["sentence2"], "label": int(ex["label"])}
            for ex in ds["train"]]
    by_c = {0: [], 1: []}
    for ex in full:
        by_c[ex["label"]].append(ex)
    per_class = max(1, cfg.n_train_pool // 2)
    demo_pool = by_c[0][:per_class] + by_c[1][:per_class]
    n_cal_pc = max(10, cfg.n_calib_pool // 2)
    calib_pool = by_c[0][per_class:per_class + n_cal_pc] + by_c[1][per_class:per_class + n_cal_pc]
    random.Random(42).shuffle(calib_pool)
    eval_set = [{"sentence1": ex["sentence1"], "sentence2": ex["sentence2"], "label": int(ex["label"])}
                for ex in ds["validation"]]
    return demo_pool, calib_pool, eval_set[:cfg.n_eval]


_MRPC_INSTR = ("Decide whether the two sentences are paraphrases (Equivalent) or not (Not Equivalent).\n"
               "Answer with only the relationship label.\n\n")


def _mrpc_prompt(ex, shots, with_fs):
    fmt = lambda x: (f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}\n"
                     f"Relationship: {MRPC_V[x['label']]}")
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    return (_MRPC_INSTR + body
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
    gold=lambda ex: MRPC_V[ex["label"]],
    score=lambda p, ex: p == MRPC_V[ex["label"]],
    sample_demos=make_balanced_sampler("label", 2),
    default_k_shots=4,
    max_new_tokens=8,
    recommended_max_seq_len=512,
    num_classes=2,
)


# --- AG News -----------------------------------------------------------------

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def _ag_load(cfg):
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
    n_cal_pc = max(10, cfg.n_calib_pool // 4)
    calib_pool = []
    for c in range(4):
        calib_pool.extend(by_c[c][per_class:per_class + n_cal_pc])
    random.Random(42).shuffle(calib_pool)
    eval_set = [{"text": ex["text"].strip(), "label": int(ex["label"])} for ex in ds["test"]]
    return demo_pool, calib_pool, eval_set[:cfg.n_eval]


_AG_INSTR = ("You are a news topic classifier. "
             "Classify the following news article into exactly one of these labels: "
             "World, Sports, Business, Sci/Tech.\n"
             "Answer with only the label name.\n\n")


def _ag_prompt(ex, shots, with_fs):
    fmt = lambda x: f"Article: {x['text'].strip()}\nTopic: {AG_LABELS[x['label']]}"
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    return _AG_INSTR + body + f"Article: {ex['text'].strip()}\nTopic:"


def _ag_parse(text):
    # robust to "science and tech", "sci-tech", etc.
    if text is None: return None
    t = text.lower().strip().replace("-", " ").replace("/", " / ").replace("&", " and ")
    t = " ".join(t.split())
    if ("sci" in t and "tech" in t) or "science" in t or "technology" in t:
        return "Sci/Tech"
    if "world" in t or "international" in t: return "World"
    if "sport" in t: return "Sports"
    if "business" in t or "finance" in t or "econom" in t: return "Business"
    return None


AG_NEWS = Task(
    name="ag_news",
    load_data=_ag_load,
    build_prompt=_ag_prompt,
    parse_pred=_ag_parse,
    gold=lambda ex: AG_LABELS[ex["label"]],
    score=lambda p, ex: p == AG_LABELS[ex["label"]],
    sample_demos=make_balanced_sampler("label", 4),
    default_k_shots=6,
    max_new_tokens=5,
    recommended_max_seq_len=1024,
    num_classes=4,
)


# --- DBPedia-14 --------------------------------------------------------------

# NOTE: keep the labels space-separated (e.g. "Educational Institution" not
# "EducationalInstitution"). Models produce the spaced version and we had a
# bug for a while where the parser only matched the concatenated string.
DBP_LABELS = [
    "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
    "Mean of Transportation", "Building", "Natural Place", "Village", "Animal",
    "Plant", "Album", "Film", "Written Work",
]
DBP_MAX_CONTENT = 100  # truncate long Wikipedia summaries


def _dbp_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("dbpedia_14")

    def norm(ex):
        return {"title": (ex.get("title") or "").strip(),
                "content": ex["content"].strip()[:DBP_MAX_CONTENT],
                "label": int(ex["label"])}

    full = [norm(ex) for ex in ds["train"]]
    by_class = {c: [] for c in range(14)}
    for ex in full:
        by_class[ex["label"]].append(ex)

    per_class = max(1, cfg.n_train_pool // 14)
    demo_pool = []
    for c in range(14):
        demo_pool.extend(by_class[c][:per_class])

    # Empirically need a bit more headroom per class on dbpedia.
    per_class_cal = max(5, cfg.n_calib_pool // 14 + 5)
    calib_pool = []
    for c in range(14):
        calib_pool.extend(by_class[c][per_class:per_class + per_class_cal])
    random.Random(42).shuffle(calib_pool)

    eval_set = [norm(ex) for ex in ds["test"]][:cfg.n_eval]
    return demo_pool, calib_pool, eval_set


_DBP_INSTR = ("You are a topic classifier. "
              "Classify each text into exactly one of these categories: "
              "Company, Educational Institution, Artist, Athlete, Office Holder, "
              "Mean of Transportation, Building, Natural Place, Village, Animal, "
              "Plant, Album, Film, Written Work.\n"
              "Answer with only the category name.\n\n")


def _dbp_prompt(ex, shots, with_fs):
    def fmt(x):
        title = x.get("title", "").strip()
        content = x["content"].strip()[:DBP_MAX_CONTENT]
        head = f"Title: {title}\n" if title else ""
        return f"{head}Text: {content}\nCategory: {DBP_LABELS[x['label']]}"
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    title = ex.get("title", "").strip()
    head = f"Title: {title}\n" if title else ""
    content = ex["content"].strip()[:DBP_MAX_CONTENT]
    return _DBP_INSTR + body + f"{head}Text: {content}\nCategory:"


def _dbp_parse(text):
    # longest first so "Educational Institution" wins over "Institution"
    if text is None: return None
    first = text.strip().split("\n")[0].lower()
    for lab in sorted(DBP_LABELS, key=len, reverse=True):
        if lab.lower() in first:
            return lab
    return None


DBPEDIA = Task(
    name="dbpedia_14",
    load_data=_dbp_load,
    build_prompt=_dbp_prompt,
    parse_pred=_dbp_parse,
    gold=lambda ex: DBP_LABELS[ex["label"]],
    score=lambda p, ex: p == DBP_LABELS[ex["label"]],
    sample_demos=make_balanced_sampler("label", 14),
    default_k_shots=14,
    max_new_tokens=10,
    recommended_max_seq_len=1024,
    num_classes=14,
)


# --- BoolQ -------------------------------------------------------------------

BOOLQ_V = {0: "No", 1: "Yes"}


def _boolq_load(cfg):
    from datasets import load_dataset
    try:
        ds = load_dataset("super_glue", "boolq")
    except Exception:
        # HF has been flaky on this mirror; try the aps fork
        ds = load_dataset("aps/super_glue", "boolq")

    def norm(ex):
        return {"passage": ex["passage"].strip(),
                "question": ex["question"].strip(),
                "label": int(ex["label"])}

    full = [norm(ex) for ex in ds["train"]]
    by_class = {0: [], 1: []}
    for ex in full:
        by_class[ex["label"]].append(ex)

    per_demo = max(1, cfg.n_train_pool // 2)
    per_cal = max(10, cfg.n_calib_pool // 2)
    demo_pool, calib_pool = [], []
    for c in (0, 1):
        demo_pool.extend(by_class[c][:per_demo])
        calib_pool.extend(by_class[c][per_demo:per_demo + per_cal])
    random.Random(42).shuffle(calib_pool)

    eval_set = [norm(ex) for ex in ds["validation"] if ex["label"] != -1][:cfg.n_eval]
    return demo_pool, calib_pool, eval_set


_BOOLQ_INSTR = "Read the passage and answer the question with 'Yes' or 'No'.\n\n"


def _boolq_prompt(ex, shots, with_fs):
    fmt = lambda x: (f"Passage: {x['passage']}\nQuestion: {x['question']}\n"
                     f"Answer: {BOOLQ_V[x['label']]}")
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    return (_BOOLQ_INSTR + body
            + f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:")


def _boolq_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0].lower()
    if "yes" in first or "true" in first: return "Yes"
    if "no" in first or "false" in first: return "No"
    return None


BOOLQ = Task(
    name="boolq",
    load_data=_boolq_load,
    build_prompt=_boolq_prompt,
    parse_pred=_boolq_parse,
    gold=lambda ex: BOOLQ_V[ex["label"]],
    score=lambda p, ex: p == BOOLQ_V[ex["label"]],
    sample_demos=make_balanced_sampler("label", 2),
    default_k_shots=2,  # passages are long, more shots blows the context
    max_new_tokens=4,
    recommended_max_seq_len=1536,
    num_classes=2,
)


# --- ARC-Challenge -----------------------------------------------------------

ARC_LABELS = {0: "A", 1: "B", 2: "C", 3: "D"}
ARC_LAB2IDX = {v: k for k, v in ARC_LABELS.items()}
_ARC_RE = re.compile(r"\b([ABCD])\b")


def _arc_normalize(ex):
    """Drop any row that isn't a clean 4-choice question."""
    labels = list(ex["choices"]["label"])
    texts = list(ex["choices"]["text"])
    if len(labels) != 4 or len(texts) != 4:
        return None
    ans = ex["answerKey"]
    if ans in ARC_LAB2IDX:
        idx = ARC_LAB2IDX[ans]
    elif isinstance(ans, str) and ans.isdigit() and 1 <= int(ans) <= 4:
        idx = int(ans) - 1
    elif ans in labels:
        idx = labels.index(ans)
    else:
        return None
    return {"question": ex["question"].strip(), "choices": texts, "label": idx}


def _arc_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Challenge")

    def take(split):
        out = []
        for ex in ds[split]:
            n = _arc_normalize(ex)
            if n is not None:
                out.append(n)
        return out

    train = take("train") + take("validation")
    test = take("test")
    rng = random.Random(42)
    rng.shuffle(train); rng.shuffle(test)

    by_label = {c: [] for c in range(4)}
    for ex in train:
        by_label[ex["label"]].append(ex)

    per_demo = max(1, cfg.n_train_pool // 4)
    per_cal = max(5, cfg.n_calib_pool // 4)
    demo_pool, calib_pool = [], []
    for c in range(4):
        demo_pool.extend(by_label[c][:per_demo])
        calib_pool.extend(by_label[c][per_demo:per_demo + per_cal])
    return demo_pool, calib_pool, test[:cfg.n_eval]


_ARC_INSTR = ("Answer the following multiple-choice science question.\n"
              "Reply with only the single letter (A, B, C, or D) of the correct answer.\n\n")


def _arc_prompt(ex, shots, with_fs):
    def fmt(x):
        c = x["choices"]
        return (f"Question: {x['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
                f"Answer: {ARC_LABELS[x['label']]}")
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    c = ex["choices"]
    return (_ARC_INSTR + body
            + f"Question: {ex['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")


def _arc_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0]
    m = _ARC_RE.search(first.upper())
    return m.group(1) if m else None


ARC_CHALLENGE = Task(
    name="arc_challenge",
    load_data=_arc_load,
    build_prompt=_arc_prompt,
    parse_pred=_arc_parse,
    gold=lambda ex: ARC_LABELS[ex["label"]],
    score=lambda p, ex: p == ARC_LABELS[ex["label"]],
    sample_demos=make_balanced_sampler("label", 4),
    default_k_shots=4,
    max_new_tokens=4,
    recommended_max_seq_len=1024,
    num_classes=4,
)


# --- GSM8K -------------------------------------------------------------------

_GSM_NUM = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")


def _gsm_extract(text):
    # last number wins (the answer typically appears after "#### ")
    if text is None: return None
    nums = _GSM_NUM.findall(text.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None


# Wei et al. canonical 8 CoT shots.
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
    n_cal = cfg.n_calib_pool
    if getattr(cfg, "use_canonical_demos", True):
        demo_pool = list(GSM_CANONICAL_8)
        calib_pool = list(load_dataset("gsm8k", "main", split=f"train[:{n_cal}]"))
    else:
        n_demo = cfg.n_train_pool
        full = list(load_dataset("gsm8k", "main", split=f"train[:{n_demo + n_cal}]"))
        # sort by length so short ones go first into the demo pool
        demo_pool = sorted(full[:n_demo], key=lambda ex: (len(ex["answer"]), len(ex["question"])))
        calib_pool = full[n_demo:]
    return demo_pool, calib_pool, test


def _gsm_prompt(ex, shots, with_fs):
    fmt = lambda x: f"Question: {x['question'].strip()}\nSolution: {x['answer'].strip()}"
    if with_fs and shots:
        block = "\n\n".join(fmt(s) for s in shots)
        return (f"[INST] Solve math word problems step by step. "
                f"At the end write the answer as: #### <number>\n\n"
                f"Here are some solved examples:\n\n{block}\n\n"
                f"Now solve this:\nQuestion: {ex['question'].strip()}\nSolution: [/INST]")
    return (f"[INST] Solve the following math word problem step by step. "
            f"At the end write the answer as: #### <number>\n\n"
            f"Question: {ex['question'].strip()}\nSolution: [/INST]")


GSM8K = Task(
    name="gsm8k",
    load_data=_gsm_load,
    build_prompt=_gsm_prompt,
    parse_pred=_gsm_extract,
    gold=lambda ex: _gsm_extract(ex["answer"]),
    score=lambda p, ex: p is not None and p == _gsm_extract(ex["answer"]),
    sample_demos=canonical_first_sampler,
    default_k_shots=8,
    canonical_demos=GSM_CANONICAL_8,
    max_new_tokens=256,
    recommended_max_seq_len=2048,
    num_classes=None,
)


# --- MMLU --------------------------------------------------------------------

MMLU_LABELS = {0: "A", 1: "B", 2: "C", 3: "D"}
_MMLU_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

# Hendrycks protocol uses subject-matched dev shots. We cache the dev set
# grouped by subject in this module-level dict (populated by _mmlu_load).
# TODO: this is ugly global state. Should pass via cfg.
_MMLU_DEV_BY_SUBJECT = {}


def _mmlu_normalize(ex):
    ans = ex["answer"]
    if isinstance(ans, str):
        ans = {"A": 0, "B": 1, "C": 2, "D": 3}[ans.strip().upper()]
    return {"question": ex["question"].strip(),
            "choices": list(ex["choices"]),
            "label": int(ans),
            "subject": ex.get("subject", "general")}


def _mmlu_load(cfg):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all")
    dev_rows = [_mmlu_normalize(ex) for ex in ds["dev"]]

    global _MMLU_DEV_BY_SUBJECT
    _MMLU_DEV_BY_SUBJECT = {}
    for ex in dev_rows:
        _MMLU_DEV_BY_SUBJECT.setdefault(ex["subject"], []).append(ex)

    val_rows = [_mmlu_normalize(ex) for ex in ds["validation"]]
    random.Random(42).shuffle(val_rows)

    by_label = {c: [] for c in range(4)}
    for ex in val_rows:
        by_label[ex["label"]].append(ex)
    calib_pool = []
    per = max(1, cfg.n_calib_pool // 4)
    for c in range(4):
        calib_pool.extend(by_label[c][:per])
    random.Random(42).shuffle(calib_pool)

    test_rows = [_mmlu_normalize(ex) for ex in ds["test"]]
    random.Random(42).shuffle(test_rows)
    return dev_rows, calib_pool, test_rows[:cfg.n_eval]


def _mmlu_subject_sampler(train_pool, k, rng, example=None):
    """Hendrycks 5-shot: prefer dev examples from the same subject as the query."""
    if example is not None:
        subj = example.get("subject", "")
        pool = _MMLU_DEV_BY_SUBJECT.get(subj, [])
        if pool:
            shots = pool[:k]
            if len(shots) == k:
                return shots
            # not enough in-subject — pad with random out-of-subject
            others = [ex for ex in train_pool if ex.get("subject") != subj]
            rng.shuffle(others)
            return shots + others[:k - len(shots)]
    # fallback: balanced by label
    by_label = {c: [] for c in range(4)}
    for ex in train_pool:
        by_label[ex["label"]].append(ex)
    base, rem = k // 4, k % 4
    shots = []
    for i, c in enumerate(range(4)):
        take = base + (1 if i < rem else 0)
        pool = by_label[c][:]
        rng.shuffle(pool)
        shots.extend(pool[:take])
    return shots


def _mmlu_prompt(ex, shots, with_fs):
    subject = ex.get("subject", "general").replace("_", " ")
    instr = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    def fmt(x):
        c = x["choices"]
        return (f"Question: {x['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
                f"Answer: {MMLU_LABELS[x['label']]}")
    body = ("\n\n".join(fmt(s) for s in shots) + "\n\n") if (with_fs and shots) else ""
    c = ex["choices"]
    return (instr + body
            + f"Question: {ex['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")


def _mmlu_parse(text):
    if text is None: return None
    first = text.strip().split("\n")[0]
    m = _MMLU_RE.search(first.upper())
    return m.group(1).upper() if m else None


MMLU = Task(
    name="mmlu",
    load_data=_mmlu_load,
    build_prompt=_mmlu_prompt,
    parse_pred=_mmlu_parse,
    gold=lambda ex: MMLU_LABELS[ex["label"]],
    score=lambda p, ex: p == MMLU_LABELS[ex["label"]],
    sample_demos=_mmlu_subject_sampler,
    default_k_shots=5,
    canonical_demos=None,
    max_new_tokens=4,
    recommended_max_seq_len=2048,
    num_classes=4,
)


# --- registry ----------------------------------------------------------------

TASKS = {
    "sst2": SST2,
    "mrpc": MRPC,
    "ag_news": AG_NEWS,
    "dbpedia_14": DBPEDIA,
    "boolq": BOOLQ,
    "arc_challenge": ARC_CHALLENGE,
    "gsm8k": GSM8K,
    "mmlu": MMLU,
}


def get_task(name):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}. options: {list(TASKS)}")
    return TASKS[name]
