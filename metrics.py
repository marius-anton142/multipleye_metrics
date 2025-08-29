# -*- coding: utf-8 -*-
import re
import pandas as pd

def iter_sentences(stimuli, nlp):
    for stim in stimuli:
        sid, sname = stim["stimulus_id"], stim["stimulus_name"]
        for pnum, page_text in enumerate(stim["pages"], start=1):
            doc = nlp(page_text)
            for sent_idx, sent in enumerate(doc.sents):
                yield sid, sname, pnum, sent_idx, sent

def iter_pages(stimuli, nlp):
    for stim in stimuli:
        sid, sname = stim["stimulus_id"], stim["stimulus_name"]
        for pnum, page_text in enumerate(stim["pages"], start=1):
            yield sid, sname, pnum, nlp(page_text)

def agg_from_sentence_rows(df, level, sum_map):
    if df.empty:
        return pd.DataFrame()
    if level == "sentence":
        return df.copy()
    if level == "page":
        g = df.groupby(["stimulus_id","stimulus_name","page"], as_index=False)
        out = g.agg(**{new:(pd.NamedAgg(old,"sum")) for old,new in sum_map.items()},
                    n_sentences=pd.NamedAgg("sent_idx","count"))
        return out
    if level == "doc":
        g = df.groupby(["stimulus_id","stimulus_name"], as_index=False)
        out = g.agg(**{new:(pd.NamedAgg(old,"sum")) for old,new in sum_map.items()},
                    n_sentences=pd.NamedAgg("sent_idx","count"))
        return out
    if level == "lang":
        sums = {new: df[old].sum() for old,new in sum_map.items()}
        sums["n_sentences"] = df["sent_idx"].count()
        return pd.DataFrame([{"level":"lang", **sums}])
    return df

def agg_from_page_rows(df, level, sum_map):
    if df.empty:
        return pd.DataFrame()
    if level == "page":
        return df.copy()
    if level == "doc":
        g = df.groupby(["stimulus_id","stimulus_name"], as_index=False)
        out = g.agg(**{new:(pd.NamedAgg(old,"sum")) for old,new in sum_map.items()})
        return out
    if level == "lang":
        sums = {new: df[old].sum() for old,new in sum_map.items()}
        return pd.DataFrame([{"level":"lang", **sums}])
    return df

def sentence_counts_by_predicate(stimuli, nlp, predicate, value_name, include_word_count=False):
    rows = []
    for sid, sname, page, sent_idx, sent in iter_sentences(stimuli, nlp):
        cnt = 0
        words = 0
        for tok in sent:
            if tok.is_alpha:
                words += 1
            if predicate(tok):
                cnt += 1
        row = {"stimulus_id":sid,"stimulus_name":sname,"page":page,"sent_idx":sent_idx, value_name:cnt}
        if include_word_count:
            row["words"] = words
        rows.append(row)
    return pd.DataFrame(rows)

def pronouns(stimuli, nlp, level="page"):
    df = sentence_counts_by_predicate(
        stimuli, nlp,
        predicate=lambda t: (t.is_alpha and t.pos_=="PRON"),
        value_name="pronouns",
        include_word_count=True
    )
    return agg_from_sentence_rows(df, level, {"pronouns":"total_pronouns","words":"total_words"})

def punctuation(stimuli, nlp, level="page"):
    df = sentence_counts_by_predicate(
        stimuli, nlp,
        predicate=lambda t: t.is_punct,
        value_name="punctuation",
        include_word_count=False
    )
    return agg_from_sentence_rows(df, level, {"punctuation":"total_punct"})

def fertility(stimuli, nlp, tokenizer, level="page"):
    rows = []
    for sid, sname, page, doc in iter_pages(stimuli, nlp):
        word_count = 0
        llm_tokens = 0
        for tok in doc:
            if tok.is_alpha:
                word_count += 1
                llm_tokens += len(tokenizer.encode(tok.text, add_special_tokens=False))
        rows.append({
            "stimulus_id":sid,"stimulus_name":sname,"page":page,
            "total_words":word_count,"total_llm_tokens":llm_tokens
        })
    df = pd.DataFrame(rows)
    return agg_from_page_rows(df, level, {"total_words":"total_words","total_llm_tokens":"total_llm_tokens"})

def ttr(stimuli, nlp, level="page"):
    rows = []
    for sid, sname, page, doc in iter_pages(stimuli, nlp):
        ptext = doc.text
        d = nlp(ptext)
        tokens = [t.text for t in d if t.is_alpha]
        n_tok = len(tokens)
        n_types = len(set(tokens))
        rows.append({"stimulus_id":sid,"stimulus_name":sname,"page":page,
                     "num_tokens":n_tok,"num_types":n_types})
    df = pd.DataFrame(rows)
    if level == "page":
        df["ttr"] = df["num_types"] / df["num_tokens"].replace(0, 1)
        return df
    out = agg_from_page_rows(df, level, {"num_tokens":"num_tokens","num_types":"num_types"})
    if level in {"doc","lang"}:
        out["ttr"] = out["num_types"] / out["num_tokens"].replace(0, 1)
    return out
