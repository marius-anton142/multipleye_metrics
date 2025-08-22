# -*- coding: utf-8 -*-
import re
import spacy
import pandas as pd

def pronouns(stimuli, nlp, level="page"):
    rows = []

    if level == "sentence":
        for stim in stimuli:
            sid = stim["stimulus_id"]
            sname = stim["stimulus_name"]

            text = " ".join(stim["pages"])
            doc = nlp(text)

            sent_idx = 0
            for sent in doc.sents:
                pron_cnt = 0
                words = 0
                for tok in sent:
                    if tok.is_alpha:
                        words += 1
                        if tok.pos_ == "PRON":
                            pron_cnt += 1
                rows.append({
                    "stimulus_id": sid,
                    "stimulus_name": sname,
                    "page": None,
                    "sent_idx": sent_idx,
                    "pronouns": pron_cnt,
                    "words": words
                })
                sent_idx += 1
        return pd.DataFrame(rows)

    for stim in stimuli:
        sid = stim["stimulus_id"]
        sname = stim["stimulus_name"]

        for pnum, page_text in enumerate(stim["pages"], start=1):
            doc = nlp(page_text)
            sent_idx = 0

            for sent in doc.sents:
                pron_cnt = 0
                word_cnt = 0
                for tok in sent:
                    if tok.is_alpha:
                        word_cnt += 1
                        if tok.pos_ == "PRON":
                            pron_cnt += 1

                rows.append({
                    "stimulus_id": sid,
                    "stimulus_name": sname,
                    "page": pnum,
                    "sent_idx": sent_idx,
                    "pronouns": pron_cnt,
                    "words": word_cnt
                })
                sent_idx += 1

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    if level == "page":
        out = (
            df.groupby(["stimulus_id","stimulus_name","page"], as_index=False)
              .agg(
                  total_pronouns=("pronouns","sum"),
                  n_sentences=("sent_idx","count"),
                  total_words=("words","sum")
              )
        )
        return out

    if level == "doc":
        out = (
            df.groupby(["stimulus_id","stimulus_name"], as_index=False)
              .agg(
                  total_pronouns=("pronouns","sum"),
                  n_sentences=("sent_idx","count"),
                  total_words=("words","sum")
              )
        )
        return out

    if level == "lang":
        total_pronouns = df["pronouns"].sum()
        n_sentences = df["sent_idx"].count()
        total_words = df["words"].sum()

        out = pd.DataFrame([{
            "level": "lang",
            "total_pronouns": total_pronouns,
            "n_sentences": n_sentences,
            "total_words": total_words
        }])
        return out

    return df

def punctuation(stimuli, nlp, level="page"):
    rows = []

    for stim in stimuli:
        sid = stim["stimulus_id"]
        sname = stim["stimulus_name"]

        for pnum, page_text in enumerate(stim["pages"], start=1):
            doc = nlp(page_text)

            sent_idx = 0
            for sent in doc.sents:
                punct_cnt = 0
                for tok in sent:
                    if tok.is_punct:
                        punct_cnt += 1

                rows.append({
                    "stimulus_id": sid,
                    "stimulus_name": sname,
                    "page": pnum,
                    "sent_idx": sent_idx,
                    "punctuation": punct_cnt
                })
                sent_idx += 1

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    if level == "page":
        out = (
            df.groupby(["stimulus_id","stimulus_name","page"], as_index=False)
              .agg(
                  total_punct=("punctuation","sum"),
                  n_sentences=("sent_idx","count")
              )
        )
        return out

    if level == "doc":
        out = (
            df.groupby(["stimulus_id","stimulus_name"], as_index=False)
              .agg(
                  total_punct=("punctuation","sum"),
                  n_sentences=("sent_idx","count")
              )
        )
        return out

    if level == "lang":
        total_punct = df["punctuation"].sum()
        n_sentences = df["sent_idx"].count()

        out = pd.DataFrame([{
            "level": "lang",
            "total_punct": total_punct,
            "n_sentences": n_sentences
        }])
        return out

    return df

def fertility(stimuli, nlp, tokenizer, level="page"):
    rows = []

    for stim in stimuli:
        sid = stim["stimulus_id"]
        sname = stim["stimulus_name"]

        for pnum, page_text in enumerate(stim["pages"], start=1):
            doc = nlp(page_text)

            word_count = 0
            llm_token_count = 0

            for tok in doc:
                if tok.is_alpha:
                    word_count += 1
                    llm_token_count += len(tokenizer.encode(tok.text, add_special_tokens=False))

            rows.append({
                "stimulus_id": sid,
                "stimulus_name": sname,
                "page": pnum,
                "total_words": word_count,
                "total_llm_tokens": llm_token_count
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    if level == "page":
        return df

    if level == "doc":
        out = (
            df.groupby(["stimulus_id","stimulus_name"], as_index=False)
              .agg(
                  total_words=("total_words","sum"),
                  total_llm_tokens=("total_llm_tokens","sum")
              )
        )
        return out

    if level == "lang":
        total_words = df["total_words"].sum()
        total_llm_tokens = df["total_llm_tokens"].sum()
        out = pd.DataFrame([{
            "level": "lang",
            "total_words": total_words,
            "total_llm_tokens": total_llm_tokens
        }])
        return out

    return df

def ttr(stimuli, nlp, level="page"):
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def ttr_from_text(text):
        doc = nlp(text)
        tokens = [t.text for t in doc if t.is_alpha]
        if not tokens:
            return 0, 0
        return len(tokens), len(set(tokens))

    rows = []

    for stim in stimuli:
        sid = stim["stimulus_id"]
        sname = stim["stimulus_name"]

        for pnum, page_text in enumerate(stim["pages"], start=1):
            ptext = preprocess(page_text)
            n_tok, n_types = ttr_from_text(ptext)
            rows.append({
                "stimulus_id": sid,
                "stimulus_name": sname,
                "page": pnum,
                "num_tokens": n_tok,
                "num_types": n_types
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    if level == "page":
        df["ttr"] = df["num_types"] / df["num_tokens"].replace(0, 1)
        return df

    if level == "doc":
        out = (
            df.groupby(["stimulus_id","stimulus_name"], as_index=False)
              .agg(
                  num_tokens=("num_tokens","sum"),
                  num_types=("num_types","sum")
              )
        )
        out["ttr"] = out["num_types"] / out["num_tokens"].replace(0, 1)
        return out

    if level == "lang":
        total_tokens = df["num_tokens"].sum()
        total_types = df["num_types"].sum()
        out = pd.DataFrame([{
            "level": "lang",
            "num_tokens": total_tokens,
            "num_types": total_types,
            "ttr": total_types / total_tokens if total_tokens else 0
        }])
        return out
