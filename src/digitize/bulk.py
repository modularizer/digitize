#!/usr/bin/env python3
import re
import sys
from fractions import Fraction
from typing import Tuple

from digitize.attrstr import attrstr
from digitize.config import DigitizeMode, DigitizeParams, default
from digitize import config as modes
from digitize.patterns import chunk_string, merge_chunks, any_word
from digitize.sentinel import _sentinel
from digitize.stages import StageResult, StageName, Stage
from digitize.tasks.fractions import frac_to_exact_str
from digitize.tasks.replace_multipliers import replace_multipliers
from digitize.tasks.simple_eval import simple_eval
from digitize.tasks.unit_cascade import get_level_merger
from digitize.units import Unit, UnitGroup, unit_modes, AllUnits, flatten_units
from digitize.tasks.roman import ROMAN_PATTERN, roman_to_int
from digitize.tasks.ordinal import ORD_N019, ORD_ORDERS, ORD_TENS, ordinal_suffix
from digitize.consts import N019, FRACTION_DEN_WORD, TENS_WORDS, DIGIT_WORD_TO_DIGIT, REPEAT_WORDS, MAGNITUDE_VALUE, \
    POWERS

_SEC__sentinel = "__DIGITIZE_ECOND_UNIT__"
_REPEAT_PREFIX = "__repeat__"



def digitize(
    s: str,
    *,
    config: DigitizeMode  | DigitizeParams = default,
    _iter: bool = True,
    call_level: int = 0,
        raw: bool = False,
    **kwargs # Accepts DigitizeParams args
) -> str | attrstr:

    initial=s
    skip = Stage.FULL in kwargs.get("skip_stages", config.skip_stages if isinstance(config, DigitizeParams) else ())
    result = StageResult(
        prev=s,
        new=None,
        stage=Stage.FULL,
        skipped=skip,
        changed=False,
        ctx={"config": config, "kwargs": kwargs},
    )

    if skip or (not s.strip()):
        s = result.finish(initial)
        return str(s) if raw else s

    # first, merge params with the default config profile selected
    # basically there are two toggles.
    # config selects a full set of params, and then **kwargs perform overrides
    params = DigitizeParams(**kwargs)
    defaults = config if isinstance(config, DigitizeParams) else getattr(modes, config)
    config = DigitizeParams(**{**defaults.non_sentinels(), **params.non_sentinels()}).finalize()

    result.ctx = config

    def stage_enabled(stage: StageName) -> StageResult | None:
        stage = StageResult(
          prev=result.stages[-1] if result.stages else None,
          changed=False,
          new=None,
          stage=stage,
          ctx=None,
          skipped=False,
          call_level=call_level,
          log=(config.log_stages is True or (config.log_stages and stage in config.log_stages)),
          log_context =(config.log_context is True or (config.log_context and stage in config.log_context)),
        )
        result.stages.append(stage)
        if config.skip_stages and stage in config.skip_stages:
            stage.skip()
            return None
        return stage



    recurse_kwargs = {
        "config": config,
        "call_level": call_level + 1,
        "_iter": False,
        "raw": False,
    }
    recurse = lambda _s, **kw: digitize(_s, **recurse_kwargs, **kw)

    # breaks allows setting up break words to indepenently analyze different sections then stitch back together
    # e.g. use ". " to separately analyze each sentence
    chunks, seps = chunk_string(s, config.breaks)
    if len(chunks) > 1:
        if update := stage_enabled(Stage.SPLIT_BY_BREAKS):
            update(chunks, f"breaks={config.breaks}, seps={seps}")
        s =  merge_chunks([recurse(c) for c in chunks], seps=seps)
        s = result.finish(s)
        return str(s) if raw else s


    # tokenize repeat words like times, etc.
    _repeat_map: dict[str, str] = {}  # store mapping for reversing near the end
    if config.support_reps and config.rep_signifiers:
        if update := stage_enabled(Stage.REPEAT_SENTINEL_PREPROCESSED):
            # Make a single alternation regex. Each signifier is treated as a full match.
            # Use a capturing group so we can store the exact matched text.
            rep_rx = re.compile(any_word(config.rep_signifiers), flags=re.IGNORECASE)

            def _rep_repl(m: re.Match) -> str:
                key = f"{_REPEAT_PREFIX}{len(_repeat_map)}_"
                _repeat_map[key] = m.group(0)  # store original, with original casing/spaces
                return key

            s2 = rep_rx.sub(_rep_repl, s)
            s = update(s2, rep_rx)

    def _is_repeat_tail(w: str) -> bool:
        return re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", w, flags=re.IGNORECASE) is not None

    # attempt to differentiate "second" in the time meaning vs "second" in the numerical meaning
    if config.attempt_to_differentiate_seconds:
        if update := stage_enabled(Stage.ATTEMPT_TO_DIFFERENTIATE_SECONDS):
            s2 = re.sub(r"\b(a|one|per|each)\s+second\b",rf"\1 {_SEC__sentinel}", s, flags=re.IGNORECASE)
            s3 = re.sub(r"\b(the)\s+second\s+(after|before|between|when)\b",rf"\1 {_SEC__sentinel} \2", s2,flags=re.IGNORECASE,)
            s = update(s3)

    # --- expand capital suffixes BEFORE lowercasing (capital-only by regex) ---
    if config.replace_multipliers:
        if update := stage_enabled(Stage.REPLACE_MULTIPLIERS):
            s2 = replace_multipliers(s, fmt_multipliers=config.fmt_multipliers, use_commas=config.use_commas)
            s = update(s2)


    word_to_num = {w: i for i, w in enumerate(N019)}
    word_to_num.update({w: (i + 2) * 10 for i, w in enumerate(TENS_WORDS)})
    ordinal_word_to_num = {}
    ordinal_magnitude_exact = {}
    if config.support_ordinals:
        ordinal_word_to_num.update(ORD_N019)
        ordinal_word_to_num.update(ORD_TENS)
        ordinal_magnitude_exact = ORD_ORDERS.copy()
    repeat_word_to_num = REPEAT_WORDS.copy()


    def is_numeric_atom(tok: str) -> bool:
        _t = tok.lower()
        return (
            _t in {"point", "oh", "o"} or # decimals
            _t in FRACTION_DEN_WORD  or # keep fraction denominators inside the numeric phrase
            (config.support_reps and _t in repeat_word_to_num) or
            (config.support_ordinals and re.fullmatch(r"\d+(st|nd|rd|th)", _t)) or
            (_t.isdigit() or _t in word_to_num or _t == "hundred" or _t in MAGNITUDE_VALUE) or
            (config.support_ordinals and (_t in ordinal_word_to_num or _t in ordinal_magnitude_exact)) or
            (config.support_reps and re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", tok, flags=re.IGNORECASE)) or
            (config.support_roman and ROMAN_PATTERN.fullmatch(tok))
        )


    def allows_and_after(prev_norm: str | None) -> bool:
        # allow grammatically incorrect 'hundred and'
        if prev_norm is None:
            return False
        prev_norm = prev_norm.lower()
        return prev_norm == "hundred" or prev_norm in MAGNITUDE_VALUE

    # returns (value, is_ordinal, is_time_phrase)
    def parse_number(norm_words: list[str]) -> Tuple[object, bool, bool] | None:
        if norm_words and norm_words[-1] == "and":
            return None

        # handle once/twice/thrice as standalone (or with trailing time/times)
        if config.support_reps and norm_words:
            w0 = norm_words[0]
            if w0 in repeat_word_to_num:
                if len(norm_words) == 1:
                    return (repeat_word_to_num[w0], False, True)
                if _is_repeat_tail(norm_words[-1]):
                    return (repeat_word_to_num[w0], False, True)

        is_time_phrase = False
        if config.support_reps and norm_words and _is_repeat_tail(norm_words[-1]):
            is_time_phrase = True
            core = norm_words[:-1]
        else:
            core = norm_words

        if not core:
            return None

        # -------------------------
        # FRACTIONS: "<numerator> <denominator>"
        #   "one third" -> 1/3
        #   "two hundredths" -> 2/100
        #   "five point 8 millionth" -> 5.8/1000000
        #   "2 halves" -> 2/2
        #
        # Safety rules:
        #   - denominator must be LAST token in phrase
        #   - fraction only if denom is plural OR numerator is (a|one|1) OR numerator is numeric (e.g. 5.8)
        # -------------------------
        denom_tok = core[-1]

        denom_val: int | None = None
        denom_is_plural = False

        if not is_time_phrase:
            # word denominators
            if denom_tok in FRACTION_DEN_WORD:
                denom_val = FRACTION_DEN_WORD[denom_tok]
                denom_is_plural = denom_tok.endswith("s") or denom_tok.endswith("ves")

            # numeric ordinals like "100th" / "100ths"
            if denom_val is None:
                mo = re.fullmatch(r"(\d+)(st|nd|rd|th)(s)?", denom_tok)
                if mo and mo.group(2) == "th":
                    denom_val = int(mo.group(1))
                    denom_is_plural = mo.group(3) is not None

            if denom_val is not None:
                numer_words = core[:-1]

                # If there's no numerator, DO NOT abort parsing — let ordinal/time logic handle it.
                # Except: allow bare "half" -> 1/2 (per your tests).
                if not numer_words:
                    if denom_tok in {"half", "halves"}:
                        return (("FRAC", "1", denom_val), False, False)
                    # fall through to normal parsing (so "third time" still becomes 3rd time)
                    denom_val = None

                if denom_val is not None:

                    # numerator: allow "a" => 1
                    if len(numer_words) == 1 and numer_words[0] == "a":
                        numer_str = "1"
                        numer_is_oneish = True
                        numer_is_numeric = True
                    else:
                        # reuse your decimal parsing style for numerator-only
                        def _parse_numeric_string(words: list[str]) -> str | None:
                            # decimal numerator: "<int> point <digits...>"
                            if "point" in words:
                                p2 = words.index("point")
                                left2 = words[:p2]
                                right2 = words[p2 + 1:]
                                if not right2:
                                    return None
                                digs: list[str] = []
                                for w in right2:
                                    if w == "and":
                                        return None
                                    if w.isdigit():
                                        digs.append(w)
                                    elif w in DIGIT_WORD_TO_DIGIT:
                                        digs.append(DIGIT_WORD_TO_DIGIT[w])
                                    else:
                                        return None
                                frac = "".join(digs)
                                if frac == "":
                                    return None

                                # parse left2 as integer (allow empty => 0)
                                if not left2:
                                    ip = 0
                                else:
                                    total2 = 0
                                    current2 = 0
                                    saw2 = False
                                    for w in left2:
                                        if w == "and":
                                            continue
                                        if w in {"oh", "o"}:
                                            current2 += 0
                                            saw2 = True
                                            continue
                                        if w.isdigit():
                                            current2 += int(w); saw2 = True; continue
                                        if w in word_to_num:
                                            current2 += word_to_num[w]; saw2 = True; continue
                                        if w == "hundred":
                                            if not saw2: return None
                                            current2 *= 100; continue
                                        if w in MAGNITUDE_VALUE:
                                            if not saw2: return None
                                            total2 += current2 * MAGNITUDE_VALUE[w]
                                            current2 = 0
                                            continue
                                        return None
                                    if not saw2:
                                        return None
                                    ip = total2 + current2

                                return f"{ip}.{frac}"

                            # integer numerator
                            # (use your existing integer loop behavior, but return string)
                            total2 = 0
                            current2 = 0
                            saw2 = False
                            for w in words:
                                if w == "and":
                                    continue
                                if w in {"oh", "o"}:
                                    current2 += 0
                                    saw2 = True
                                    continue
                                if w.isdigit():
                                    current2 += int(w); saw2 = True; continue
                                if w in word_to_num:
                                    current2 += word_to_num[w]; saw2 = True; continue
                                if w == "hundred":
                                    if not saw2: return None
                                    current2 *= 100; continue
                                if w in MAGNITUDE_VALUE:
                                    if not saw2: return None
                                    total2 += current2 * MAGNITUDE_VALUE[w]
                                    current2 = 0
                                    continue
                                return None
                            if not saw2:
                                return None
                            return str(total2 + current2)

                        numer_str = _parse_numeric_string(numer_words)
                        if numer_str is None:
                            return None

                        numer_is_oneish = numer_str in {"1", "1.0"}
                        numer_is_numeric = True  # if we parsed it, it’s numeric

                    # gating rule so we don't break ordinals in normal prose
                    if not (denom_is_plural or numer_is_oneish or numer_is_numeric):
                        return None

                    return (("FRAC", numer_str, denom_val), False, False)


        # -------------------------
        # DECIMALS: "<int> point <digits...>"
        #   "zero point five" -> 0.5
        #   "point five" -> 0.5
        #   "two point zero five" -> 2.05
        # -------------------------
        if "point" in core:
            # only support the first "point" for now
            p = core.index("point")
            left = core[:p]
            right = core[p + 1 :]

            # require at least one digit after point
            if not right:
                return None

            # right side must be ONLY digit-words (0-9) or digit tokens
            digits: list[str] = []
            for w in right:
                if w == "and":
                    return None
                if w.isdigit():
                    digits.append(w)  # preserve "05" if it appears
                    continue
                if w in DIGIT_WORD_TO_DIGIT:
                    digits.append(DIGIT_WORD_TO_DIGIT[w])
                    continue
                return None

            frac_digits = "".join(digits)
            if frac_digits == "":
                return None

            # parse left as an integer using the existing logic (but reject ordinals)
            # allow empty left => 0
            if not left:
                int_part = 0
            else:
                total = 0
                current = 0
                saw_any = False
                is_ord_left = False

                for w in left:
                    if w == "and":
                        continue

                    # allow "oh"/"o" as 0 on the LEFT side of "point"
                    if w in {"oh", "o"}:
                        current += 0
                        saw_any = True
                        continue

                    # reject ordinals for decimals in stage 1
                    if config.support_ordinals:
                        if re.fullmatch(r"\d+(st|nd|rd|th)", w):
                            return None
                        if w in ordinal_word_to_num or w in ordinal_magnitude_exact:
                            return None

                    if w.isdigit():
                        current += int(w)
                        saw_any = True
                        continue

                    if w in word_to_num:
                        current += word_to_num[w]
                        saw_any = True
                        continue

                    if w == "hundred":
                        if not saw_any:
                            return None
                        current *= 100
                        continue

                    if w in MAGNITUDE_VALUE:
                        if not saw_any:
                            return None
                        total += current * MAGNITUDE_VALUE[w]
                        current = 0
                        continue

                    return None

                if not saw_any:
                    return None

                int_part = total + current

            return (("DEC", int_part, frac_digits), False, False)

        total = 0
        current = 0
        saw_any = False
        is_ord = False

        for w in core:
            if w == "and":
                continue

            if config.support_ordinals:
                mo = re.fullmatch(r"(\d+)(st|nd|rd|th)", w)
                if mo:
                    current += int(mo.group(1))
                    saw_any = True
                    is_ord = True
                    continue

            if w.isdigit():
                current += int(w)
                saw_any = True
                continue

            if w in word_to_num:
                current += word_to_num[w]
                saw_any = True
                continue

            if config.support_ordinals and w in ordinal_word_to_num:
                current += ordinal_word_to_num[w]
                saw_any = True
                is_ord = True
                continue

            if w == "hundred":
                if not saw_any:
                    return None
                current *= 100
                continue

            if config.support_ordinals and w in ordinal_magnitude_exact:
                if not saw_any:
                    # bare "hundredth" => 1 * 100
                    current = ordinal_magnitude_exact[w]
                    saw_any = True
                else:
                    current *= ordinal_magnitude_exact[w]
                is_ord = True
                continue

            if w in MAGNITUDE_VALUE:
                if not saw_any:
                    return None
                total += current * MAGNITUDE_VALUE[w]
                current = 0
                continue
                
            if config.support_roman and ROMAN_PATTERN.fullmatch(w):
                current += roman_to_int(w)
                saw_any = True
                continue

            return None

        return (total + current, is_ord, is_time_phrase) if saw_any else None

    # tokenization that preserves whitespace and punctuation, and keeps numeric ordinals like "1st"
    tokens = re.findall(
        rf"{_REPEAT_PREFIX}\d+_|\d+(?:st|nd|rd|th)|[A-Za-z]+|\d+|\s+|[^A-Za-z\d\s]+",
        s,
        flags=re.IGNORECASE,
    )


    out: list[str] = []
    raw: list[str] = []
    norm: list[str] = []
    pending_ws: str = ""

    def _flush_phrase():
        nonlocal raw, norm, pending_ws
        if not norm:
            return

        # Decide whether this phrase is worth attempting to parse
        has_convertible = False
        for w in norm:
            if w == "and":
                continue

            if w in FRACTION_DEN_WORD:
                has_convertible = True
                break

            if w.isdigit():
                continue
            if w in word_to_num or w == "hundred" or w in MAGNITUDE_VALUE:
                has_convertible = True
                break
            if config.support_ordinals and (w in ordinal_word_to_num or w in ordinal_magnitude_exact or re.fullmatch(r"\d+(st|nd|rd|th)", w)):
                has_convertible = True
                break
            if config.support_reps and re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", w, flags=re.IGNORECASE):
                has_convertible = True
                break
            if config.support_reps and w in repeat_word_to_num:
                has_convertible = True
                break
            if config.support_roman and ROMAN_PATTERN.fullmatch(w):
                has_convertible = True
                break


        if not has_convertible:
            out.append("".join(raw))
            raw = []
            norm = []
            if pending_ws:
                out.append(pending_ws)
                pending_ws = ""
            return

        parsed = parse_number(norm)
        if parsed is None:
            out.append("".join(raw))
        else:
            n, is_ord, is_time = parsed


            # fractions
            if isinstance(n, tuple) and len(n) == 3 and n[0] == "FRAC":
                numer_str, denom_val = n[1], n[2]
                num = f"{numer_str}/{denom_val}"
                out.append(
                    config.fmt
                    .replace("%n", num)
                    .replace("%s", "s")
                    .replace("%r", "x")
                    .replace("%i", "".join(raw))
                )


            # decimals
            elif isinstance(n, tuple) and len(n) == 3 and n[0] == "DEC":
                int_part, frac_digits = n[1], n[2]
                int_part_str = f"{int_part:,}" if config.use_commas else str(int_part)
                num = f"{int_part_str}.{frac_digits}"
                # decimals are never ordinals/reps in stage 1
                out.append(
                    config.fmt
                    .replace("%n", num)
                    .replace("%s", "s")  # irrelevant but keep pipeline stable
                    .replace("%r", "x")
                    .replace("%i", "".join(raw))
                )
            else:
                # existing integer behavior
                num = f"{n:,}" if config.use_commas else str(n)
                plural_s = "s" if abs(n) != 1 else ""
                if _repeat_map:
                    x = "".join(raw)
                    pat = re.compile(rf"^(.*)?({_REPEAT_PREFIX}\d+_)(.*)$")
                    m = re.match(pat, x)
                    r = _repeat_map[m.group(2)] if m else "x"
                    i = re.sub(pat, rf"\1{r}", x)
                else:
                    r = "x"
                    i = "".join(raw)


                if is_time and config.support_reps:
                    if is_ord and config.support_ordinals:
                        suf = ordinal_suffix(n)
                        out.append(
                            config.fmt_nth_time
                            .replace("%n", num)
                            .replace("%o", suf)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )
                    else:
                        out.append(
                            config.fmt_rep
                            .replace("%n", num)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )
                else:
                    if is_ord and config.support_ordinals:
                        suf = ordinal_suffix(n)
                        out.append(
                            config.fmt_ordinal
                            .replace("%n", num)
                            .replace("%o", suf)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )
                    else:
                        out.append(
                            config.fmt
                            .replace("%n", num)
                            .replace("%s", plural_s)
                            .replace("%r",  r)
                            .replace("%i", i)
                        )



        raw = []
        norm = []
        if pending_ws:
            out.append(pending_ws)
            pending_ws = ""

    def _next_nonspace(j: int) -> str | None:
        k = j + 1
        while k < len(tokens) and tokens[k].isspace():
            k += 1
        return tokens[k] if k < len(tokens) else None

    def _peek_decimal_start(i: int) -> bool:
        """
        True if tokens after index i look like: <ws>* 'point' <ws>* <digit-or-digitword>
        """
        nxt = _next_nonspace(i)
        if nxt is None or nxt.lower() != "point":
            return False
        # find token after 'point'
        j = i + 1
        while j < len(tokens) and tokens[j].isspace():
            j += 1
        if j >= len(tokens) or tokens[j].lower() != "point":
            return False
        k = j + 1
        while k < len(tokens) and tokens[k].isspace():
            k += 1
        if k >= len(tokens):
            return False
        t = tokens[k].lower()
        return t.isdigit() or t in DIGIT_WORD_TO_DIGIT


    def _commit_pending_ws_into_phrase():
        nonlocal pending_ws
        if pending_ws:
            raw.append(pending_ws)
            pending_ws = ""

    def _hyphen_is_internal(prev_norm: str | None, next_tok: str | None) -> bool:
        if prev_norm is None or next_tok is None:
            return False
        p = prev_norm.lower()
        nxt = next_tok.lower()

        if p in TENS_WORDS:
            if nxt.isdigit() or nxt in word_to_num:
                return True
            if config.support_ordinals and nxt in ordinal_word_to_num:
                return True
        return False

    for i, t in enumerate(tokens):
        if t.isspace():
            if norm:
                pending_ws += t
            else:
                out.append(t)
            continue

        if re.fullmatch(rf"{_REPEAT_PREFIX}\d+_|[A-Za-z]+|\d+|\d+(?:st|nd|rd|th)", t, flags=re.IGNORECASE):
            tl = t.lower()

            # allow "a" as numerator ONLY when it starts a fraction phrase: "a third", "a half", etc.
            if tl == "a":
                nxt = _next_nonspace(i)
                if nxt is not None and nxt.lower() in FRACTION_DEN_WORD:
                    if norm:
                        _commit_pending_ws_into_phrase()
                    raw.append(t)        # keep "a"
                    norm.append("a")     # normalize
                    continue

            # special handling for "time/times":
            # only include it as part of phrase if the phrase so far parses as a number (cardinal/ordinal/repeat word)
            # special handling for "time/times" sentinels
            if config.support_reps and re.fullmatch(rf"{_REPEAT_PREFIX}\d+_", t, flags=re.IGNORECASE):
                nxt = _next_nonspace(i)

                # If another numeric atom follows, this is NOT repetition; it's "A times B"
                # So: flush the left number, emit the original word ("times"), and continue.
                if nxt is not None and is_numeric_atom(nxt):
                    _flush_phrase()
                    out.append(_repeat_map.get(t, t))  # emit original signifier (e.g., "times")
                    continue

                # otherwise keep existing repetition behavior
                if norm:
                    if parse_number(norm) is not None:
                        _commit_pending_ws_into_phrase()
                        raw.append(t)
                        norm.append(t.lower())
                    else:
                        _flush_phrase()
                        out.append(_repeat_map.get(t, t))
                else:
                    out.append(_repeat_map.get(t, t))
                continue


            if tl == "and":
                if norm:
                    nxt = _next_nonspace(i)
                    prev_norm = next((w for w in reversed(norm) if w != "and"), None)
                    if allows_and_after(prev_norm) and (nxt is not None and is_numeric_atom(nxt)):
                        _commit_pending_ws_into_phrase()
                        raw.append(t)
                        norm.append("and")
                    else:
                        _flush_phrase()
                        out.append(t)
                else:
                    out.append(t)
                continue

            # special-case: "oh point ..." / "o point ..." should behave like "zero point ..."
            if tl in {"oh", "o"} and _peek_decimal_start(i):
                if norm:
                    _commit_pending_ws_into_phrase()
                raw.append(t)          # keep original text (oh/o)
                norm.append("zero")    # treat as numeric 0 for parsing
                continue


            if is_numeric_atom(t):
                if norm:
                    _commit_pending_ws_into_phrase()
                raw.append(t)
                norm.append(tl)
            else:
                _flush_phrase()
                out.append(t)
            continue

        if t == "-":
            if norm:
                nxt = _next_nonspace(i)
                prev_norm = next((w for w in reversed(norm) if w != "and"), None)
                if _hyphen_is_internal(prev_norm, nxt):
                    _commit_pending_ws_into_phrase()
                    raw.append("-")
                    continue
            _flush_phrase()
            out.append("-")
            continue

        _flush_phrase()
        out.append(t)

    _flush_phrase()
    if pending_ws:
        out.append(pending_ws)

    if update := stage_enabled(Stage.MAIN_LOOP):
        s2 = "".join(out).replace("%p", "").replace("%p", "")
        s = update(s2)

    if config.attempt_to_differentiate_seconds:
        if update := stage_enabled(Stage.RESUB_SECONDS):
            s2 = re.sub(_SEC__sentinel, "second", s)
            s = update(s2)

    all_units = AllUnits(flatten_units(config.units))
    # print(f"{all_units=}")
    # print(f"ug: {[ug for ug in all_units if not isinstance(ug, Unit)]}")

    for u in all_units:
        if update := stage_enabled(Stage.UNIT_SUB):
            s2 = u.sub(s, all_units)
            s = update(s2, u.key)
    if update := stage_enabled(Stage.A_SET_OF):
        s2 = re.sub(r" a set\(s\) of ", " ", s)
        s = update(s2)

    if _repeat_map:
        if update := stage_enabled(Stage.RESUB_REPEAT_MAP):
            s2 = re.sub(
                rf"{_REPEAT_PREFIX}\d+_",
                lambda m: _repeat_map.get(m.group(0), m.group(0)),
                s,
            )
            s = update(s2)

    if config.parse_signs:
        num_rx = r"(\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)"
        if update := stage_enabled(Stage.NEGATIVE):
            s2 = re.sub(rf"\b(neg|negative|minus)\s+{num_rx}\b", r"-\2", s, flags=re.IGNORECASE)
            s = update(s2)
        if update := stage_enabled(Stage.POSITIVE):
            s2 = re.sub(rf"\b(pos|positive|plus)\s+{num_rx}\b", r"+\2", s, flags=re.IGNORECASE)
            s = update(s2)
        if update := stage_enabled(Stage.PLUS_MINUS_SPACE):
            s2 = re.sub(rf"\+\s+{num_rx}\b", r"+\1", s, flags=re.IGNORECASE)
            s3 = re.sub(rf"-\s+{num_rx}\b", r"-\1", s2, flags=re.IGNORECASE)
            s = update(s3)

    # x over y  /  x divided by y  ->  x/y
    if update := stage_enabled(Stage.DIVIDED_BY):
        s2 = re.sub(
            r"\b([+-]?\d+(?:\.\d+)?)\s+(?:over|divided\s+by)\s+(\d+(?:\.\d+)?)\b",
            rf"\1{config.div}\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)
    # x over y  /  x divided by y  ->  x/y
    if update := stage_enabled(Stage.DIVIDED_BY):
        s2 = re.sub(
            r"\b([+-]?\d+(?:\.\d+)?)\s+(?:over|divided\s+by|out of|into|of)\s+(\d+(?:\.\d+)?)\b",
            rf"\1{config.div}\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # multiplication: "<num> times <num>" or "<num> multiplied by <num>" -> "<num>*<num>"
    num_atom = r"(?:[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)"

    if update := stage_enabled(Stage.PLUS_MINUS_SPACE):
        s2 = re.sub(
            rf"({num_atom})\s\+({num_atom})",
            r"\1+\2",
            s,
            flags=re.IGNORECASE,
        )
        s3 = re.sub(
            rf"({num_atom})\s-({num_atom})",
            r"\1-\2",
            s2,
            flags=re.IGNORECASE,
        )
        s = update(s3)

    def _frac_exponent_to_ordinal(m: re.Match) -> str:
        base = m.group(1)
        numer = m.group(2)
        denom = int(m.group(3))

        # only if numerator is a plain integer
        if not numer.isdigit():
            return m.group(0)

        # only for "hundredth/thousandth/millionth..." style denominators
        # (i.e. powers of 10 >= 100)
        if denom < 100:
            return m.group(0)
        t = denom
        while t % 10 == 0:
            t //= 10
        if t != 1:
            return m.group(0)

        exp = int(numer) * denom
        return f"{base} to the {exp} power"


    if update := stage_enabled(Stage.POWER_A):
        s2 = re.sub(
            r"\b([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s+to\s+the\s+(\d+)/(\d+)\s+power\b",
            _frac_exponent_to_ordinal,
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)
    if update := stage_enabled(Stage.POWER_B):
        s2 = re.sub(
            rf"\b({num_atom})\s(?:raised )?(?:to the power of|to the)\s({num_atom})(?:rd|st|th)?(?: (?:power|exponent|degree))?\b",
            rf"\1{config.power}\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)
    if update := stage_enabled(Stage.POWER_C):
        s2 = re.sub(
            rf"\b({num_atom})\s(?:raised )?(?:to the power of|to the)\s({num_atom})(?:rd|st|th)\b",
            rf"\1{config.power}\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    for k, v in POWERS.items():
        if update := stage_enabled(Stage.POWER_D):
            s2 = re.sub(
                rf"\b({num_atom})\s(?:raised )?(?:to the power of|to the)?\s({k})(?: power)?\b",
                rf"\1{config.power}{v}",
                s,
                flags=re.IGNORECASE,
            )
            s = update(s2, k)

    # helpers
    ws = r"\s+"
    num_atom = r"[+-]?\d+(?:\.\d+)?(?:/\d+)?"

    # square root of x  ->  sqrt(x)
    if update := stage_enabled(Stage.SQUARE_ROOT):
        s2 = re.sub(
            rf"\b(square)\s+root(?:\s+of)?{ws}({num_atom})\b",
            rf"\2{config.power}(1/2)",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # cube root of x  ->  cbrt(x)
    if update := stage_enabled(Stage.CUBED_ROOT):
        s2 = re.sub(
            rf"\b(cube)\s+root(?:\s+of)?{ws}({num_atom})\b",
            rf"\2{config.power}(1/3)",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # nth root of x  ->  root(n,x)
    # "the 5th root of 32", "5th root of 32", "5 root of 32" (if you want)
    if update := stage_enabled(Stage.NTH_ROOT):
        s2 = re.sub(
            rf"\b(?:the\s+)?({num_atom})(?:st|nd|rd|th)?\s+root(?:\s+of)?{ws}({num_atom})\b",
            rf"\2{config.power}(1/\1)",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)


    # --- Scientific notation normalization (place right before return) ---
    num_atom_sci = r"[+-]?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?"

    # 1) "6 e -5" -> "6e-5"
    if update := stage_enabled(Stage.SCI_A):
        s2 = re.sub(
            rf"\b({num_atom_sci})\s*e\s*([+-]?\d+)\b",
            r"\1e\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # 2) "6e-5" -> "6{mult}10{power}-5"
    if update := stage_enabled(Stage.SCI_B):
        s2 = re.sub(
            rf"\b([+-]?\d+(?:\.\d+)?)(?:e([+-]?\d+))\b",
            lambda m: f"{m.group(1)}{config.mult}10{config.power}{m.group(2)}",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # 3) "6*10^5", "6×10**(5)", "6 x 10^5" -> "6{mult}10{power}5"
    if update := stage_enabled(Stage.SCI_C):
        ten_pow = rf"10(?:\s*)?(?:\^|\*\*)(?:\s*)?(\(?[+-]?\d+\)?)"
        s2 = re.sub(
            rf"\b({num_atom_sci})\s*(?:\*|x|×)\s*{ten_pow}\b",
            lambda m: f"{m.group(1)}{config.mult}(10{config.power}{m.group(2)})",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # --- ACTUAL_MATH: mixed numbers like "1 and 1/2" ---
    # supports: "1 and 1/2", "1 and 2/3", etc.
    if config.combine_add is not False:  # default True
        _res = 3 if config.res is _sentinel else int(config.res) if config.res is not None else config.res
    else:
        _res = None


    def _mixed_repl(m: re.Match | str, _res) -> str:
        whole_s = m.group(1)
        num_s = m.group(2)
        den_s = m.group(3)

        try:
            whole = Fraction(whole_s)
            num = Fraction(num_s)
            den = Fraction(den_s)
            if den == 0:
                return m.group(0)

            val = whole + (num / den)

            # EXACT-ONLY MODE
            if _res is None:
                # only emit if decimal terminates
                d = val.denominator
                while d % 2 == 0:
                    d //= 2
                while d % 5 == 0:
                    d //= 5
                if d != 1:
                    return m.group(0)

                return frac_to_exact_str(val)

            # ROUNDED MODE
            return frac_to_exact_str(
                val.limit_denominator()  # already exact; formatting handles rounding elsewhere
            )

        except Exception:
            return m.group(0)



    # Only collapse when the RHS is a fraction token your pipeline produced.
    # This avoids touching phrases like "rock and roll".
    if update := stage_enabled(Stage.AND_A):
        s2 = re.sub(
            r"\b([+-]?\d+(?:\.\d+)?)\s+and\s+([+-]?\d+(?:\.\d+)?)/(\d+)\b",
            lambda f: _mixed_repl(f, config.res),
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    # --- unit mixed-number: (a|N) <unit> and a/b  ->  (N + a/b) <unit> ---
    # Assumes "half" etc is already replaced into "1/2" earlier.

    def _unit_and_frac(m: re.Match) -> str:
        whole_txt = m.group(1).lower()   # "a" or digits
        unit = m.group(2)               # e.g. "day", "days"
        a = int(m.group(3))
        b = int(m.group(4))

        whole = 1 if whole_txt == "a" else int(whole_txt)
        val = whole + (a / b)

        # if you have combine_add/res, use them; otherwise just default to 3 decimals
        if config.combine_add is False:
            whole_out = "1" if whole_txt == "a" else m.group(1)
            return f"{whole_out} {unit} and {a}/{b}"

        places = config.res if ("res" in locals() and isinstance(config.res, int)) else 3
        out = f"{val:.{places}f}".rstrip("0").rstrip(".")

        plural_unit = unit
        if unit.endswith("ay"):
            plural_unit = unit + "s"
        elif unit.endswith("ry"):
            plural_unit = unit[:-2] + "ries"
        elif not unit.endswith("s"):
            plural_unit = unit + "s"
        return f"{out} {plural_unit}"

    if update := stage_enabled(Stage.AND_B):
        s2 = re.sub(
            r"\b(a|\d+)\s+(\S+)\s+and\s+(\d+)/(\d+)$",
            _unit_and_frac,
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    if update := stage_enabled(Stage.OF):
        s2 = re.sub(
            r"(\d+(?:\/|\.)\d+) of (?:an?\s?set(?:\(s\))?s? of )?(\d+)",
            rf"(\1){config.mult}\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)
    if update := stage_enabled(Stage.AND_C):
        s2 = re.sub(
            r"(\d+) and (\d+(?:\.|\/)\d+)",
            rf"\1 + \2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)


    if config.do_simple_evals:
        if update := stage_enabled(Stage.SIMPLE_EVALS):
            s2 = simple_eval(s, power=config.power, mult=config.mult, div=config.div, eval_fractions=config.do_fraction_evals,res=config.res)
            s = update(s2)

    if update := stage_enabled(Stage.TIMES):
        s2 = re.sub(
            rf"({num_atom})\s?(?:time|multiplied|timesed|occurence|instance|attempt|multiply|multiple|set)(?:\(s\))?s?(?: (?:by|of))?\s+({num_atom})",
            rf"\1{config.mult}\2",
            s,
            flags=re.IGNORECASE,
        )
        s = update(s2)

    if config.do_simple_evals:
        if update := stage_enabled(Stage.SIMPLE_EVALS_2):
            s2 = simple_eval(s, power=config.power, mult=config.mult, div=config.div, eval_fractions=config.do_fraction_evals,res=config.res)
            s = update(s2)

    # replace hanging
    if update := stage_enabled(Stage.HANGING_SET):
        s2 = re.sub(rf"\b(?:an? )?set(?:\(s\))?s? of (\d)", r"\1", s, flags=re.IGNORECASE)
        s = update(s2)

    def merge_units(s, u, v=1):
        if isinstance(u, UnitGroup):
            s = u.base.base_merge(s) if v==1 else u.base.base_merge2(s)
        elif isinstance(u, Unit):
            s = u.base_merge(s) if v==1 else u.base_merge2(s)
        else:
            for u2 in u:
                s = merge_units(s, u2, v=v)
        return s
    all_base_units = all_units.all_base_units()
    # print("more bases", new_units)

    if update := stage_enabled(Stage.MERGE_UNITS):
        s2 = merge_units(s, all_base_units)
        s = update(s2)
    if update := stage_enabled(Stage.MERGE_UNITS_2):
        s2 = merge_units(s, all_base_units, v=2)
        s = update(s2)

    if config.do_simple_evals:
        if update := stage_enabled(Stage.SIMPLE_EVALS_3):
            s2 = simple_eval(s, power=config.power, mult=config.mult, div=config.div, eval_fractions=config.do_fraction_evals,res=config.res)
            s = update(s2)

    if update := stage_enabled(Stage.MERGE_UNITS_B):
        s2 = merge_units(s, all_base_units)
        s = update(s2)
    if not _iter:
        return s
    first = s
    i = 0
    prevs = [s]
    while i < 100:
        if update := stage_enabled(Stage.ITER):
            s2 = recurse(s)
            s.stages = s2.result.stages
            update(s2, i)
            if s2 == s:
                break
            if s2 in prevs:
                s=first
                break
            if len(s2) > len(s):
                return s
            prevs.append(s2)
            s = s2
        else:
            break
        i += 1
    else:
        raise StopIteration(f"hit {i}")

    # print("almost final", s)
    if config.unit_mode  != unit_modes.BASE:
        groups = [] if isinstance(config.units, Unit) else [g for g in config.units if isinstance(g, UnitGroup)] if not isinstance(config.units, UnitGroup) else [config.units]
        for g in groups:
            # get the first child for each value
            mapping = {k: v if isinstance(v, str) else list(v)[0] for k, v in g.names.items()}
            k = g.base.key
            mapping[1] = k

            mapping = dict(sorted(mapping.items(), reverse=True)) # e.g. {0.000001: 'µs', 0.001: 'ms', 1: 's', 60: 'm', 3600: 'h', 24*3600: 'd', 7*24*3600: 'w'}
            levels = list(mapping.keys()) # [7*24*3600, 23*3600, 3600, 60, 1, 0.001, 0.000001]
            names = list(mapping.values()) # ['w', 'd', 'h', 'm', 's', 'ms', 'µs']

            repl = get_level_merger(levels, names, unit_max_cascade=config.unit_max_cascade)  # or 1/2/3...

            if update := stage_enabled(Stage.UNIT_MERGE):
                s2 = re.sub(
                    rf"(\d+(?:(?:\.|\/)\d+)?)\s*{k}(?![A-Za-z])",
                    lambda m: repl(m, res=config.res, int_cascade_mode=config.int_cascade_mode),
                    s,
                )
                s = update(s2, g)

    s = result.finish(s)
    return str(s) if raw else s




if __name__ == "__main__":
    from digitize.cli import main
    # # print(build_md_table(TESTS))
    # # # pass
    # # # loop(config="units", raise_exc=True)
    # # # loop(raise_exc=True)
    # # # print("simple_eval", simple_eval("5*12 donuts"))
    if not sys.argv[1:]:
        raise SystemExit(main(["", "-m", "demo"]))
    raise SystemExit(main())