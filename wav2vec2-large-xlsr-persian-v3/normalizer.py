from parsivar import Normalizer
from dictionary import dictionary_mapping, fixator_dictionary

import num2fawords
import re
import string


_normalizer = Normalizer(half_space_char="\u200c", statistical_space_correction=True)
chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬", 'ٔ', ",", "?",
    ".", "!", "-", ";", ":", '"', "“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š', 'ّ', 'ْ',
]
chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)
chars_to_ignore = f"""[{"".join(chars_to_ignore)}]"""
zwnj = "\u200c"
silent_chars = ["ا", "د", "ذ", "ر", "ز", "و", "آ"] + [zwnj] + [" "]


def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))


def remove_special_characters(text, chars_to_ignore_regex):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text


def convert_word_nums_to_text(word):
    try:
        word = int(word)
        word = num2fawords.words(word)
    except:
        word = word

    return word


def normalizer_at_word_level(text):
    words = text.split()
    _text = []

    for word in words:
        word = convert_word_nums_to_text(word)
        word = fixator_dictionary.get(word, word)

        _text.append(word)

    return " ".join(_text) + " "


def finder(ss, s, starter=False):
    found = []
    for m in re.finditer(ss, s):
        if starter:
            found.append(m.start())
        else:
            found.append((m.start(), m.end()))

    return found


def substring_replace(ss, s, start, end, stripped=True):
    s_start = s[:start]
    s_end = s[end:]

    counter = 0
    if stripped:
        counter = 1 if s_start.endswith(" ") else counter
        s_start = s_start.rstrip()

    return s_start + ss + s_end, counter


def normalizer(
        batch,
        is_normalize=True,
        return_dict=True,
        filter_trivials=False,
        remove_extra_space=False
):
    text = batch["sentence"].lower().strip()

    # Parsivar normalizer
    if is_normalize:
        text = _normalizer.normalize(text)

    # Dictionary mapping
    text = multiple_replace(text, dictionary_mapping)
    text = re.sub(" +", " ", text)

    # Remove specials
    text = remove_special_characters(text, chars_to_ignore)
    text = re.sub(" +", " ", text)

    # Replace connected آ
    special, pointer = "آ", int("0")
    for f in sorted(finder(special, text, True)):
        index = f + pointer - 1
        if len(text) >= index:
            if text[index] not in silent_chars:
                new_text, extra_pointer = substring_replace(
                    f"{text[index]}{zwnj}", text, index, index + 1, stripped=True)
                text = new_text
                pointer += 1 + 1 - 1 - extra_pointer

    # Replace connected ها
    pointer = int("0")
    special_list = [
        # "ام", "ای", "است", "ایم", "اید", "اند",
        "هایمان", "هایم", "هایت", "هایش",
        "هایتان", "هایشان", "هام", "هات",
        "هاتان", "هامون", "هامان", "هاش",
        "هاتون", "هاشان", "هاشون",
        "هایی", "های", "هاس", "ها"
    ]
    for special in special_list:
        pointer = 0
        text = text
        for f in sorted(finder(special, text, False)):
            start, end = f[0] + pointer - 1, f[1] + pointer - 1
            if len(text) >= (end + 1):
                if len(text) == (end + 1):
                    new_text, extra_pointer = substring_replace(
                        f"{zwnj}{special}",
                        text,
                        start + 1,
                        end + 1,
                        stripped=True)
                    text = new_text
                    pointer += 1 + 1 - 1 - extra_pointer
                else:
                    if text[end + 1] == " ":
                        new_text, extra_pointer = substring_replace(
                            f"{zwnj}{special}",
                            text,
                            start + 1,
                            end + 1,
                            stripped=True)
                        text = new_text
                        pointer += 1 + 1 - 1 - extra_pointer

    special, pointer = "افزار", int("0")
    for f in sorted(finder(special, text, False)):
        start, end = f[0] + pointer - 1, f[1] + pointer - 1

        if len(text) >= (end + 1):
            new_text, extra_pointer = substring_replace(f"{zwnj}{special}", text, start + 1, end + 1, stripped=True)
            text = new_text
            pointer += 1 + 1 - 1 - extra_pointer

    # Replace connected ها
    pointer = int("0")
    special_list = [
        "ترین", "تر"
    ]
    for special in special_list:
        pointer = 0
        text = text
        for f in sorted(finder(special, text, False)):
            start, end = f[0] + pointer - 1, f[1] + pointer - 1
            if len(text) >= (end + 1):
                if len(text) == (end + 1):
                    new_text, extra_pointer = substring_replace(
                        f"{zwnj}{special}",
                        text,
                        start + 1,
                        end + 1,
                        stripped=True)
                    text = new_text
                    pointer += 1 + 1 - 1 - extra_pointer
                else:
                    if text[end + 1] == " ":
                        new_text, extra_pointer = substring_replace(
                            f"{zwnj}{special}",
                            text,
                            start + 1,
                            end + 1,
                            stripped=True)
                        text = new_text
                        pointer += 1 + 1 - 1 - extra_pointer

    # Normalizer at word level
    text = normalizer_at_word_level(text)
    text = re.sub(" +", " ", text)

    if remove_extra_space:
        text = text.strip()
    else:
        text = text.strip() + " "

    if filter_trivials:
        if not len(text) > 2:
            text = None

    if not return_dict:
        return text

    batch["sentence"] = text
    return batch

