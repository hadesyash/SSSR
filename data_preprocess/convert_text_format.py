"""
Convert spoken-format text labels to Gaddy's digit format.

Spoken: "eight thirty three pm" → Gaddy: "08:33 PM"
Spoken: "friday june eleven" → Gaddy: "Friday June 11"
Spoken: "september twelve twenty thirteen" → Gaddy: "September 12 2013"

Usage:
  # Update JSON files in a features directory:
  python convert_text_format.py --features_dir ~/aml_lab/data/combined_emg_features

  # Dry run (show conversions without modifying files):
  python convert_text_format.py --features_dir ~/aml_lab/data/combined_emg_features --dry_run

  # Convert a single text string:
  python convert_text_format.py --text "eight thirty three pm"
"""

import os
import json
import glob
import argparse

# ==========================================================================
# Word-to-number mappings
# ==========================================================================
WORD_TO_NUM = {
    'oh': 0, 'zero': 0,
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}

WEEKDAYS = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
MONTHS = {'january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december'}


def parse_number(words):
    """Parse a number from the start of a word list.
    Handles: single digits, teens, tens, compounds (twenty five), oh-prefix (oh nine).
    Returns: (number, remaining_words) or (None, original_words).
    """
    if not words:
        return None, words

    w = words[0].lower()
    if w not in WORD_TO_NUM:
        return None, words

    n = WORD_TO_NUM[w]
    rest = words[1:]

    # Compound: "twenty"/"thirty"/etc + single digit (1-9)
    if n in (20, 30, 40, 50, 60, 70, 80, 90) and rest:
        w2 = rest[0].lower()
        if w2 in WORD_TO_NUM and 1 <= WORD_TO_NUM[w2] <= 9:
            return n + WORD_TO_NUM[w2], rest[1:]

    # "oh" + single digit (1-9) → that digit (formats as 0X with :02d)
    if n == 0 and rest:
        w2 = rest[0].lower()
        if w2 in WORD_TO_NUM and 1 <= WORD_TO_NUM[w2] <= 9:
            return WORD_TO_NUM[w2], rest[1:]

    return n, rest


def parse_year(words):
    """Parse a 4-digit year from words like 'twenty thirteen' or 'twenty oh nine'.
    Century (19/20) + year-within-century.
    Returns: (year, remaining_words) or (None, original_words).
    """
    if not words:
        return None, words

    # Parse century part
    century, rest = parse_number(words)
    if century is None or century not in (19, 20):
        return None, words

    # Parse year-within-century
    year_part, rest2 = parse_number(rest)
    if year_part is None:
        return None, words

    return century * 100 + year_part, rest2


def spoken_to_gaddy(text):
    """Convert a spoken-format text label to Gaddy's digit format.

    Handles these phrase types:
    1. Time with AM/PM:   "eight thirty three pm"  → "08:33 PM"
    2. Time without AM/PM: "one oh two"            → "01:02"
    3. Time with o'clock:  "eleven o'clock am"     → "11:00 AM"
    4. Weekday+Month+Date: "friday june eleven"    → "Friday June 11"
    5. Month+Date+Year:    "september twelve twenty thirteen" → "September 12 2013"
    6. Weekday+Month+Date+Year: "wednesday june four twenty twenty eight" → "Wednesday June 04 2028"
    """
    # Normalize o'clock
    normalized = text.strip().lower().replace("o'clock", "oclock")
    words = normalized.split()

    if not words:
        return text

    # Detect phrase components
    has_weekday = words[0] in WEEKDAYS
    month_idx = None
    for i, w in enumerate(words):
        if w in MONTHS:
            month_idx = i
            break
    has_month = month_idx is not None
    has_ampm = words[-1] in ('am', 'pm')
    has_oclock = 'oclock' in words

    # --- Type: Weekday + Month + Date [+ Year] ---
    if has_weekday and has_month:
        weekday = words[0].title()
        month = words[month_idx].title()

        after_month = words[month_idx + 1:]
        # Remove am/pm if present (shouldn't be, but safety)
        if after_month and after_month[-1] in ('am', 'pm'):
            after_month = after_month[:-1]

        date_num, remaining = parse_number(after_month)
        if date_num is None:
            return text  # Can't parse

        result = f"{weekday} {month} {date_num:02d}"

        # Check for year
        if remaining:
            year, _ = parse_year(remaining)
            if year:
                result = f"{weekday} {month} {date_num:02d} {year}"

        return result

    # --- Type: Month + Date + Year ---
    if has_month and not has_weekday:
        month = words[month_idx].title()

        after_month = words[month_idx + 1:]
        date_num, remaining = parse_number(after_month)
        if date_num is None:
            return text

        result = f"{month} {date_num:02d}"

        if remaining:
            year, _ = parse_year(remaining)
            if year:
                result = f"{month} {date_num:02d} {year}"

        return result

    # --- Type: Time with o'clock ---
    if has_oclock:
        oclock_idx = words.index('oclock')
        hour_words = words[:oclock_idx]
        hour_num, _ = parse_number(hour_words)
        if hour_num is None:
            return text

        result = f"{hour_num:02d}:00"
        if has_ampm:
            result += f" {words[-1].upper()}"
        return result

    # --- Type: Time (with or without AM/PM) ---
    time_words = words[:-1] if has_ampm else words

    hour_num, remaining = parse_number(time_words)
    if hour_num is None:
        return text

    if remaining:
        minute_num, _ = parse_number(remaining)
        if minute_num is None:
            minute_num = 0
    else:
        minute_num = 0

    result = f"{hour_num:02d}:{minute_num:02d}"
    if has_ampm:
        result += f" {words[-1].upper()}"

    return result


# ==========================================================================
# Main: update JSON files or test single string
# ==========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert spoken text to Gaddy format')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='Directory with feature JSON files to update')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show conversions without modifying files')
    parser.add_argument('--text', type=str, default=None,
                        help='Convert a single text string')
    args = parser.parse_args()

    if args.text:
        print(f'"{args.text}" → "{spoken_to_gaddy(args.text)}"')
        exit(0)

    if not args.features_dir:
        print("Provide --features_dir or --text")
        exit(1)

    json_files = sorted(glob.glob(os.path.join(args.features_dir, '*.json')))
    json_files = [f for f in json_files if 'norm_stats' not in f and 'split' not in f]

    updated = 0
    unchanged = 0
    errors = []

    for jp in json_files:
        with open(jp) as f:
            meta = json.load(f)

        original = meta['text']
        converted = spoken_to_gaddy(original)

        if original == converted:
            unchanged += 1
            if args.dry_run:
                print(f"  UNCHANGED: \"{original}\"")
        else:
            updated += 1
            if args.dry_run:
                print(f"  \"{original}\" → \"{converted}\"")
            else:
                meta['text'] = converted
                meta['original_spoken_text'] = original
                with open(jp, 'w') as f:
                    json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'DRY RUN' if args.dry_run else 'DONE'}: {updated} updated, {unchanged} unchanged")
    if not args.dry_run:
        print(f"Original spoken text saved in 'original_spoken_text' field")
    print(f"{'='*60}")
