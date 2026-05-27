import re
import numpy as np

FEATURE_ORDER = [
    'text_length',
    'word_count',
    'exclamation_count',
    'has_url',
    'has_phone',
    'has_code',
    'uppercase_ratio'
]

NUM_FEATURES = len(FEATURE_ORDER)

_URL_RE = re.compile(
    r'https?://|www\.|bit\.ly|goo\.gl|tinyurl|t\.co|is\.gd|ow\.ly',
    re.IGNORECASE
)

_PHONE_RE = re.compile(
    r'(?<!\d)(?:\+62|0)[\s\-]?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?!\d)'
)

_CODE_RE = re.compile(
    r'\b[A-Z0-9]{4,12}\b'
)


def extract_num_features(teks: str):

    teks = teks or ""

    text_length = float(
        len(teks)
    )

    words = teks.split()

    word_count = float(
        len(words)
    )


    exclamation_count = float(
        teks.count("!")
    )


    has_url = (
        1.0 if _URL_RE.search(teks)
        else 0.0
    )


    has_phone = (
        1.0 if _PHONE_RE.search(teks)
        else 0.0
    )


    has_code = (
        1.0 if _CODE_RE.search(teks)
        else 0.0
    )

    letters = [
        c
        for c in teks
        if c.isalpha()
    ]

    if len(letters) > 0:

        uppercase_ratio = float(
            sum(
                1
                for c in letters
                if c.isupper()
            ) / len(letters)
        )

    else:

        uppercase_ratio = 0.0

    vector = np.array(
        [
            text_length,
            word_count,
            exclamation_count,
            has_url,
            has_phone,
            has_code,
            uppercase_ratio
        ],
        dtype=np.float32
    )

    vector = vector.reshape(
        1,
        NUM_FEATURES
    )

    return vector

if __name__=="__main__":

    sample="""
    Selamat!
    Anda menang hadiah gratis.
    Klik:
    http://bit.ly/test

    Hubungi:
    08123456789

    KODE2025
    """

    x=extract_num_features(
        sample
    )

    print(x)

    print(
        "shape:",
        x.shape
    )

    for n,v in zip(
        FEATURE_ORDER,
        x[0]
    ):

        print(
            f"{n}: {v}"
        )