# Emilia-zh+en

## Language distribution

Unit: second

1k: {'en': 1731768, 'zh': 1861438, 'ko': 13}
5k: {'zh': 9491499, 'en': 8497319, 'fr': 15, 'ko': 57, 'de': 232}
10k: {'zh': 18910917, 'en': 17067984, 'ja': 138, 'fr': 110, 'ko': 159, 'de': 205}
50k: {'zh': 94747317, 'en': 85131075, 'ko': 587, 'fr': 1788, 'de': 2166, 'ja': 248}

## Format

```json
[
    {
        "vid": "xmly00000_68439925_542730030",
        "json_path": "qianyi/mp3/xima-en-lixuyuan-part1/xmly00000_68439925_542730030/xmly00000_68439925_542730030.json",
        "wav_path": [
            "qianyi/mp3/xima-en-lixuyuan-part1/xmly00000_68439925_542730030/xmly00000_68439925_542730030_2.wav",
            "qianyi/mp3/xima-en-lixuyuan-part1/xmly00000_68439925_542730030/xmly00000_68439925_542730030_3.wav",
            "qianyi/mp3/xima-en-lixuyuan-part1/xmly00000_68439925_542730030/xmly00000_68439925_542730030_4.wav",
            "qianyi/mp3/xima-en-lixuyuan-part1/xmly00000_68439925_542730030/xmly00000_68439925_542730030_7.wav"
        ],
        "language": [
            "en",
            "en",
            "en",
            "en"
        ],
        "duration": [
            11.137999999999998,
            12.733000000000004,
            5.263000000000005,
            6.570999999999998
        ],
        "phone_count": [
            82,
            116,
            55,
            68
        ],
        "valid_duration": 81,
        "filtered_duration": 35,
        "filtered_idx": "2,3,4,7"
    }
]
```


## Usage

```python
import gzip
import json


# Save JSON data to a compressed GZIP file
def save_compressed_json(data, filename):
    with gzip.open(filename, "wt", encoding="utf-8") as f:
        json.dump(data, f)


# Load JSON data from a compressed GZIP file
def load_compressed_json(filename):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        return json.load(f)


def decompress_json(filename):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        with open(filename.replace(".gz", ""), "w") as f_out:
            f_out.write(f.read())
```