# 📘 Text Feature Explanation

This document explains all the extracted features in the `TextFeatureExtractor` class. These features capture **compression-based, statistical, linguistic, and readability properties** of a text document.

## **1️⃣ Compression-Based Features**

These features measure how well a text compresses using different algorithms, capturing **redundancy and predictability** in the data.


| **Feature**                   | **Description**                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| `compression_ratio_zlib`      | Compression ratio using Zlib. Lower ratio = more redundant text.                      |
| `compression_ratio_bz2`       | Compression ratio using Bzip2.                                                         |
| `compression_ratio_lzma`      | Compression ratio using LZMA.                                                          |
| `compression_ratio_brotli`    | Compression ratio using Brotli.                                                        |
| `compression_ratio_zstd`      | Compression ratio using Zstd.                                                          |
| `compression_distance_zlib`   | **Normalized Compression Distance** using Zlib. Higher distance = more different text. |
| `compression_distance_bz2`    | NCD using Bzip2.                                                                       |
| `compression_distance_lzma`   | NCD using LZMA.                                                                       |
| `compression_distance_brotli` | NCD using Brotli.                                                                     |
| `compression_distance_zstd`   | NCD using** Zstd**.                                                                   |
| `compression_speed_zlib`      | **Time taken per byte** for Zlib compression.                                          |
| `compression_speed_bz2`       | Compression speed for Bzip2.                                                           |
| `compression_speed_lzma`      | Compression speed for LZMA.                                                            |
| `compression_speed_brotli`    | Compression speed for Brotli.                                                          |
| `compression_speed_zstd`      | Compression speed for Zstd.                                                            |

## **2️⃣ Statistical Features**

These features describe **the structure and complexity of the text**.


| **Feature**               | **Description**                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| `shannon_entropy`         | Measures**character-level randomness** in text. Higher entropy = more unpredictable text. |
| `avg_word_length`         | Average word length in characters.                                                        |
| `lexical_richness`        | Ratio of**unique words / total words** (measures vocabulary diversity).                   |
| `avg_sentence_length`     | Average number of words per sentence.                                                     |
| `punctuation_density`     | % of punctuation marks in the text.                                                       |
| `stopword_ratio`          | % of**common stopwords** (e.g., "the", "and", "is").                                      |
| `uppercase_ratio`         | % of uppercase letters in the text.                                                       |
| `digit_ratio`             | % of digits in the text.                                                                  |
| `special_character_ratio` | % of**non-alphanumeric** special characters.                                              |

## **3️⃣ Readability & Complexity Features**

These features measure **text difficulty and linguistic properties**.


| **Feature**              | **Description**                                                                  |
| ------------------------ | -------------------------------------------------------------------------------- |
| `flesch_readability`     | **Flesch-Kincaid Readability** score (higher = easier to read).                  |
| `syllable_per_word`      | Average**syllables per word** (higher = more complex words).                     |
| `dale_chall_readability` | Alternative readability score using**common word lists** (lower = easier).       |
| `entropy_per_word`       | Shannon entropy**normalized by word count** (measures randomness per word).      |
| `passive_voice_ratio`    | % of sentences using**passive voice** (historical texts use more passive voice). |

## **4️⃣ Vocabulary Richness & Style**

These features capture **writing style, richness, and uniqueness**.


| **Feature**            | **Description**                                              |
| ---------------------- | ------------------------------------------------------------ |
| `type_token_ratio`     | **Unique words / total words** (higher = richer vocabulary). |
| `hapax_legomena_ratio` | % of words that appear**only once** in the text.             |

## **📌 Why These Features Matter**

These features help capture:

- **Changes in language over time** (older texts have different vocabulary & structure).
- **Compression & randomness patterns** (complex texts are harder to compress).
- **Readability differences** (historical texts are more complex).
- **Stylistic choices** (passive voice, punctuation use, and vocabulary richness).

These insights can **help classify texts by decade or author** based on their writing style and complexity! 🚀
