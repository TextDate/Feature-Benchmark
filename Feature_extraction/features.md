# 📘 Text Feature Extractor – Feature Descriptions

This README provides detailed explanations for each feature extracted by the `TextFeatureExtractor` module. These features aim to capture various linguistic, statistical, and structural properties of a text file, which can be used for analysis, classification, or machine learning tasks.

---

## Compression and Entropy Features

### `compression_ratio_markov_order_<N>`
- **Description**: Measures how well a Markov model of order `N` can compress the text.
- **How it's computed**: Using the PPM (Prediction by Partial Matching) algorithm, the text is encoded with a model of order `N`. The compression ratio is the ratio of compressed size (in bits) to original size (in bits).
- **Why it's useful**: A more compressible text typically has more structure or redundancy, which is often linked to simpler or more repetitive writing.

### `nrc_markov_order_<N>`
- **Normalized Relative Compression (NRC)**
- **Formula**:  
  \[
  NRC = \frac{C(x||x)}{|x| \cdot \log_2(|A|)}
  \]
  where `C(x||x)` is the compressed length of `x` using a model trained on `x`, and `|A|` is the alphabet size.
- **Why it's useful**: Normalizes compression by text length and alphabet size, making it comparable across different files.

### `nrc_markov_shannon_order_<N>`
- **Description**: Ratio between Markov-based entropy and Shannon entropy.
- **Why it's useful**: This tells us how much additional structure a Markov model captures over the basic entropy estimate.

### `shannon_entropy`
- **Description**: Measures the average information per character.
- **Why it's useful**: Higher entropy means more unpredictability. Low entropy may suggest repetition or low lexical diversity.

---

## Lexical and Structural Features

### `avg_word_length`
- **How it's computed**: Total characters / total words.
- **Insight**: Longer words might indicate more complex vocabulary.

### `lexical_richness`
- **How it's computed**:  
  \[
  \frac{\text{Unique words}}{\text{Total words}}
  \]
- **Insight**: High values indicate diverse vocabulary; low values suggest repetition.

### `avg_sentence_length`
- **How it's computed**: Average number of words per sentence (based on period `.` separators).
- **Insight**: Longer sentences may reflect more complex syntax or literary style.

### `punctuation_density`
- **How it's computed**: Punctuation characters / total characters.
- **Insight**: Reflects syntactic density, style, and tone.

### `uppercase_ratio`
- **How it's computed**: Uppercase letters / total letters.
- **Insight**: May capture titles, acronyms, or emphasis styles.

### `digit_ratio`
- **How it's computed**: Digits / total characters.
- **Insight**: Useful in distinguishing numeric-heavy documents like reports or manuals.

### `special_character_ratio`
- **How it's computed**: Special non-alphanumeric, non-punctuation characters / total characters.
- **Insight**: High values may indicate noise or encoding issues.

---

## Language-Specific Features

### `stopword_ratio`
- **How it's computed**:  
  \[
  \frac{\text{Stopwords in text}}{\text{Total words}}
  \]
- **Insight**: High stopword ratios suggest conversational or narrative style; low may indicate more technical or sparse language.

### `flesch_readability` *(English Only)*
- **How it's computed**: Using `textstat.flesch_reading_ease()`.
- **Insight**:  
  - 90–100: Very easy to read (5th grade)
  - 60–70: Plain English
  - 0–30: College graduate level
- **Limitation**: Only reliable for English texts.

### `syllable_per_word`
- **How it's computed**:  
  \[
  \frac{\text{Total syllables}}{\text{Total words}}
  \]
- **Insight**: Higher values indicate more complex word choices.

---

## Word Distance Features

These capture the average spacing between key functional or target words (e.g., "the", "of", "and").

### `mean_word_distance_<word>`
- **How it's computed**: Mean distance (in words) between repeated occurrences of the same word.
- **Insight**: Reflects rhythm, repetition, and word spacing patterns.

### `median_word_distance_<word>`
- **Similar to above**, but uses the **median** rather than the mean, making it more robust to outliers.

---

## Utility

### `file_name`
- Just the filename for identification.

### `decade`
- The decade in which the text was first published. Used for chronological analysis.

---

## 💬 Language Support Notes

- **Stopword-based metrics** depend on the language and will fallback to empty if unavailable.
- **Flesch readability** is only computed for English due to its language-specific nature.
- **Target words** must be provided in the appropriate language via JSON.