from ast import arg
import itertools
import json
import os
import string
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from nltk.corpus import stopwords
import textstat
from tqdm import tqdm
import vomm
import traceback
import argparse
import psutil
import threading
import time
import multiprocessing
import gc



class TextFeatureExtractor:
    def __init__(self, file_info_path, lang="english", num_threads=8, lowercase=False):
        """Initializes the feature extractor with stopwords and file info."""
        self.file_info_path = file_info_path
        self.file_info = pd.read_csv(file_info_path)[["file_name", "year", "decade", "century",]]
        self.language = lang
        self.stop_words = self.get_stopwords(lang)
        self.num_threads = num_threads
        self.lowercase = lowercase


    def extract_features_from_directory(self, directory: str, output_csv: str, target_words: set[str],
                                        order=1, alphabet_size=256, word_distance_type="mean", batch_size=1000):
        """
        Extract features in memory-safe batches using multiprocessing.
        Each batch starts a new process pool to limit memory growth.
        """
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if file.endswith(".txt")
        ]

        print(f"Found {len(all_files)} text files in {directory}", flush=True)
        total_batches = math.ceil(len(all_files) / batch_size)
        output_paths = []
        

        for i in range(total_batches):
            results = []
            timeout_files = []
            batch_path = f"{output_csv}_batch_{i:03}.csv"
            if os.path.exists(batch_path):
                print(f"[SKIP] Batch {i+1} already processed ({batch_path})")
                output_paths.append(batch_path)
                continue
            
            
            batch = all_files[i * batch_size:(i + 1) * batch_size]
            print(f"\nBatch {i + 1}/{total_batches}: Processing files {i * batch_size}–{i * batch_size + len(batch) - 1}", flush=True)
            

            args_iter = list(zip(
                batch,
                itertools.repeat(target_words),
                itertools.repeat(order),
                itertools.repeat(alphabet_size),
                itertools.repeat(word_distance_type),
                itertools.repeat(self.file_info_path),
                itertools.repeat(self.language),
                itertools.repeat(self.num_threads)
            ))

            with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
                for result, args in zip(tqdm(executor.map(extract_with_timeout_wrapper, args_iter), desc=f"Processing Batch {i+1}" ,total=len(args_iter)), args_iter):
                    if result is None:
                        timeout_files.append(args)
                    else:
                        results.append(result)

            batch_df = pd.DataFrame(results)
            batch_path = f"{output_csv}_batch_{i:03}.csv"
            batch_df.to_csv(batch_path, index=False)
            output_paths.append(batch_path)

            print(f"[MEMORY] After batch {i+1} memory usage:")
            self.log_mem()

            del results, batch_df, batch
            gc.collect()

        dfs = [pd.read_csv(p) for p in output_paths]
        final_df = pd.concat(dfs, ignore_index=True)
        final_df = final_df.merge(self.file_info, on="file_name", how="left")
        final_df.to_csv(output_csv, index=False)
        
        if timeout_files:
            print(f"\n[INFO] Retrying {len(timeout_files)} timed out files sequentially...", flush=True)
            retried = []

            for args in tqdm(timeout_files, desc="Retrying timeouts", unit="file"):
                result = extract_with_timeout_worker(*args)
                if result:
                    retried.append(result)

            if retried:
                retry_df = pd.DataFrame(retried)
                retry_df = retry_df.merge(self.file_info, on="file_name", how="left")
                retry_df.to_csv(f"{output_csv}_retries.csv", index=False)

                final_df = pd.concat([final_df, retry_df], ignore_index=True)
                final_df.to_csv(output_csv, index=False)

                print(f"[INFO] Appended {len(retried)} retried files to final output.", flush=True)


        print(f"\nFeature extraction complete. Data saved to {output_csv}", flush=True)
        self.log_mem("Final ")



    def _extract_from_file_bound(self, args):
        text_path, target_words, order, alphabet_size, word_distance_type = args
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()

            features = self.extract(text, target_words,
                                    order=order,
                                    alphabet_size=alphabet_size,
                                    word_distance_type=word_distance_type)
            features["file_name"] = os.path.basename(text_path)
            return features
        except Exception as e:
            print(f"Error processing {text_path}: {e}", flush=True)
            return None



    def extract(self, text:str, target_words=set[str], order=1, alphabet_size=256, word_distance_type="mean"):
        """Extracts all features dynamically from a given text."""
        if self.lowercase:
            text = text.lower()
        features = {
            "compression_ratio_markov_order_1": self.compression_ratio_markov(text, order=1),
            "nrc_markov_order_1": self.nrc_markov(text, order=1, alphabet_size=alphabet_size),
            "nrc_markov_shannon_order_1": self.nrc_markov_shannon(text, order=1),
            f"compression_ratio_markov_order_{order}": self.compression_ratio_markov(text, order=order),
            f"nrc_markov_order_{order}": self.nrc_markov(text, order=order, alphabet_size=alphabet_size),
            f"nrc_markov_shannon_order_{order}": self.nrc_markov_shannon(text, order=order),
            "shannon_entropy": self.shannon_entropy(text),
            "avg_word_length": self.average_word_length(text),
            "lexical_richness": self.lexical_richness(text),
            "avg_sentence_length": self.average_sentence_length(text),
            "punctuation_density": self.punctuation_density(text),
            "syllable_per_word": self.syllable_per_word(text),
            "uppercase_ratio": self.uppercase_ratio(text),
            "digit_ratio": self.digit_ratio(text),
            "special_character_ratio": self.special_character_ratio(text)
        }
        gc.collect()
        
        if self.language == "english":
            features["flesch_readability"] = self.flesch_readability(text)
        
        if len(self.stop_words) > 0:
            features["stopword_ratio"] = self.stopword_ratio(text)
            
        if target_words:
            
            if word_distance_type == "mean":
                word_distances = self.mean_word_distance(text, target_words)
                
            elif word_distance_type == "median":
                word_distances = self.median_word_distance(text, target_words)

            for feature_name, value in word_distances.items():
                features[feature_name] = value
        gc.collect()
        
        return features

    def compression_ratio_markov(self, text:str, order=1):
        if not text:
            return None
        try:
            model = vomm.ppm()
            data = [ord(char) for char in text]
            model.fit(data, d=order)
            compressed_length = -model.logpdf(data) / math.log(2)
            original_length = len(data) * 8
            return compressed_length / original_length
        except Exception as e:
            print(f"[vomm] Crash in compression_ratio_markov: {e}", flush=True)
            traceback.print_exc()
            return None


    def nrc_markov(self, text:str, order=1, alphabet_size=256):
        """
        Calculates NRC(x‖x) = C(x‖x) / (|x| * log2 |A|)

        :param text: The text to model and compress
        :param order: Order of the Markov (PPM) model
        :param alphabet_size: Alphabet size (256 for full ASCII/byte range)
        :return: NRC value (float)
        """
        if not text:
            return None

        data = [ord(c) for c in text]

        model = vomm.ppm()
        model.fit(data, d=order)

        compressed_bits = -model.logpdf(data) / math.log(2)
        max_bits = len(data) * math.log2(alphabet_size)

        return compressed_bits / max_bits if max_bits > 0 else None
    
    def nrc_markov_shannon(self, text:str, order=1):
        """Calculates the Normalized Relative Compression (NRC) using an order-1 Markov model."""
        if not text:
            return 0
        
        shannon_entropy = self.shannon_entropy(text)
        
        markov_entropy = self.compression_ratio_markov(text, order=order) * 8
        
        return markov_entropy / shannon_entropy if shannon_entropy > 0 else 0

    def shannon_entropy(self, text:str):
        """Computes Shannon entropy."""
        if not text:
            return 0
        char_counts = Counter(text)
        total_chars = len(text)
        return -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())

    def average_word_length(self, text:str):
        """Calculates the average word length."""
        words = text.split()
        return 0 if not words else sum(len(word) for word in words) / len(words)

    def lexical_richness(self, text:str):
        """Computes lexical richness."""
        words = text.split()
        return 0 if not words else len(set(words)) / len(words)

    def average_sentence_length(self, text:str):
        """Computes average sentence length in words."""
        sentences = text.split('. ')
        words = [sentence.split() for sentence in sentences]
        return 0 if not words else sum(len(sentence) for sentence in words) / len(words)
    
    def get_stopwords(self, lang="english"):
        try:
            return set(stopwords.words(lang))
        except OSError:
            print(f"Stopwords not available for language: {lang}, defaulting to empty set.", flush=True)
            return set()

    def punctuation_density(self, text:str):
        return 0 if not text else sum(1 for char in text if char in string.punctuation) / len(text)

    def stopword_ratio(self, text:str):
        words = text.split()
        return 0 if not words else sum(1 for word in words if word.lower() in self.stop_words) / len(words)

    def flesch_readability(self, text:str):
        return textstat.flesch_reading_ease(text)

    def syllable_per_word(self, text:str):
        words = text.split()
        return 0 if not words else sum(textstat.syllable_count(word) for word in words) / len(words)

    def uppercase_ratio(self, text:str):
        total_letters = sum(1 for char in text if char.isalpha())
        return 0 if total_letters == 0 else sum(1 for char in text if char.isupper()) / total_letters

    def digit_ratio(self, text:str):
        return 0 if len(text) == 0 else sum(1 for char in text if char.isdigit()) / len(text)

    def special_character_ratio(self, text:str):
        return 0 if len(text) == 0 else sum(1 for char in text if not char.isalnum() and char not in string.punctuation and not char.isspace()) / len(text)

    def mean_word_distance(self, text:str, target_words:list[str]):
        """Computes the mean distance (in words) between occurrences of specified target words."""
        words = text.lower().split()
        word_positions = defaultdict(list)

        for index, word in enumerate(words):
            if word in target_words:
                word_positions[word].append(index)

        mean_distances = {}
        for word, positions in word_positions.items():
            if len(positions) > 1:
                distances = np.diff(positions)
                mean_distances[word] = np.mean(distances)
            else:
                mean_distances[word] = None

        return mean_distances
    
    def median_word_distance(self, text:str, target_words:list[str]):
        """Computes the median distance (in words) between occurrences of specified target words."""
        words = text.lower().split()
        word_positions = defaultdict(list)

        for index, word in enumerate(words):
            if word in target_words:
                word_positions[word].append(index)

        median_distances = {}
        for word, positions in word_positions.items():
            if len(positions) > 1:
                distances = np.diff(positions)
                median_distances[word] = np.median(distances)
            else:
                median_distances[word] = None

        return median_distances
    
    def compute_global_alphabet_size(self, directories: list[str]) -> int:
        """
        Scans multiple directories and computes the number of unique characters
        across all text files in those directories, with tqdm progress bar.
        """
        unique_chars = set()
        all_files = []

        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".txt"):
                        all_files.append(os.path.join(root, file))

        print(f"Found {len(all_files)} text files to determine global alphabet size.", flush=True)
        
        for file_path in tqdm(all_files, desc="Scanning Files", unit="file"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if self.lowercase:
                        text = text.lower()
                    unique_chars.update(text)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}", flush=True)

        print(f"Global alphabet size: {len(unique_chars)}", flush=True)
        
        return len(unique_chars)
    
    def load_target_words(self, path=None):
        if not path:
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(w.strip().lower() for w in f.readlines() if w.strip())

    def log_mem(self, prefix=""):
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        print(f"[MEMORY] {prefix}Memory usage: {mem:.2f} GB", flush=True)
        
def start_memory_logger(interval=300):
    """Print memory usage every `interval` seconds."""
    def log():
        while True:
            mem = psutil.Process().memory_info().rss / 1024**3
            print(f"[MEMORY] Current memory usage: {mem:.2f} GB", flush=True)
            time.sleep(interval)
    threading.Thread(target=log, daemon=True).start()
    
def extract_with_timeout_worker(text_path, target_words, order, alphabet_size, word_distance_type, file_info_path, lang, num_threads):
    return timeout_worker(
        do_work,
        (text_path, target_words, order, alphabet_size, word_distance_type, file_info_path, lang, num_threads),
        timeout=300
    )

def do_work(text_path, target_words, order, alphabet_size, word_distance_type, file_info_path, lang, num_threads):
    with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()

    extractor = TextFeatureExtractor(file_info_path, lang=lang, num_threads=num_threads)
    features = extractor.extract(
        text,
        target_words=target_words,
        order=order,
        alphabet_size=alphabet_size,
        word_distance_type=word_distance_type
    )
    features["file_name"] = os.path.basename(text_path)
    return features


def extract_with_timeout_wrapper(args):
    return extract_with_timeout_worker(*args)


def _run_with_result(func, args, return_dict):
    return_dict["res"] = func(*args)

def timeout_worker(func, args, timeout=300):
    """Runs a function in a subprocess with timeout."""
    with multiprocessing.Manager() as manager:
        return_dict = manager.dict()
        p = multiprocessing.Process(target=_run_with_result, args=(func, args, return_dict))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return None
        return return_dict.get("res", None)
    
def extract_missing_files(csv_path: str, source_dir: str, extractor: TextFeatureExtractor,
                          target_words: set[str], order: int = 1, alphabet_size: int = 256,
                          word_distance_type: str = "mean", batch_size: int = 50):
    """
    Finds files in `source_dir` missing from `csv_path` and extracts features for them in batches.
    """
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        extracted_files = set(existing_df['file_name'].dropna().unique())
    else:
        existing_df = pd.DataFrame()
        extracted_files = set()

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_dir)
        for file in files if file.endswith(".txt")
    ]
    all_filenames = set(os.path.basename(path) for path in all_files)
    missing_filenames = all_filenames - extracted_files

    if not missing_filenames:
        print("[INFO] No missing files found.")
        return

    print(f"[INFO] Found {len(missing_filenames)} missing files. Extracting now...")

    filename_to_path = {os.path.basename(p): p for p in all_files}
    missing_paths = [filename_to_path[f] for f in missing_filenames]

    for batch_idx in range(0, len(missing_paths), batch_size):
        batch = missing_paths[batch_idx:batch_idx + batch_size]
        results = []

        for path in tqdm(batch, desc=f"Extracting batch {batch_idx//batch_size+1}", unit="file"):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                features = extractor.extract(text, target_words, order, alphabet_size, word_distance_type)
                features["file_name"] = os.path.basename(path)
                results.append(features)
            except Exception as e:
                print(f"[ERROR] Failed to process {path}: {e}")

        if results:
            new_df = pd.DataFrame(results)
            new_df = new_df.merge(extractor.file_info, on="file_name", how="left")
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            existing_df = combined_df
            print(f"[INFO] Appended {len(results)} records in batch {batch_idx//batch_size+1} to {csv_path}")
        else:
            print(f"[INFO] No new data extracted in batch {batch_idx//batch_size+1}")

        gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Extract text features from file datasets.")
    parser.add_argument("--file_info", required=True, help="Path to file_info.csv")
    parser.add_argument("--train", required=False, help="Directory containing training text files")
    parser.add_argument("--train_out", required=False, help="Output CSV for training features")
    parser.add_argument("--valid", required=False, help="Directory containing validation text files")
    parser.add_argument("--valid_out", required=False, help="Output CSV for validation features")
    parser.add_argument("--test", required=False, help="Directory containing test text files")
    parser.add_argument("--test_out", required=False, help="Output CSV for test features")
    parser.add_argument("--gutenberg", required=False, help="Directory containing Gutenberg text files")
    parser.add_argument("--gutenberg_out", required=False, help="Output CSV for Gutenberg features")
    parser.add_argument("--target_words", required=True, help="Path to target_words.json")
    parser.add_argument("--lang", default="english", help="Language for stopword filtering and word lists")
    parser.add_argument("--order", type=int, default=1, help="Markov model order")
    parser.add_argument("--word_distance", choices=["mean", "median"], default="mean", help="Distance type for target words")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Number of files to process in each chunk")
    parser.add_argument("--threads", type=int, default=8, help="Number of parallel threads")
    parser.add_argument("--lowercase", action="store_true", help="Convert all text to lowercase before processing")
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    args = parse_args()
    
    start_memory_logger(interval=300)

    extractor = TextFeatureExtractor(
        file_info_path=args.file_info,
        lang=args.lang,
        num_threads=args.threads,
        lowercase=args.lowercase
    )

    if args.target_words:
        with open(args.target_words, "r", encoding="utf-8") as f:
            word_config = json.load(f)
        target_words = set(word_config.get(args.lang, []))
    else:
        target_words = set()

    if args.train and args.valid and args.test:
        global_alphabet_size = extractor.compute_global_alphabet_size([args.train, args.valid, args.test])
    elif args.gutenberg:
        global_alphabet_size = extractor.compute_global_alphabet_size([args.gutenberg])
    if args.train:
        extractor.extract_features_from_directory(args.train, args.train_out, target_words, args.order, global_alphabet_size, args.word_distance, args.chunk_size)
        extract_missing_files(
    csv_path=args.train_out,
    source_dir=args.train,
    extractor=extractor,
    target_words=target_words,
    order=args.order,
    alphabet_size=global_alphabet_size,
    word_distance_type=args.word_distance
    )
    if args.valid:
        extractor.extract_features_from_directory(args.valid, args.valid_out, target_words, args.order, global_alphabet_size, args.word_distance, args.chunk_size)
        extract_missing_files(
    csv_path=args.valid_out,
    source_dir=args.valid,
    extractor=extractor,
    target_words=target_words,
    order=args.order,
    alphabet_size=global_alphabet_size,
    word_distance_type=args.word_distance
    )
    if args.test:
        extractor.extract_features_from_directory(args.test, args.test_out, target_words, args.order, global_alphabet_size, args.word_distance, args.chunk_size)
        extract_missing_files(
    csv_path=args.test_out,
    source_dir=args.test,
    extractor=extractor,
    target_words=target_words,
    order=args.order,
    alphabet_size=global_alphabet_size,
    word_distance_type=args.word_distance
    )
    if args.gutenberg:
        extractor.extract_features_from_directory(args.gutenberg, args.gutenberg_out, target_words, args.order, global_alphabet_size, args.word_distance, args.chunk_size)        
        extract_missing_files(
    csv_path=args.gutenberg_out,
    source_dir=args.gutenberg,
    extractor=extractor,
    target_words=target_words,
    order=args.order,
    alphabet_size=global_alphabet_size,
    word_distance_type=args.word_distance
    )

    

    
    print("[INFO] Feature extraction completed for all datasets.")
    print("[INFO] Memory usage at the end of the script:")
    extractor.log_mem("Final ")
    print("[INFO] All done!")
