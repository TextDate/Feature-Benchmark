import json
import os
import string
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
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
import random
import pickle
import signal
import resource
import sys
import warnings
import logging
# Neologism detection integrated directly in TextFeatureExtractor

# Global variables for worker initialization
_global_extractor = None
_global_target_words = None
_global_reference_models = None
_global_reference_data = None




class TextFeatureExtractor:
    def __init__(self, file_info_path, lang="english", num_threads=8, lowercase=False, reference_text=None, order=1, prebuilt_models=None, prebuilt_reference_data=None, use_ncd=True):
        """Initializes the feature extractor with stopwords and file info."""
        self.file_info_path = file_info_path
        self.file_info = pd.read_csv(file_info_path)[["file_name", "year", "decade", "century",]]
        self.language = lang
        self.stop_words = self.get_stopwords(lang)
        self.num_threads = num_threads
        self.lowercase = lowercase
        self.reference_text = reference_text
        self.reference_models = prebuilt_models if prebuilt_models is not None else {}
        self.use_ncd = use_ncd

        # Initialize neologism detection data
        self.neologism_words = {}
        self.neologism_periods = {}
        self._load_neologism_data()

        # Use pre-computed reference data if provided, otherwise compute it
        if prebuilt_reference_data is not None:
            self._reference_data = prebuilt_reference_data
        elif self.reference_text:
            ref_text = self.reference_text.lower() if self.lowercase else self.reference_text
            self._reference_data = [ord(char) for char in ref_text]
        else:
            self._reference_data = None

        # Only build reference models if they weren't provided and reference text is available
        if self.reference_text and not prebuilt_models:
            self._build_reference_model(1)  # Always build for order=1
            if order != 1:
                self._build_reference_model(order)  # Build for the specified order if different

    def _build_reference_model(self, order):
        """Build reference model for a specific order on-demand with caching."""
        if not self.reference_text or order in self.reference_models:
            return

        # Use cached reference data if available
        if not hasattr(self, '_reference_data') or self._reference_data is None:
            ref_text = self.reference_text.lower() if self.lowercase else self.reference_text
            self._reference_data = [ord(char) for char in ref_text]

        try:
            print(f"Fitting reference model for order {order}...", flush=True)

            # Log memory before model building
            save_memory_snapshot()

            model = vomm.ppm()
            model.fit(self._reference_data, d=order)
            self.reference_models[order] = model
            print(f"Reference model for order {order} built successfully.", flush=True)

            # Log memory after model building
            save_memory_snapshot()

        except Exception as e:
            print(f"Warning: Failed to build reference model for order {order}: {e}", flush=True)

    def build_and_serialize_reference_models(self, orders):
        """Build reference models for multiple orders and return serialized data for worker processes."""
        if not self.reference_text:
            return None, None

        # Build models for all required orders
        for order in orders:
            self._build_reference_model(order)

        # Serialize models for transfer to workers
        try:
            serialized_models = {}
            for order, model in self.reference_models.items():
                # Serialize each model to bytes
                serialized_models[order] = pickle.dumps(model)

            # Also serialize reference data
            serialized_reference_data = pickle.dumps(self._reference_data) if hasattr(self, '_reference_data') else None

            return serialized_models, serialized_reference_data
        except Exception as e:
            print(f"Warning: Failed to serialize reference models: {e}", flush=True)
            return None, None

    def extract_features_from_directory(self, directory: str, output_csv: str, target_words: set[str],
                                        order=1, alphabet_size=256, word_distance_type="mean", batch_size=1000,
                                        exclude_files_csv=None):
        """
        Extract features in memory-safe batches using multiprocessing.
        Each batch starts a new process pool to limit memory growth.
        """
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if file.endswith(".txt")
        ]

        # Exclude reference files if provided
        if exclude_files_csv and os.path.exists(exclude_files_csv):
            exclude_df = pd.read_csv(exclude_files_csv)
            exclude_filenames = set(exclude_df['file_name'])
            original_count = len(all_files)
            all_files = [f for f in all_files if os.path.basename(f) not in exclude_filenames]
            print(f"Excluded {original_count - len(all_files)} reference files", flush=True)

        print(f"Found {len(all_files)} text files in {directory}", flush=True)
        total_batches = math.ceil(len(all_files) / batch_size)
        output_paths = []
        

        for i in range(total_batches):
            results = []
            timeout_files = []
            batch_path = f"{output_csv}_batch_{i:03}.csv"
            if os.path.exists(batch_path):
                print(f"[SKIP] Batch {i+1} already processed ({batch_path})", flush=True)
                output_paths.append(batch_path)
                continue
            
            
            batch = all_files[i * batch_size:(i + 1) * batch_size]
            print(f"\nBatch {i + 1}/{total_batches}: Processing files {i * batch_size}–{i * batch_size + len(batch) - 1}", flush=True)
            

            # Use multiprocessing with worker initialization
            print(f"[INFO] Using multiprocessing with {self.num_threads} threads")

            # Pre-serialize reference models for efficient transfer to workers
            print(f"[INFO] Preparing reference models for worker processes...", flush=True)
            serialized_models, serialized_reference_data = self.build_and_serialize_reference_models([1, order] if order != 1 else [1])

            # Create simplified arguments for worker
            simplified_args = [(file_path, order, alphabet_size, word_distance_type) for file_path in batch]

            print(f"[INFO] Starting ProcessPoolExecutor with {self.num_threads} workers for batch {i+1}", flush=True)

            # Enhanced ProcessPoolExecutor with comprehensive monitoring
            executor_start_time = time.time()

            with ProcessPoolExecutor(
                max_workers=self.num_threads,
                initializer=initialize_worker_with_models,
                initargs=(self.file_info_path, self.language, self.lowercase,
                         target_words, serialized_models, serialized_reference_data, self.use_ncd)
            ) as executor:
                try:
                    print(f"[INFO] ProcessPoolExecutor created successfully. Submitting {len(simplified_args)} tasks...", flush=True)

                    # Submit all tasks with timeout monitoring
                    future_to_args = {}
                    for args in simplified_args:
                        future = executor.submit(file_processor, args)
                        future_to_args[future] = args

                    print(f"[INFO] All tasks submitted. Waiting for completion...", flush=True)

                    # Monitor task completion with timeout
                    results = []
                    timeout_files = []
                    completed_count = 0

                    # Use as_completed with overall timeout

                    try:
                        for future in tqdm(as_completed(future_to_args),
                                         total=len(future_to_args),
                                         desc=f"Processing Batch {i+1}"):
                            args = future_to_args[future]
                            try:
                                # Get result with individual timeout
                                result = future.result()
                                if result is not None:
                                    results.append(result)
                                else:
                                    timeout_files.append(args)
                                    print(f"[WARNING] Task returned None for {args[0]}", flush=True)
                            except TimeoutError:
                                print(f"[TIMEOUT] Task timed out for {args[0]}", flush=True)
                                timeout_files.append(args)
                            except Exception as task_e:
                                print(f"[TASK-ERROR] Task failed for {args[0]}: {task_e}", flush=True)
                                timeout_files.append(args)

                            completed_count += 1

                            # Log progress every 50 files
                            if completed_count % 50 == 0:
                                print(f"[PROGRESS] Completed {completed_count}/{len(simplified_args)} tasks in batch {i+1}", flush=True)

                    except TimeoutError:
                        print(f"[TIMEOUT] Overall batch timeout reached. Cancelling remaining tasks...", flush=True)

                        # Cancel remaining tasks
                        for future in future_to_args:
                            if not future.done():
                                future.cancel()
                                timeout_files.append(future_to_args[future])

                    executor_time = time.time() - executor_start_time
                    print(f"[INFO] ProcessPoolExecutor completed in {executor_time:.1f} seconds. Results: {len(results)}, Timeouts: {len(timeout_files)}", flush=True)

                except Exception as e:
                    executor_time = time.time() - executor_start_time
                    print(f"[MULTIPROCESSING-ERROR] ProcessPoolExecutor failed after {executor_time:.1f} seconds: {e}", flush=True)
                    print(f"[MULTIPROCESSING-ERROR] Error type: {type(e).__name__}", flush=True)
                    traceback.print_exc()

                    # Log system state at multiprocessing failure
                    try:
                        current_process = psutil.Process()
                        main_memory = current_process.memory_info().rss / 1024**3
                        children = current_process.children(recursive=True)
                        child_count = len(children)
                        child_memory = sum(child.memory_info().rss for child in children if child.is_running()) / 1024**3

                        print(f"[MULTIPROCESSING-ERROR] System state at failure:", flush=True)
                        print(f"[MULTIPROCESSING-ERROR] Main process memory: {main_memory:.2f} GB", flush=True)
                        print(f"[MULTIPROCESSING-ERROR] Child processes: {child_count}, using {child_memory:.2f} GB", flush=True)

                        # Try to identify which workers are still alive
                        for i, child in enumerate(children):
                            try:
                                child_mem = child.memory_info().rss / 1024**3
                                print(f"[MULTIPROCESSING-ERROR] Child {i} (PID {child.pid}): {child_mem:.2f} GB, Status: {child.status()}", flush=True)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                print(f"[MULTIPROCESSING-ERROR] Child {i}: No longer accessible", flush=True)

                    except Exception as state_e:
                        print(f"[MULTIPROCESSING-ERROR] Could not get system state: {state_e}", flush=True)
                    # Fall back to sequential processing with proper initialization
                    initialize_worker_with_models(self.file_info_path, self.language, self.lowercase,
                                                 target_words, serialized_models, serialized_reference_data, self.use_ncd)
                    for args in tqdm(simplified_args, desc=f"Processing Batch {i+1}"):
                        try:
                            result = file_processor(args)
                            if result is None:
                                timeout_files.append(args)
                            else:
                                results.append(result)
                        except Exception as e:
                            print(f"Error processing {args[0]}: {e}")
                            timeout_files.append(args)

            if results:
                batch_df = pd.DataFrame(results)
                batch_path = f"{output_csv}_batch_{i:03}.csv"
                batch_df.to_csv(batch_path, index=False)
                output_paths.append(batch_path)
                del batch_df
            else:
                print(f"WARNING: Batch {i+1} produced no results (likely due to OOM or processing failures)", flush=True)

            print(f"[MEMORY] After batch {i+1} memory usage:", flush=True)
            self.log_memory_usage()

            # Log worker memory distribution for multiprocessing analysis
            log_worker_memory_distribution()

            # Detect potential memory leaks
            detect_memory_leaks()

            del results, batch
            gc.collect()

        if not output_paths:
            print("ERROR: No batches produced any results. All processing failed.", flush=True)
            raise RuntimeError("No successful feature extraction results")

        dfs = []
        for p in output_paths:
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    dfs.append(df)
                else:
                    print(f"WARNING: Skipping empty CSV file: {p}", flush=True)
            except pd.errors.EmptyDataError:
                print(f"WARNING: Skipping malformed CSV file: {p}", flush=True)
            except Exception as e:
                print(f"WARNING: Error reading {p}: {e}", flush=True)

        if not dfs:
            print("ERROR: No valid CSV files could be read.", flush=True)
            raise RuntimeError("No valid feature extraction results")

        final_df = pd.concat(dfs, ignore_index=True)
        final_df = final_df.merge(self.file_info, on="file_name", how="left")
        final_df.to_csv(output_csv, index=False)
        
        if timeout_files:
            print(f"\n[INFO] Retrying {len(timeout_files)} timed out files sequentially...", flush=True)
            retried = []

            # Pre-serialize reference models for retry processing
            serialized_models, serialized_reference_data = self.build_and_serialize_reference_models([1, order] if order != 1 else [1])

            # Initialize worker for sequential retry
            initialize_worker_with_models(self.file_info_path, self.language, self.lowercase,
                                        target_words, serialized_models, serialized_reference_data, self.use_ncd)

            for args in tqdm(timeout_files, desc="Retrying timeouts", unit="file"):
                result = file_processor(args)
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
        self.log_memory_usage("Final ")



    def _extract_from_file_bound(self, args):
        text_path, target_words, order, alphabet_size, word_distance_type = args
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()

            filename = os.path.basename(text_path)
            features = self.extract(text, target_words,
                                    order=order,
                                    alphabet_size=alphabet_size,
                                    word_distance_type=word_distance_type,
                                    use_ncd=self.use_ncd)
            features["file_name"] = filename
            return features
        except Exception as e:
            print(f"Error processing {text_path}: {e}", flush=True)
            return None



    def extract(self, text:str, target_words=set[str], order=None, alphabet_size=69, word_distance_type="mean", use_ncd=True):
        """Extracts all features dynamically from a given text."""
        if self.lowercase:
            text = text.lower()

        # Pre-compute commonly used values to avoid redundant processing
        words = text.split()  # Compute once, reuse multiple times
        char_counter = Counter(text)  # Compute once for entropy and other char-based features
        text_length = len(text)  # Cache text length
        word_count = len(words)  # Cache word count

        # Build features dictionary - conditionally include NCD
        features = {
            "Compression_Ratio_Order_1": self.compression_ratio(text, order=1),
            "NRC_Order_1": self.nrc(text, order=1, alphabet_size=alphabet_size),
            "Entropy_Ratio_Order_1": self.entropy_ratio(text, order=1),
            "Shannon_Entropy": self.shannon_entropy(char_counter),
            "Avg_Word_Length": self.average_word_length(words, word_count),
            "Lexical_Richness": self.lexical_richness(words, word_count),
            "Avg_Sentence_Length": self.average_sentence_length(text),
            "Punctuation_Density": self.punctuation_density(text, text_length),
            "Syllable_Per_Word": self.syllable_per_word(words, word_count),
            "Digit_Ratio": self.digit_ratio(text, text_length)
        }

        # Add NCD if enabled
        if use_ncd:
            features["NCD_Order_1"] = self.ncd(text, order=1)

        # Add higher order features if order is specified and not 1
        if order is not None and order != 1:
            higher_order_features = {
                f"Compression_Ratio_Order_{order}": self.compression_ratio(text, order=order),
                f"NRC_Order_{order}": self.nrc(text, order=order, alphabet_size=alphabet_size),
                f"Entropy_Ratio_Order_{order}": self.entropy_ratio(text, order=order),
            }

            # Add higher order NCD if enabled
            if use_ncd:
                higher_order_features[f"NCD_Order_{order}"] = self.ncd(text, order=order)

            features.update(higher_order_features)

        if self.language == "english":
            features["Flesch_Readability"] = self.flesch_readability(text)

        if len(self.stop_words) > 0:
            features["Stopword_Ratio"] = self.stopword_ratio(words, word_count)

        if target_words:
            if word_distance_type == "mean":
                word_distances = self.mean_word_distance(words, target_words)
            elif word_distance_type == "median":
                word_distances = self.median_word_distance(words, target_words)

            for feature_name, value in word_distances.items():
                features[feature_name] = value

        # Add neologism features (no data leakage - no ground truth year used)
        neologism_features = self._detect_neologisms(text)
        features.update(neologism_features)

        return features

    def compression_ratio(self, text:str, order=1):
        """
        Calculates compression ratio using Markov model of specified order.

        :param text: The text to compress
        :param order: Order of the Markov (PPM) model
        :return: Compression ratio (compressed bits / original bits)
        """
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
            print(f"[vomm] Crash in compression_ratio: {e}", flush=True)
            traceback.print_exc()
            return None


    def nrc(self, text:str, order=1, alphabet_size=256):
        """
        Calculates NRC using pre-built reference model if available, otherwise self-reference.
        NRC(x‖y) = C(x‖y) / (|x| * log2 |A|) where y is reference text

        :param text: The text to compress
        :param order: Order of the Markov (PPM) model
        :param alphabet_size: Alphabet size (256 for full ASCII/byte range)
        :return: NRC value (float)
        """
        if not text:
            return None

        data = [ord(c) for c in text]

        # Use pre-built reference model if available, otherwise self-reference
        if order in self.reference_models:
            model = self.reference_models[order]
        else:
            # Fall back to self-reference
            model = vomm.ppm()
            model.fit(data, d=order)

        try:
            compressed_bits = -model.logpdf(data) / math.log(2)
            max_bits = len(data) * math.log2(alphabet_size)
            return compressed_bits / max_bits if max_bits > 0 else None
        except Exception as e:
            print(f"Warning: NRC calculation failed: {e}", flush=True)
            return None

    def entropy_ratio(self, text:str, order=1):
        """Calculates the ratio between Markov-based entropy and Shannon entropy for given order."""
        if not text:
            return 0

        char_counter = Counter(text)
        shannon_entropy = self.shannon_entropy(char_counter)

        markov_entropy = self.compression_ratio(text, order=order) * 8

        return markov_entropy / shannon_entropy if shannon_entropy > 0 else 0

    def shannon_entropy(self, char_counter):
        """Computes Shannon entropy using pre-computed character counter."""
        if not char_counter:
            return 0
        total_chars = sum(char_counter.values())
        return -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counter.values())

    def average_word_length(self, words, word_count=None):
        """Calculates the average word length using pre-split words."""
        if not words:
            return 0
        word_count = word_count if word_count is not None else len(words)
        return sum(len(word) for word in words) / word_count

    def lexical_richness(self, words, word_count=None):
        """Computes lexical richness using pre-split words."""
        if not words:
            return 0
        word_count = word_count if word_count is not None else len(words)
        return len(set(words)) / word_count

    def average_sentence_length(self, text:str):
        """Computes average sentence length in words."""
        sentences = text.split('. ')
        words = [sentence.split() for sentence in sentences]
        return 0 if not words else sum(len(sentence) for sentence in words) / len(words)
    
    def get_stopwords(self, lang="english"):
        """
        Gets stopwords for the specified language.

        :param lang: Language code (e.g., 'english', 'french')
        :return: Set of stopwords for the language
        """
        try:
            return set(stopwords.words(lang))
        except OSError:
            print(f"Stopwords not available for language: {lang}, defaulting to empty set.", flush=True)
            return set()

    def punctuation_density(self, text:str, text_length=None):
        """Calculates the ratio of punctuation characters to total characters using cached length."""
        if not text:
            return 0
        text_length = text_length if text_length is not None else len(text)
        return sum(1 for char in text if char in string.punctuation) / text_length

    def syllable_per_word(self, words, word_count=None):
        """Calculates the average number of syllables per word using pre-split words."""
        if not words:
            return 0
        word_count = word_count if word_count is not None else len(words)
        return sum(textstat.syllable_count(word) for word in words) / word_count

    # def uppercase_ratio(self, text:str):
    #     """Calculates the ratio of uppercase letters to total letters."""
    #     total_letters = sum(1 for char in text if char.isalpha())
    #     return 0 if total_letters == 0 else sum(1 for char in text if char.isupper()) / total_letters


    def digit_ratio(self, text:str, text_length=None):
        """Calculates the ratio of digits to total characters using cached length."""
        if not text:
            return 0
        text_length = text_length if text_length is not None else len(text)
        return sum(1 for char in text if char.isdigit()) / text_length

    # def special_character_ratio(self, text:str, text_length=None):
    #     """[DEPRECATED] Calculates the ratio of special characters using cached length."""
    #     if not text:
    #         return 0
    #     text_length = text_length if text_length is not None else len(text)
    #     return sum(1 for char in text if not char.isalnum() and char not in string.punctuation and not char.isspace()) / text_length

    def flesch_readability(self, text:str):
        """Calculates Flesch reading ease score (English only)."""
        try:
            return textstat.flesch_reading_ease(text)
        except Exception as e:
            print(f"Warning: Flesch readability calculation failed: {e}", flush=True)
            return None

    def stopword_ratio(self, words, word_count=None):
        """Calculates the ratio of stopwords to total words using pre-split words."""
        if not words:
            return 0
        word_count = word_count if word_count is not None else len(words)
        return sum(1 for word in words if word.lower() in self.stop_words) / word_count

    def ncd(self, text:str, order=1):
        """
        Calculates Normalized Compression Distance (NCD) between text and reference.
        NCD(x,y) = (C(xy) - min(C(x),C(y))) / max(C(x),C(y))

        :param text: The text to compare
        :param order: Order of the Markov (PPM) model
        :return: NCD value between 0 and 1, or None if no reference available
        """
        if not text or (not self.reference_text and not hasattr(self, '_reference_data')):
            return None

        try:
            # Prepare text data
            text_data = text.lower() if self.lowercase else text
            text_bytes = [ord(c) for c in text_data]

            # Use pre-computed reference data if available, otherwise compute from reference_text
            if hasattr(self, '_reference_data') and self._reference_data is not None:
                ref_bytes = self._reference_data
                # Convert reference data back to string for concatenation
                ref_chars = [chr(c) for c in ref_bytes]
                ref_data = ''.join(ref_chars)
            else:
                ref_data = self.reference_text.lower() if self.lowercase else self.reference_text
                ref_bytes = [ord(c) for c in ref_data]

            # Convert to byte arrays for concatenation
            concat_bytes = [ord(c) for c in text_data + ref_data]

            # Build text model and calculate compressed size
            model_text = vomm.ppm()
            model_text.fit(text_bytes, d=order)
            c_text = -model_text.logpdf(text_bytes) / math.log(2)

            # Use pre-built reference model if available (like NRC does)
            if order in self.reference_models:
                model_ref = self.reference_models[order]
                c_ref = -model_ref.logpdf(ref_bytes) / math.log(2)
            else:
                # Fall back to building reference model
                model_ref = vomm.ppm()
                model_ref.fit(ref_bytes, d=order)
                c_ref = -model_ref.logpdf(ref_bytes) / math.log(2)

            # Build concatenated model (must be unique for each file)
            model_concat = vomm.ppm()
            model_concat.fit(concat_bytes, d=order)
            c_concat = -model_concat.logpdf(concat_bytes) / math.log(2)

            # Calculate NCD
            min_c = min(c_text, c_ref)
            max_c = max(c_text, c_ref)

            if max_c == 0:
                return 0

            ncd_value = (c_concat - min_c) / max_c
            return max(0, min(1, ncd_value))  # Ensure value is between 0 and 1

        except Exception as e:
            print(f"Warning: NCD calculation failed: {e}", flush=True)
            return None

    def mean_word_distance(self, words, target_words:list[str]):
        """Computes the mean distance using pre-split words."""
        word_positions = defaultdict(list)

        # Convert words to lowercase for comparison if not already done
        words_lower = [word.lower() for word in words] if not self.lowercase else words

        for index, word in enumerate(words_lower):
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
    

    def median_word_distance(self, words, target_words:list[str]):
        """Computes the median distance using pre-split words."""
        word_positions = defaultdict(list)

        # Convert words to lowercase for comparison if not already done
        words_lower = [word.lower() for word in words] if not self.lowercase else words

        for index, word in enumerate(words_lower):
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

    def create_reference_file(self, source_directory: str, reference_file_path: str,
                             used_files_csv: str, percentage: float = 0.05):
        """
        Creates a reference file by concatenating a percentage of files from source directory.
        Saves list of used files to a CSV for exclusion from main processing.
        Uses caching - if reference file already exists, loads it instead of recreating.

        Args:
            source_directory: Directory containing source text files
            reference_file_path: Path to save the reference text file
            used_files_csv: Path to save CSV listing files used for reference
            percentage: Percentage of files to use (default 0.05 = 5%)
        """
        # Check if reference file already exists
        if os.path.exists(reference_file_path) and os.path.exists(used_files_csv):
            print(f"Reference file already exists, loading: {reference_file_path}", flush=True)
            try:
                with open(reference_file_path, 'r', encoding='utf-8') as f:
                    reference_text = f.read()
                print(f"Loaded existing reference text: {len(reference_text):,} characters", flush=True)
                return reference_text
            except Exception as e:
                print(f"Error loading existing reference file: {e}. Creating new one...", flush=True)


        # Get all text files
        all_files = []
        for root, _, files in os.walk(source_directory):
            for file in files:
                if file.endswith(".txt"):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            print(f"No text files found in {source_directory}")
            return

        # Try to create decade-balanced sample using metadata
        metadata_dict = dict(zip(self.file_info['file_name'], self.file_info['decade']))

        # Group files by decade
        files_by_decade = defaultdict(list)
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            if file_name in metadata_dict:
                decade = metadata_dict[file_name]
                files_by_decade[decade].append(file_path)

        if files_by_decade:
            print(f"Creating DECADE-BALANCED reference from {percentage*100:.1f}% of files in each decade:", flush=True)
            for decade, files in files_by_decade.items():
                print(f"  {decade}s: {len(files)} files available")

            # Sample percentage from each decade
            selected_files = []
            random.seed(42)  # For reproducibility

            for decade, files in files_by_decade.items():
                num_to_select = max(1, int(len(files) * percentage))
                decade_selected = random.sample(files, min(num_to_select, len(files)))
                selected_files.extend(decade_selected)
                print(f"  Selected {len(decade_selected)} files from {decade}s ({percentage*100:.1f}%)")

            print(f"Total: {len(selected_files)} files selected (decade-balanced)", flush=True)
        else:
            print(f"[WARNING] Could not create decade-balanced sample, using random sampling", flush=True)
            # Fallback to random sampling if metadata grouping fails
            num_files = max(1, int(len(all_files) * percentage))
            random.seed(42)  # For reproducibility
            selected_files = random.sample(all_files, num_files)
            print(f"Creating reference from {num_files} files ({percentage*100:.1f}% of {len(all_files)} total files)", flush=True)

        # Concatenate selected files
        reference_text = ""
        used_files_data = []

        for file_path in tqdm(selected_files, desc="Creating reference file"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if self.lowercase:
                        text = text.lower()
                    reference_text += text + "\n"

                used_files_data.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path)
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}", flush=True)

        # Save reference file
        with open(reference_file_path, 'w', encoding='utf-8') as f:
            f.write(reference_text)

        # Save used files list
        used_files_df = pd.DataFrame(used_files_data)
        used_files_df.to_csv(used_files_csv, index=False)

        print(f"Reference file saved to: {reference_file_path}", flush=True)
        print(f"Used files list saved to: {used_files_csv}", flush=True)
        print(f"Reference text length: {len(reference_text):,} characters", flush=True)

        return reference_text

    def _load_neologism_data(self):
        """Load neologism words and periods from target_words.json."""
        try:
            # Try to load from the same directory as target_words.json
            target_words_path = os.path.join(os.path.dirname(__file__), "target_words.json")
            if not os.path.exists(target_words_path):
                # Fallback: look in current directory
                target_words_path = "target_words.json"

            with open(target_words_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "neologisms" in data:
                self.neologism_words = data["neologisms"]
                # Convert to sets for faster lookup
                for period in self.neologism_words:
                    words = self.neologism_words[period]
                    if self.lowercase:
                        words = [w.lower() for w in words]
                    self.neologism_words[period] = set(words)

            if "neologism_periods" in data:
                self.neologism_periods = data["neologism_periods"]

        except Exception as e:
            print(f"[WARNING] Could not load neologism data: {e}", flush=True)
            self.neologism_words = {}
            self.neologism_periods = {}

    def _detect_neologisms(self, text):
        """
        Detect neologism presence in text without using ground truth year (no data leakage).

        Args:
            text: Text to analyze

        Returns:
            Dictionary of neologism features
        """
        if not self.neologism_words or not self.neologism_periods:
            # Return empty features if no neologism data loaded
            return {f"contains_{period}_words": False for period in self.neologism_periods.keys()}

        # Preprocess text for word matching
        if self.lowercase:
            text = text.lower()

        # Extract words, handling hyphenated compounds and contractions
        import re
        words = re.findall(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b", text)
        word_set = set(words)

        features = {}

        # Check each period for word presence (no anachronism calculation)
        for period_name in self.neologism_periods.keys():
            period_words = self.neologism_words.get(period_name, set())
            found_words = []

            for neologism in period_words:
                # Handle multi-word phrases
                if ' ' in neologism:
                    phrase_pattern = neologism.replace(' ', r'\s+')
                    if re.search(r'\b' + phrase_pattern + r'\b', text, re.IGNORECASE):
                        found_words.append(neologism)
                elif neologism in word_set:
                    found_words.append(neologism)

            # Create boolean presence feature for this period
            features[f"contains_{period_name}_words"] = len(found_words) > 0

        # Add valid aggregate features (no ground truth dependency)
        # These represent vocabulary modernity without using claimed year
        modern_periods = ["late_modern", "digital_dawn", "digital_native"]
        historical_periods = ["early_modern", "industrial", "late_industrial"]

        features["contains_modern_vocabulary"] = any(features[f"contains_{period}_words"] for period in modern_periods)
        features["contains_historical_vocabulary"] = any(features[f"contains_{period}_words"] for period in historical_periods)
        features["vocabulary_modernity_score"] = sum(features[f"contains_{period}_words"] for period in modern_periods) / max(1, len(modern_periods))

        return features

    def load_target_words(self, path=None):
        if not path:
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(w.strip().lower() for w in f.readlines() if w.strip())

    def log_memory_usage(self, prefix=""):
        """Log current memory usage."""
        try:
            process = psutil.Process(os.getpid())
            main_memory = process.memory_info().rss / 1024**3

            children = process.children(recursive=True)
            if children:
                print(f"[MEMORY] {prefix}Main: {main_memory:.2f} GB", flush=True)
                for i, child in enumerate(children):
                    try:
                        child_memory = child.memory_info().rss / 1024**3
                        print(f"[MEMORY] {prefix}Worker {i+1}: {child_memory:.2f} GB", flush=True)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        print(f"[MEMORY] {prefix}Worker {i+1}: Finished", flush=True)
            else:
                print(f"[MEMORY] {prefix}Current memory usage: {main_memory:.2f} GB", flush=True)
        except Exception as e:
            print(f"[MEMORY] {prefix}Unable to get memory info: {e}", flush=True)
        
def start_memory_logger(interval=60):
    """Print current memory usage every `interval` seconds."""
    def log_memory():
        while True:
            try:
                process = psutil.Process(os.getpid())
                main_memory = process.memory_info().rss / 1024**3

                children = process.children(recursive=True)
                if children:
                    print(f"[MEMORY] Main: {main_memory:.2f} GB", flush=True)
                    for i, child in enumerate(children):
                        try:
                            child_memory = child.memory_info().rss / 1024**3
                            print(f"[MEMORY] Worker {i+1}: {child_memory:.2f} GB", flush=True)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            print(f"[MEMORY] Worker {i+1}: Finished", flush=True)
                else:
                    print(f"[MEMORY] Current memory usage: {main_memory:.2f} GB", flush=True)
            except Exception as e:
                print(f"[MEMORY] Unable to get memory info: {e}", flush=True)
            time.sleep(interval)

    threading.Thread(target=log_memory, daemon=True).start()
    print(f"[MEMORY] Started memory logger with {interval}s interval", flush=True)


def log_worker_memory_distribution():
    """Log current memory usage."""
    try:
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        print(f"[MEMORY] Current memory usage: {memory_gb:.2f} GB", flush=True)
    except Exception as e:
        print(f"[MEMORY] Unable to get memory info: {e}", flush=True)


def detect_memory_leaks(history_size=10):
    """Log current memory usage."""
    try:
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        print(f"[MEMORY] Current memory usage: {memory_gb:.2f} GB", flush=True)
    except Exception as e:
        print(f"[MEMORY] Unable to get memory info: {e}", flush=True)


def save_memory_snapshot(filename_prefix="memory_snapshot"):
    """Log current memory usage."""
    try:
        memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        print(f"[MEMORY] Current memory usage: {memory_gb:.2f} GB", flush=True)
    except Exception as e:
        print(f"[MEMORY] Unable to get memory info: {e}", flush=True)
    
def check_system_limits():
    """Check and log system resource limits."""
    try:
        # Check file descriptor limits
        soft_fd, hard_fd = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[DEBUG] File descriptor limits: soft={soft_fd}, hard={hard_fd}", flush=True)

        # Check memory limits
        try:
            soft_mem, hard_mem = resource.getrlimit(resource.RLIMIT_AS)
            print(f"[DEBUG] Virtual memory limits: soft={soft_mem/(1024**3):.1f}GB, hard={hard_mem/(1024**3):.1f}GB" if hard_mem != resource.RLIM_INFINITY else "[DEBUG] Virtual memory limits: unlimited", flush=True)
        except (ValueError, OverflowError):
            print(f"[DEBUG] Virtual memory limits: unlimited", flush=True)

        # Check current file descriptor usage
        process = psutil.Process()
        num_fds = process.num_fds()
        print(f"[DEBUG] Current file descriptors in use: {num_fds}", flush=True)

        return True
    except Exception as e:
        print(f"[DEBUG] Error checking system limits: {e}", flush=True)
        return False

def setup_worker_error_handling():
    """Set up comprehensive error handling for worker processes."""
    def signal_handler(signum, frame):
        print(f"[WORKER-ERROR] Worker process received signal {signum}: {signal.Signals(signum).name}", flush=True)
        print(f"[WORKER-ERROR] Worker PID: {os.getpid()}, Frame: {frame.f_code.co_filename}:{frame.f_lineno}", flush=True)

        # Log memory state at termination
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            print(f"[WORKER-ERROR] Worker memory at termination: {memory_gb:.2f} GB", flush=True)
        except:
            pass

        sys.exit(1)

    # Register signal handlers for common termination signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)
    if hasattr(signal, 'SIGKILL'):  # Note: SIGKILL cannot be caught, but we'll try
        try:
            signal.signal(signal.SIGKILL, signal_handler)
        except OSError:
            pass  # SIGKILL cannot be caught

    # Set up logging to capture warnings
    logging.basicConfig(level=logging.WARNING, format='[WORKER-%(levelname)s] %(message)s')

    # Capture Python warnings
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        print(f"[WORKER-WARNING] {category.__name__}: {message} at {filename}:{lineno}", flush=True)

    warnings.showwarning = warning_handler

def verify_pickle_data(serialized_models, serialized_reference_data):
    """Verify that pickled data can be safely deserialized."""
    try:
        if serialized_models:
            for order, serialized_model in serialized_models.items():
                if not isinstance(serialized_model, bytes):
                    raise TypeError(f"Model for order {order} is not bytes: {type(serialized_model)}")

        if serialized_reference_data:
            if not isinstance(serialized_reference_data, bytes):
                raise TypeError(f"Reference data is not bytes: {type(serialized_reference_data)}")

        return True

    except Exception as e:
        print(f"[ERROR] Pickle data verification failed: {e}", flush=True)
        return False

def monitor_worker_initialization(timeout_seconds=60):
    """Monitor worker initialization with timeout."""
    start_time = time.time()
    pid = os.getpid()

    def timeout_handler(signum, frame):
        elapsed = time.time() - start_time
        print(f"[WORKER-TIMEOUT] Worker {pid} initialization timed out after {elapsed:.1f} seconds", flush=True)
        print(f"[WORKER-TIMEOUT] Current frame: {frame.f_code.co_filename}:{frame.f_lineno}", flush=True)

        # Log memory state at timeout
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            print(f"[WORKER-TIMEOUT] Worker memory at timeout: {memory_gb:.2f} GB", flush=True)
        except:
            pass

        sys.exit(1)

    # Set up timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    return start_time

def clear_worker_timeout():
    """Clear the worker initialization timeout alarm."""
    signal.alarm(0)

def initialize_worker_with_models(file_info_path, language, lowercase, target_words, serialized_models, serialized_reference_data, use_ncd=True):
    """
    Initialize worker process with pre-built serialized models to avoid expensive rebuilding.
    Enhanced with comprehensive error handling and debugging.
    """
    global _global_extractor, _global_target_words, _global_reference_models, _global_reference_data

    # Set up error handling first
    setup_worker_error_handling()

    # Monitor initialization with timeout
    start_time = monitor_worker_initialization(timeout_seconds=120)  # 2 minutes timeout

    try:
        pid = os.getpid()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3

        # Verify pickle data integrity
        if not verify_pickle_data(serialized_models, serialized_reference_data):
            raise ValueError("Pickle data verification failed")

        # Deserialize reference models and data
        reference_models = {}
        reference_data = None

        if serialized_models:
            for order, serialized_model in serialized_models.items():
                reference_models[order] = pickle.loads(serialized_model)

        if serialized_reference_data:
            reference_data = pickle.loads(serialized_reference_data)

        # Create extractor with pre-built models
        _global_extractor = TextFeatureExtractor(
            file_info_path=file_info_path,
            lang=language,
            num_threads=1,  # Single thread per worker
            lowercase=lowercase,
            reference_text=None,  # Don't pass reference_text to avoid model rebuilding
            prebuilt_models=reference_models,
            prebuilt_reference_data=reference_data,
            use_ncd=use_ncd
        )

        # Set global variables
        _global_target_words = target_words
        _global_reference_models = reference_models
        _global_reference_data = reference_data

        # Clear timeout
        clear_worker_timeout()

        return True

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[WORKER-ERROR] Worker {os.getpid()} initialization failed after {elapsed_time:.1f} seconds: {e}", flush=True)
        print(f"[WORKER-ERROR] Error type: {type(e).__name__}", flush=True)
        traceback.print_exc()

        # Log system state at failure
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            num_fds = process.num_fds()
            print(f"[WORKER-ERROR] Worker state at failure: {memory_gb:.2f} GB memory, {num_fds} file descriptors", flush=True)
        except:
            print(f"[WORKER-ERROR] Unable to get worker state at failure", flush=True)

        # Clear timeout before re-raising
        clear_worker_timeout()
        raise

# Legacy function for backward compatibility
def initialize_worker(file_info_path, language, reference_text, lowercase, order, target_words):
    """Legacy worker initialization - kept for backward compatibility."""
    global _global_extractor, _global_target_words
    try:
        _global_extractor = TextFeatureExtractor(
            file_info_path=file_info_path,
            lang=language,
            num_threads=1,  # Single thread per worker
            lowercase=lowercase,
            reference_text=reference_text,
            order=order
        )
        _global_target_words = target_words
        print(f"[INFO] Worker initialized successfully (legacy)", flush=True)
    except Exception as e:
        print(f"[ERROR] Worker initialization failed: {e}", flush=True)
        traceback.print_exc()
        raise


def file_processor(args):
    """Worker function that reuses global extractor instance with enhanced error handling."""
    global _global_extractor, _global_target_words

    file_path, order, alphabet_size, word_distance_type = args
    pid = os.getpid()

    try:
        # Check if worker is properly initialized
        if _global_extractor is None:
            error_msg = f"[WORKER-ERROR] Worker {pid} not properly initialized - global extractor is None"
            print(error_msg, flush=True)
            return None

        if _global_target_words is None:
            print(f"[WORKER-WARNING] Worker {pid} has no target words", flush=True)


        # Monitor memory before processing
        try:
            process = psutil.Process()
            pre_memory = process.memory_info().rss / 1024**3
            num_fds = process.num_fds()


            if num_fds > 500:  # Warn if too many file descriptors
                print(f"[WORKER-WARNING] Worker {pid} high FD usage: {num_fds} before processing {os.path.basename(file_path)}", flush=True)
        except Exception as mem_e:
            print(f"[WORKER-WARNING] Could not check pre-processing memory: {mem_e}", flush=True)
            pre_memory = None

        # Read file with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()

            if not text:
                print(f"[WORKER-WARNING] Empty file: {file_path}", flush=True)
                return None

            text_length = len(text)
            if text_length > 10**7:  # Warn for very large files (>10MB)
                print(f"[WORKER-WARNING] Large file: {os.path.basename(file_path)} ({text_length:,} characters)", flush=True)

        except Exception as read_e:
            print(f"[WORKER-ERROR] Failed to read file {file_path}: {read_e}", flush=True)
            return None

        # Extract features with timeout and error handling
        try:
            filename = os.path.basename(file_path)
            features = _global_extractor.extract(
                text,
                target_words=_global_target_words,
                order=order,
                alphabet_size=alphabet_size,
                word_distance_type=word_distance_type,
                use_ncd=_global_extractor.use_ncd
            )

            if not features:
                print(f"[WORKER-WARNING] No features extracted from {file_path}", flush=True)
                return None

            features["file_name"] = filename

        except Exception as extract_e:
            print(f"[WORKER-ERROR] Feature extraction failed for {file_path}: {extract_e}", flush=True)
            print(f"[WORKER-ERROR] Error type: {type(extract_e).__name__}", flush=True)
            traceback.print_exc()
            return None

        # Monitor memory after processing
        try:
            post_memory = process.memory_info().rss / 1024**3
            if pre_memory is not None:
                memory_delta = post_memory - pre_memory
                if abs(memory_delta) > 0.5:  # Log significant memory changes
                    print(f"[WORKER-MEMORY] Worker {pid} memory change: {memory_delta:+.2f} GB after processing {os.path.basename(file_path)}", flush=True)
        except Exception as post_mem_e:
            print(f"[WORKER-WARNING] Could not check post-processing memory: {post_mem_e}", flush=True)

        # Explicit cleanup to reduce memory footprint
        del text

        return features

    except Exception as e:
        print(f"[WORKER-ERROR] Unexpected error in worker {pid} processing {file_path}: {e}", flush=True)
        print(f"[WORKER-ERROR] Error type: {type(e).__name__}", flush=True)
        traceback.print_exc()

        # Log worker state at error
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            num_fds = process.num_fds()
            print(f"[WORKER-ERROR] Worker {pid} state at error: {memory_gb:.2f} GB memory, {num_fds} file descriptors", flush=True)
        except:
            print(f"[WORKER-ERROR] Unable to get worker {pid} state at error", flush=True)

        return None


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
    parser.add_argument("--gutenberg_info", required=False, help="Path to gutenberg metadata CSV file (if different from main file_info)")
    parser.add_argument("--target_words", required=True, help="Path to target_words.json")
    parser.add_argument("--lang", default="english", help="Language for stopword filtering and word lists")
    parser.add_argument("--order", type=int, default=1, help="Markov model order")
    parser.add_argument("--word_distance", choices=["mean", "median"], default="mean", help="Distance type for target words")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Number of files to process in each chunk")
    parser.add_argument("--threads", type=int, default=8, help="Number of parallel threads for training/validation phases")
    parser.add_argument("--inference_threads", type=int, default=None, help="Number of parallel threads for test/gutenberg phases (defaults to --threads value if not specified)")
    parser.add_argument("--lowercase", action="store_true", help="Convert all text to lowercase before processing")
    parser.add_argument("--create_references", action="store_true", help="Create reference files from 5% of train/validation datasets")
    parser.add_argument("--reference_percentage", type=float, default=0.05, help="Percentage of files to use for reference")
    parser.add_argument("--validation_reference_percentage", type=float, default=0.05, help="Percentage of validation files to use as reference for test/gutenberg")
    parser.add_argument("--no-ncd", action="store_true", help="Disable NCD (Normalized Compression Distance) features for faster processing")
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    args = parse_args()

    # Set default inference_threads to threads if not specified
    if args.inference_threads is None:
        args.inference_threads = args.threads

    print(f"[THREADING] Training/Validation phases will use {args.threads} threads", flush=True)
    print(f"[THREADING] Test/Gutenberg phases will use {args.inference_threads} threads", flush=True)

    # Start enhanced memory logging with more frequent updates during feature extraction
    start_memory_logger(interval=10)

    print("[MEMORY] === INITIAL MEMORY STATE ===", flush=True)
    # Create a temporary extractor just to log initial memory state
    temp_extractor = TextFeatureExtractor(args.file_info, args.lang, 1, args.lowercase, order=args.order)
    temp_extractor.log_memory_usage("Initial system state - ")
    del temp_extractor
    gc.collect()

    if args.target_words:
        with open(args.target_words, "r", encoding="utf-8") as f:
            word_config = json.load(f)
        target_words = set(word_config.get(args.lang, []))
    else:
        target_words = set()

    # Create reference files if requested
    reference_files = {}
    if args.create_references:
        print("[INFO] Creating reference files...", flush=True)

        temp_extractor = TextFeatureExtractor(
            file_info_path=args.file_info,
            lang=args.lang,
            num_threads=args.threads,
            lowercase=args.lowercase,
            order=args.order
        )

        if args.train:
            train_ref_text = temp_extractor.create_reference_file(
                args.train,
                f"{args.train}_reference_balanced.txt",
                f"{args.train}_reference_files.csv",
                args.reference_percentage
            )
            reference_files['train'] = train_ref_text

        # Only create validation reference if we're actually processing validation data
        # If validation dir is provided only for test/gutenberg reference, skip this
        if args.valid and args.valid_out:
            valid_ref_text = temp_extractor.create_reference_file(
                args.valid,
                f"{args.valid}_reference_balanced.txt",
                f"{args.valid}_reference_files.csv",
                args.reference_percentage
            )
            reference_files['valid'] = valid_ref_text

    # Load reference texts for processing
    train_reference = reference_files.get('train', None)
    valid_reference = reference_files.get('valid', None)


    def load_validation_reference(validation_dir, reference_percentage=0.05):
        print(f"[INFO] Loading {reference_percentage*100:.1f}% of validation dataset as DECADE-BALANCED reference for test/gutenberg...", flush=True)
        # Create cached reference from percentage of validation dataset (decade-balanced)
        validation_reference_path = f"{validation_dir}_reference_balanced_{reference_percentage*100:.1f}pct.txt"

        if os.path.exists(validation_reference_path):
            print(f"Loading cached decade-balanced validation reference: {validation_reference_path}", flush=True)
            try:
                with open(validation_reference_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error loading cached reference: {e}. Creating new one...", flush=True)

        if not os.path.exists(validation_dir):
            print(f"[ERROR] Validation directory {validation_dir} doesn't exist", flush=True)
            return None

        print("Creating decade-balanced validation reference from validation files...", flush=True)

        # Get all validation files
        all_valid_files = []
        for root, _, files in os.walk(validation_dir):
            for file in files:
                if file.endswith(".txt"):
                    all_valid_files.append(os.path.join(root, file))

        if not all_valid_files:
            print(f"[ERROR] No .txt files found in {validation_dir}", flush=True)
            return None

        print(f"Found {len(all_valid_files)} validation files", flush=True)

        # Try to load file info to get decade information
        # Look for metadata file in parent directory structure
        metadata_file = None
        search_dir = validation_dir
        for _ in range(3):  # Search up to 3 levels up
            potential_metadata = os.path.join(search_dir, "balanced_dataset_metadata.csv")
            if os.path.exists(potential_metadata):
                metadata_file = potential_metadata
                break
            search_dir = os.path.dirname(search_dir)

        if not metadata_file:
            print("[WARNING] Could not find metadata file, falling back to random sampling")
            # Fallback to original random sampling if no metadata found
            import random
            random.seed(42)  # For reproducible results
            num_files_to_use = max(1, int(len(all_valid_files) * reference_percentage))
            selected_files = random.sample(all_valid_files, num_files_to_use)
        else:
            # Load metadata and create decade-balanced sample
            print(f"Loading metadata from: {metadata_file}", flush=True)
            try:
                metadata_df = pd.read_csv(metadata_file)
                if 'file_name' not in metadata_df.columns or 'decade' not in metadata_df.columns:
                    raise ValueError("Metadata file must contain 'file_name' and 'decade' columns")

                # Group files by decade
                files_by_decade = defaultdict(list)
                metadata_dict = dict(zip(metadata_df['file_name'], metadata_df['decade']))

                for file_path in all_valid_files:
                    file_name = os.path.basename(file_path)
                    if file_name in metadata_dict:
                        decade = metadata_dict[file_name]
                        files_by_decade[decade].append(file_path)

                print(f"Files grouped by decade:")
                for decade, files in files_by_decade.items():
                    print(f"  {decade}s: {len(files)} files")

                # Sample reference_percentage from each decade
                selected_files = []
                import random
                random.seed(42)  # For reproducible results

                for decade, files in files_by_decade.items():
                    num_to_select = max(1, int(len(files) * reference_percentage))
                    decade_selected = random.sample(files, min(num_to_select, len(files)))
                    selected_files.extend(decade_selected)
                    print(f"  Selected {len(decade_selected)} files from {decade}s ({reference_percentage*100:.1f}%)")

            except Exception as e:
                print(f"[WARNING] Error processing metadata file: {e}, falling back to random sampling")
                import random
                random.seed(42)
                num_files_to_use = max(1, int(len(all_valid_files) * reference_percentage))
                selected_files = random.sample(all_valid_files, num_files_to_use)

        print(f"Selected {len(selected_files)} out of {len(all_valid_files)} validation files for decade-balanced reference", flush=True)

        valid_full_text = ""
        for file_path in tqdm(selected_files, desc="Loading decade-balanced validation reference"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if args.lowercase:
                        text = text.lower()
                    valid_full_text += text + "\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}", flush=True)

        # Save the reference for future use
        print(f"Saving decade-balanced validation reference to: {validation_reference_path}", flush=True)
        with open(validation_reference_path, 'w', encoding='utf-8') as f:
            f.write(valid_full_text)
        print(f"Decade-balanced validation reference text length: {len(valid_full_text):,} characters", flush=True)
        return valid_full_text

    # Initialize as None - will only be loaded when actually needed
    test_reference = None
    gutenberg_reference = None

    # Define validation reference path for checking existence
    validation_reference_path = f"{args.valid}_reference_{args.validation_reference_percentage*100:.1f}pct.txt" if args.valid else f"../Dataset/Sorted_texts_trimmed/validation_reference_{args.validation_reference_percentage*100:.1f}pct.txt"

    # Compute actual alphabet size based on dataset and lowercase mode
    print("[INFO] Computing global alphabet size from datasets...", flush=True)
    temp_extractor = TextFeatureExtractor(args.file_info, args.lang, 1, args.lowercase)

    # Collect all available directories for alphabet size computation
    directories_to_scan = []
    if args.train:
        directories_to_scan.append(args.train)
    if args.valid:
        directories_to_scan.append(args.valid)
    if args.test:
        directories_to_scan.append(args.test)

    if directories_to_scan:
        global_alphabet_size = temp_extractor.compute_global_alphabet_size(directories_to_scan)
    else:
        # Fallback to hardcoded value if no directories available
        global_alphabet_size = 43 if args.lowercase else 69
        print(f"[WARNING] No directories available for scanning, using fallback alphabet size: {global_alphabet_size}", flush=True)

    print(f"Using computed alphabet size: {global_alphabet_size} (lowercase mode: {args.lowercase})", flush=True)
    del temp_extractor

    if args.train:
        print(f"[TRAIN] Creating training extractor with {args.threads} threads", flush=True)
        train_extractor = TextFeatureExtractor(
            file_info_path=args.file_info,
            lang=args.lang,
            num_threads=args.threads,
            lowercase=args.lowercase,
            reference_text=train_reference,
            order=args.order,
            use_ncd=not getattr(args, 'no_ncd', False)
        )
        exclude_csv = f"{args.train}_reference_files.csv" if args.create_references else None
        train_extractor.extract_features_from_directory(
            args.train, args.train_out, target_words, args.order,
            global_alphabet_size, args.word_distance, args.chunk_size, exclude_csv
        )

    if args.valid and args.valid_out:
        print(f"[VALID] Creating validation extractor with {args.threads} threads", flush=True)
        valid_extractor = TextFeatureExtractor(
            file_info_path=args.file_info,
            lang=args.lang,
            num_threads=args.threads,
            lowercase=args.lowercase,
            reference_text=valid_reference,
            order=args.order,
            use_ncd=not getattr(args, 'no_ncd', False)
        )
        exclude_csv = f"{args.valid}_reference_files.csv" if args.create_references else None
        valid_extractor.extract_features_from_directory(
            args.valid, args.valid_out, target_words, args.order,
            global_alphabet_size, args.word_distance, args.chunk_size, exclude_csv
        )

    if args.test:
        # Load validation reference if available (either from current run or previous run)
        validation_dir = args.valid
        validation_reference_path = f"{validation_dir}_reference_{args.validation_reference_percentage*100:.1f}pct.txt"
        if args.valid or os.path.exists(validation_reference_path):
            print("[INFO] Loading validation reference for test processing...", flush=True)
            test_reference = load_validation_reference(validation_dir, args.validation_reference_percentage)
        else:
            print("[WARNING] No validation reference available for test processing", flush=True)
            test_reference = None
        print(f"[TEST] Creating test extractor with {args.inference_threads} threads (inference phase)", flush=True)
        test_extractor = TextFeatureExtractor(
            file_info_path=args.file_info,
            lang=args.lang,
            num_threads=args.inference_threads,
            lowercase=args.lowercase,
            reference_text=test_reference,
            order=args.order,
            use_ncd=not getattr(args, 'no_ncd', False)
        )
        test_extractor.extract_features_from_directory(
            args.test, args.test_out, target_words, args.order,
            global_alphabet_size, args.word_distance, args.chunk_size
        )

    if args.gutenberg:
        # Load validation reference if available (either from current run or previous run)
        validation_dir = args.valid
        validation_reference_path = f"{validation_dir}_reference_{args.validation_reference_percentage*100:.1f}pct.txt"
        if args.valid or os.path.exists(validation_reference_path):
            print("[INFO] Loading validation reference for gutenberg processing...", flush=True)
            gutenberg_reference = load_validation_reference(validation_dir, args.validation_reference_percentage)
        else:
            print("[WARNING] No validation reference available for gutenberg processing", flush=True)
            gutenberg_reference = None
        # Use separate gutenberg metadata file if provided, otherwise use main file_info
        gutenberg_info_path = args.gutenberg_info if args.gutenberg_info else args.file_info
        print(f"[GUTENBERG] Creating gutenberg extractor with {args.inference_threads} threads (inference phase)", flush=True)
        gutenberg_extractor = TextFeatureExtractor(
            file_info_path=gutenberg_info_path,
            lang=args.lang,
            num_threads=args.inference_threads,
            lowercase=args.lowercase,
            reference_text=gutenberg_reference,
            order=args.order,
            use_ncd=not getattr(args, 'no_ncd', False)
        )
        gutenberg_extractor.extract_features_from_directory(
            args.gutenberg, args.gutenberg_out, target_words, args.order,
            global_alphabet_size, args.word_distance, args.chunk_size
        )

    print("[INFO] Feature extraction completed for all datasets...", flush=True)

    # Final memory state logging
    print("[MEMORY] === FINAL MEMORY STATE ===", flush=True)
    if 'temp_extractor' in locals():
        temp_extractor = locals()['temp_extractor']
    else:
        temp_extractor = TextFeatureExtractor(args.file_info, args.lang, 1, args.lowercase, order=args.order)
    temp_extractor.log_memory_usage("Final processing state - ")
    save_memory_snapshot()
    del temp_extractor
    gc.collect()

    print("[INFO] All done!", flush=True)
