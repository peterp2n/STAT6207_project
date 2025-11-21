"""
Automatic Keyword Generation for Amazon Book Database using TF-IDF

This script generates high-precision keywords for each book based on:
- Product name, description, best sellers rank, features, author, reading age
- TF-IDF algorithm to extract the most representative and distinctive terms
- Extended stop words list including book-domain specific terms

Output: amazon_with_keywords.csv with new 'keywords' column (top 10 keywords per book)

For STAT6207 Deep Learning Final Project - Sales Prediction & Distribution Analysis
"""

import pandas as pd
import re
from collections import Counter
import math


# ============================================================================
# EXTENDED STOP WORDS (General + NLTK + Book Domain Specific)
# ============================================================================
stop_words = set([
    # Original general words (refined)
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those',
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of', 'to', 'and', 'or',
    'but', 'not', 'no', 'yes', 'from', 'up', 'down', 'out', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now',
    
    # NLTK extensions (pronouns, auxiliary verbs, prepositions, etc.)
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'am', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'if', 'because', 'until', 'while', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'nor',
    
    # Book domain specific extensions (high-frequency low-information words)
    'book', 'books', 'children', 'child', 'story', 'stories', 'series', 'author', 'authors',
    'reading', 'read', 'age', 'ages', 'years', 'year', 'fun', 'new', 'one', 'see', 'top',
    'like', 'get', 'great', 'first', 'little', 'world', 'time', 'life', 'family', 'home',
    'day', 'night', 'way', 'back', 'end', 'part', 'set', 'collection', 'edition', 'volume',
    'make', 'made', 'help', 'best', 'ever', 'also', 'much', 'many', 'well', 'good', 'know',
    'would', 'could', 'might', 'must', 'shall', 'may', 'us', 'around', 'across', 'along'
])


# ============================================================================
# TF (Term Frequency) CALCULATION
# ============================================================================
def get_tf(text):
    """
    Calculate Term Frequency (TF) for words in text.
    EXCLUDES all numeric content (numbers, ISBNs, codes).
    
    Args:
        text: Input text string
        
    Returns:
        Counter dict with TF scores (normalized by total word count)
    """
    words = re.findall(r'\w+', str(text).lower())
    
    # Filter: keep ONLY alphabetic words, exclude stop words, length > 2
    filtered_words = []
    for w in words:
        # Skip anything containing digits (777, 068, abc123, etc.)
        if any(char.isdigit() for char in w):
            continue
        # Skip stop words
        if w in stop_words:
            continue
        # Keep only meaningful alphabetic words (length > 2)
        if len(w) > 2:
            filtered_words.append(w)
    
    words = filtered_words
    total = len(words)
    tf = Counter(words)
    for w in tf:
        tf[w] = tf[w] / total if total > 0 else 0
    return tf


# ============================================================================
# MAIN KEYWORD GENERATION FUNCTION
# ============================================================================
def generate_keywords(
    input_file='amazon_cleaned.csv',
    output_file='amazon_with_keywords.csv',
    top_n=10
):
    """
    Generate keywords for each book using TF-IDF algorithm.
    
    Args:
        input_file: Path to cleaned CSV file
        output_file: Path to save CSV with keywords
        top_n: Number of top keywords to extract per book
        
    Returns:
        DataFrame with keywords column added
    """
    print("=" * 80)
    print("AMAZON BOOK KEYWORD GENERATOR (TF-IDF)")
    print("STAT6207 Deep Learning Final Project")
    print("=" * 80)
    
    # Load CSV file
    print(f"\nLoading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig', on_bad_lines='skip')
        print(f"✓ Loaded {len(df)} books with {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Use ONLY description column for keyword extraction
    print("\nPreparing description text for keyword analysis...")
    
    if 'description' not in df.columns:
        print("❌ Error: 'description' column not found in CSV!")
        return None
    
    print("  Using column: description only")
    df['combined_text'] = df['description'].fillna('')
    
    N = len(df)  # Total number of documents
    print(f"  Total documents: {N}")
    
    # Calculate Document Frequency (DF)
    print("\nCalculating Document Frequency (DF)...")
    doc_freq = Counter()
    for idx, text in enumerate(df['combined_text']):
        if (idx + 1) % 500 == 0:
            print(f"  Processing: {idx + 1}/{N} documents...")
        words = set(re.findall(r'\w+', str(text).lower()))
        
        # Exclude ALL numeric content, keep only alphabetic words
        filtered_words = set()
        for w in words:
            # Skip anything containing digits
            if any(char.isdigit() for char in w):
                continue
            # Skip stop words
            if w in stop_words:
                continue
            # Keep only meaningful alphabetic words (length > 2)
            if len(w) > 2:
                filtered_words.add(w)
        
        doc_freq.update(filtered_words)
    
    print(f"✓ Found {len(doc_freq)} unique terms")
    
    # Calculate IDF (Inverse Document Frequency)
    print("\nCalculating IDF scores...")
    idf = {w: math.log(N / (doc_freq[w] + 1)) for w in doc_freq}
    
    # Calculate TF-IDF and extract top keywords for each book
    print(f"\nExtracting top {top_n} keywords per book using TF-IDF...")
    keywords_list = []
    
    for idx, text in enumerate(df['combined_text']):
        if (idx + 1) % 500 == 0:
            print(f"  Processing: {idx + 1}/{N} books...")
        
        # Calculate TF
        tf = get_tf(text)
        
        # Calculate TF-IDF
        tfidf = {w: tf[w] * idf.get(w, 0) for w in tf}
        
        # Sort by TF-IDF score and get top N
        sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Join keywords with comma
        keywords = ', '.join([w for w, score in sorted_tfidf])
        keywords_list.append(keywords)
    
    # Add keywords column to dataframe
    df['keywords'] = keywords_list
    
    # Drop temporary combined_text column
    df.drop('combined_text', axis=1, inplace=True)
    
    # Save to new CSV file
    print(f"\nSaving results to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved {len(df)} books with keywords")
    
    # Display sample results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS (First 5 Books)")
    print("=" * 80)
    
    sample_cols = ['product_name', 'keywords']
    existing_sample_cols = [col for col in sample_cols if col in df.columns]
    
    if existing_sample_cols:
        print(df[existing_sample_cols].head(5).to_string(index=False))
    else:
        print("⚠ Could not display sample (columns not found)")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    # Count books with keywords
    books_with_keywords = df['keywords'].notna().sum()
    print(f"Books with keywords: {books_with_keywords}/{len(df)} ({books_with_keywords/len(df)*100:.1f}%)")
    
    # Average number of keywords
    avg_keywords = df['keywords'].str.split(',').str.len().mean()
    print(f"Average keywords per book: {avg_keywords:.1f}")
    
    # Most common keywords (across all books)
    all_keywords = []
    for kw_str in df['keywords'].dropna():
        keywords = [k.strip() for k in kw_str.split(',')]
        all_keywords.extend(keywords)
    
    if all_keywords:
        top_keywords = Counter(all_keywords).most_common(20)
        print(f"\nTop 20 Most Common Keywords:")
        for kw, count in top_keywords:
            print(f"  {kw}: {count} books")
    
    print("\n" + "=" * 80)
    print("✅ KEYWORD GENERATION COMPLETE!")
    print(f"   Output file: {output_file}")
    print("   Ready for sales prediction & distribution analysis")
    print("=" * 80)
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    INPUT_FILE = 'amazon_cleaned.csv'
    OUTPUT_FILE = 'amazon_with_keywords.csv'
    TOP_N_KEYWORDS = 5  # Extract top 5 keywords per book (focused)
    
    try:
        df = generate_keywords(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            top_n=TOP_N_KEYWORDS
        )
        
        if df is not None:
            print("\n✓ Success! You can now use amazon_with_keywords.csv for:")
            print("  1. Sales distribution analysis")
            print("  2. Predictive modeling (regression/classification)")
            print("  3. Deep learning feature extraction")
            print("  4. Market segmentation by keyword clusters")
    
    except FileNotFoundError:
        print(f"\n❌ Error: {INPUT_FILE} not found!")
        print("   Make sure you've run data_cleaning.py first to generate the cleaned CSV.")
    except KeyboardInterrupt:
        print("\n\n⚠ Keyword generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
