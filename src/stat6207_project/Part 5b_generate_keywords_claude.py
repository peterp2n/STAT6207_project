"""
Keyword Extraction using Claude AI (via Poe API)
使用 Claude AI 提取关键词（通过 Poe API）

This script uses Claude Haiku model to extract semantically relevant keywords
from book descriptions, providing better contextual understanding than TF-IDF.

For STAT6207 Deep Learning Final Project
"""

import openai
import pandas as pd
import os
import time
from typing import List, Optional

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Initialize OpenAI client for Poe API
client = openai.OpenAI(
    api_key="PEfUC0EuyBbmXz4l5uDqlIfNgsld68ohpFij8WtVwQ4",
    base_url="https://api.poe.com/v1",
)

# Model configuration
MODEL_NAME = "claude-haiku-4.5"
THINKING_BUDGET = 63999
NUM_KEYWORDS = 5  # Number of keywords to extract per book


# ============================================================================
# KEYWORD EXTRACTION FUNCTION
# ============================================================================

def extract_keywords_claude(
    description: str, 
    num_keywords: int = NUM_KEYWORDS,
    max_retries: int = 3
) -> List[str]:
    """
    Extract keywords from book description using Claude AI.
    
    Args:
        description: Book description text
        num_keywords: Number of keywords to extract
        max_retries: Maximum number of API call retries
        
    Returns:
        List of extracted keywords (lowercase, comma-separated)
    """
    # Handle empty or invalid descriptions
    if pd.isna(description) or not isinstance(description, str):
        return []
    
    description = description.strip()
    if not description:
        return []
    
    # Truncate very long descriptions to avoid token limits
    if len(description) > 2000:
        description = description[:2000] + "..."
    
    # Construct prompt for Claude
    prompt = f"""Extract exactly {num_keywords} most important keywords from this book description.

Requirements:
- Return ONLY the keywords, separated by commas
- Use lowercase letters
- Focus on: genre, themes, target audience, key concepts
- Avoid: common words like "book", "story", "great"
- Be concise and relevant

Book description:
{description}

Keywords:"""
    
    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a keyword extraction expert specializing in book categorization."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Low temperature for consistency
                max_tokens=100  # Limit response length
            )
            
            # Parse response
            keywords_text = response.choices[0].message.content.strip()
            
            # Clean and split keywords
            keywords = [
                kw.strip().lower() 
                for kw in keywords_text.split(',')
            ]
            
            # Remove empty strings and limit to requested number
            keywords = [kw for kw in keywords if kw][:num_keywords]
            
            return keywords
            
        except openai.AuthenticationError:
            print(f"❌ Authentication error: Invalid API key")
            return []
        except openai.NotFoundError:
            print(f"❌ Model '{MODEL_NAME}' not available")
            return []
        except openai.RateLimitError:
            wait_time = (attempt + 1) * 5
            print(f"⚠ Rate limit hit. Waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"⚠ Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return []
            time.sleep(2)
    
    return []


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_csv_with_claude(
    input_csv: str = 'amazon_cleaned.csv',
    output_csv: str = 'amazon_with_claude_keywords.csv',
    sample_size: Optional[int] = None
):
    """
    Process CSV file and add Claude-extracted keywords.
    
    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        sample_size: If set, only process first N rows (for testing)
    """
    print("=" * 80)
    print("KEYWORD EXTRACTION USING CLAUDE AI")
    print("STAT6207 Deep Learning Final Project")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} books")
    
    # Sample if requested
    if sample_size:
        df = df.head(sample_size)
        print(f"⚠ Processing sample of {sample_size} books for testing")
    
    # Check if description column exists
    if 'description' not in df.columns:
        print("❌ Error: 'description' column not found in CSV")
        return
    
    # Count books with descriptions
    has_description = df['description'].notna().sum()
    print(f"📊 Books with descriptions: {has_description}/{len(df)}")
    
    # Extract keywords
    print(f"\n🤖 Extracting keywords using Claude ({MODEL_NAME})...")
    print(f"   Keywords per book: {NUM_KEYWORDS}")
    print(f"   API: Poe (https://api.poe.com/v1)")
    print()
    
    keywords_list = []
    successful = 0
    
    for idx, row in df.iterrows():
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"   Processing {idx + 1}/{len(df)}... "
                  f"({successful} successful)", end='\r')
        
        # Extract keywords
        keywords = extract_keywords_claude(row['description'])
        
        if keywords:
            successful += 1
            keywords_str = ', '.join(keywords)
        else:
            keywords_str = ''
        
        keywords_list.append(keywords_str)
        
        # Rate limiting: small delay between requests
        if idx < len(df) - 1:
            time.sleep(0.5)
    
    print(f"\n✓ Completed: {successful}/{len(df)} books processed successfully")
    
    # Add keywords to dataframe
    df['claude_keywords'] = keywords_list
    
    # Show examples
    print("\n📝 Sample results:")
    print("-" * 80)
    for idx in range(min(5, len(df))):
        if df.iloc[idx]['claude_keywords']:
            print(f"\nBook {idx + 1}: {df.iloc[idx]['product_name'][:50]}...")
            print(f"Keywords: {df.iloc[idx]['claude_keywords']}")
    
    # Save results
    print(f"\n💾 Saving to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df)} books with Claude keywords")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total books: {len(df)}")
    print(f"Successfully extracted: {successful}")
    print(f"Failed: {len(df) - successful}")
    print(f"Success rate: {successful/len(df)*100:.1f}%")
    print("=" * 80)


# ============================================================================
# COMPARISON WITH TF-IDF
# ============================================================================

def compare_methods(
    tfidf_csv: str = 'amazon_with_keywords.csv',
    claude_csv: str = 'amazon_with_claude_keywords.csv'
):
    """
    Compare TF-IDF keywords vs Claude keywords.
    
    Args:
        tfidf_csv: CSV with TF-IDF keywords
        claude_csv: CSV with Claude keywords
    """
    print("\n" + "=" * 80)
    print("COMPARING TF-IDF VS CLAUDE KEYWORDS")
    print("=" * 80)
    
    df_tfidf = pd.read_csv(tfidf_csv)
    df_claude = pd.read_csv(claude_csv)
    
    print("\nSample comparison (first 3 books):")
    print("-" * 80)
    
    for idx in range(min(3, len(df_tfidf))):
        print(f"\nBook: {df_tfidf.iloc[idx]['product_name'][:60]}...")
        print(f"TF-IDF:  {df_tfidf.iloc[idx]['keywords']}")
        print(f"Claude:  {df_claude.iloc[idx]['claude_keywords']}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == '--test':
                # Test mode: process only 10 books
                print("🧪 TEST MODE: Processing 10 books only")
                process_csv_with_claude(
                    input_csv='amazon_cleaned.csv',
                    output_csv='amazon_with_claude_keywords_test.csv',
                    sample_size=10
                )
            elif sys.argv[1] == '--compare':
                # Compare TF-IDF vs Claude
                compare_methods()
            else:
                print("Usage:")
                print("  python Part 5b_generate_keywords_claude.py")
                print("  python Part 5b_generate_keywords_claude.py --test")
                print("  python Part 5b_generate_keywords_claude.py --compare")
        else:
            # Full processing
            user_input = input(
                "\n⚠ WARNING: This will process all 1731 books using Poe API.\n"
                "   This may consume significant API credits.\n"
                "   Estimated time: ~15 minutes (with rate limiting)\n\n"
                "   Run in test mode first? (y/n): "
            )
            
            if user_input.lower() == 'y':
                print("\n🧪 Running test mode (10 books)...")
                process_csv_with_claude(
                    input_csv='amazon_cleaned.csv',
                    output_csv='amazon_with_claude_keywords_test.csv',
                    sample_size=10
                )
            else:
                print("\n🚀 Running full processing...")
                process_csv_with_claude(
                    input_csv='amazon_cleaned.csv',
                    output_csv='amazon_with_claude_keywords.csv'
                )
        
        print("\n✅ Script completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
