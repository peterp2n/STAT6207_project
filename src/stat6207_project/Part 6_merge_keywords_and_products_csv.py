"""
Merge TF-IDF and Claude Keywords
合并 TF-IDF 和 Claude 关键词

This script merges keywords from both TF-IDF and Claude methods into a single
dataset, allowing you to use both approaches in your deep learning model.

For STAT6207 Deep Learning Final Project
"""

import pandas as pd


def merge_keywords():
    """Merge TF-IDF and Claude keywords into single dataset, then merge with products.csv."""
    print("=" * 80)
    print("MERGING TF-IDF AND CLAUDE KEYWORDS + PRODUCTS")
    print("STAT6207 Deep Learning Final Project")
    print("=" * 80)
    
    # Load both datasets
    print("\n📂 Loading datasets...")
    df_tfidf = pd.read_csv('amazon_with_keywords.csv')
    df_claude = pd.read_csv('amazon_with_claude_keywords.csv')
    df_products = pd.read_csv('products.csv')
    
    print(f"✓ Loaded TF-IDF: {len(df_tfidf)} books")
    print(f"✓ Loaded Claude: {len(df_claude)} books")
    
    # Start with TF-IDF dataset as base
    df_merged = df_tfidf.copy()
    
    # Add Claude keywords column
    print("\n🔗 Merging keywords...")
    df_merged['claude_keywords'] = df_claude['claude_keywords']
    
    # Create combined keywords column (TF-IDF + Claude)
    print("🔗 Creating combined keywords column...")
    combined_keywords = []
    
    for idx in range(len(df_merged)):
        tfidf_kw = df_merged.iloc[idx]['keywords']
        claude_kw = df_merged.iloc[idx]['claude_keywords']
        
        # Collect all keywords
        all_kw = []
        
        # Add TF-IDF keywords
        if pd.notna(tfidf_kw) and str(tfidf_kw).strip():
            all_kw.extend([kw.strip() for kw in str(tfidf_kw).split(',')])
        
        # Add Claude keywords
        if pd.notna(claude_kw) and str(claude_kw).strip():
            all_kw.extend([kw.strip() for kw in str(claude_kw).split(',')])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_kw = []
        for kw in all_kw:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_kw.append(kw)
        
        combined_keywords.append(', '.join(unique_kw) if unique_kw else '')
    
    df_merged['keywords_combined'] = combined_keywords
    
    # Clean up: drop individual columns and rename combined to keywords
    print("\n🧹 Cleaning up columns...")
    df_merged = df_merged.drop(columns=['keywords', 'claude_keywords', 'asin'])
    df_merged = df_merged.rename(columns={'keywords_combined': 'keywords'})
    
    # Merge with products.csv (left join with products on the left)
    print("🔗 Merging with products.csv...")
    # Convert barcode columns to string for consistent merge
    df_products['barcode'] = df_products['barcode'].astype(str)
    df_merged['barcode'] = df_merged['barcode'].astype(str)
    
    # Rename products price columns to keep them
    df_products = df_products.rename(columns={
        'price': 'price_hkd',
        'book_original_price': 'original_price_hkd'
    })
    
    df_final = pd.merge(df_products, df_merged, on='barcode', how='left',
                        suffixes=('_prod', '_amazon'))
    
    # Convert Amazon price from USD to HKD (USD to HKD rate ~ 7.8)
    print("💱 Converting Amazon price from USD to HKD...")
    usd_to_hkd_rate = 7.8
    if 'price' in df_final.columns:
        df_final['price_amazon_hkd'] = df_final['price'] * usd_to_hkd_rate
        df_final = df_final.drop(columns=['price'])
    
    print(f"✓ Final dataset: {len(df_final)} rows")
    print(f"  Products: {len(df_products)} rows")
    print(f"  Amazon: {len(df_merged)} rows")
    print(f"  Matched: {df_final['keywords'].notna().sum()} rows have keywords")
    
    # Update df_merged to df_final for rest of the script
    df_merged = df_final
    
    # Reorder columns to specified sequence (products columns first, then amazon)
    print("🔄 Reordering columns...")
    column_order = [
        'product_name', 'author', 'publisher', 'barcode', 'isbn_10', 'isbn_13',
        'availability', 'best_sellers_rank', 'number_of_reviews', 'customer_reviews',
        'rating', 'price', 'description', 'keywords', 'print_length', 'item_weight',
        'length', 'width', 'height', 'publication_date', 'reading_age', 'book_format',
        'edition', 'product_url', 'features', 'language', 'part_of_series'
    ]
    # Only include columns that exist
    column_order = [col for col in column_order if col in df_merged.columns]
    # Add any remaining columns not in the order list
    remaining_cols = [col for col in df_merged.columns if col not in column_order]
    df_merged = df_merged[column_order + remaining_cols]
    
    # Statistics
    print("\n📊 Statistics:")
    print("-" * 80)
    
    keywords_count = (df_merged['keywords'].notna() & 
                      (df_merged['keywords'] != '')).sum()
    
    print(f"Books with merged keywords:    {keywords_count:>4} ({keywords_count/len(df_merged)*100:.1f}%)")
    
    # Average keyword counts
    avg_keywords = df_merged['keywords'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) and x else 0
    ).mean()
    
    print(f"\nAverage keywords per book: {avg_keywords:.2f}")
    
    # Show examples (only rows with keywords)
    print("\n📝 Sample merged keywords:")
    print("-" * 80)
    
    sample_rows = df_merged[df_merged['keywords'].notna()].head(5)
    for idx, row in sample_rows.iterrows():
        product_name = row['product_name'] if pd.notna(row['product_name']) else 'N/A'
        if isinstance(product_name, str):
            product_name = product_name[:60]
        keywords = row['keywords'] if pd.notna(row['keywords']) else 'N/A'
        print(f"\n📖 Book: {product_name}...")
        print(f"Keywords: {keywords}")
    
    # Save merged dataset
    output_file = 'amazon_with_merged_keywords_and_products.csv'
    print(f"\n💾 Saving merged dataset...")
    df_merged.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total books: {len(df_merged)}")
    print(f"Keywords column contains TF-IDF + Claude AI merged and deduplicated")
    print(f"Average {avg_keywords:.2f} keywords per book")
    print("\n💡 Ready for deep learning model training!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        merge_keywords()
        print("\n✅ Merge completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
