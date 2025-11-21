"""
Download high-resolution product images from Amazon for CNN/Deep Learning

Downloads the largest available product image for each ISBN from Amazon product pages.
Images are saved as {ISBN}.jpg in the 'images/' folder.

For STAT6207 Deep Learning Final Project
"""

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path
from PIL import Image
from io import BytesIO


def setup_driver():
    """Initialize Chrome driver with options for faster loading."""
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-images')  # Don't load images in browser (we'll download separately)
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def get_high_res_image_url(driver, product_url):
    """
    Get the highest resolution product image URL from Amazon page.
    
    Amazon image URL structure:
    - Thumbnails: ._AC_UL320_.jpg (320px)
    - Medium: ._AC_UL480_.jpg (480px)
    - Large: ._AC_UL640_.jpg (640px)
    - Extra Large: ._AC_UL800_.jpg (800px)
    - Original/High-res: Remove size suffix or use largest available
    
    Returns the URL of the highest resolution image available.
    """
    try:
        driver.get(product_url)
        time.sleep(1.5)  # Wait for page load
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Method 1: Try to get image from main product image container
        # Amazon typically shows high-res images in the image block
        image_urls = []
        
        # Find the main product image
        main_image = soup.find('img', {'id': 'landingImage'})
        if not main_image:
            main_image = soup.find('img', {'class': 'a-dynamic-image'})
        
        if main_image:
            # Get data-old-hires (highest resolution)
            if main_image.get('data-old-hires'):
                image_urls.append(main_image.get('data-old-hires'))
            
            # Get data-a-dynamic-image (contains multiple resolutions)
            if main_image.get('data-a-dynamic-image'):
                import json
                try:
                    dynamic_images = json.loads(main_image.get('data-a-dynamic-image'))
                    # Sort by resolution (width * height) and get largest
                    for url, dims in dynamic_images.items():
                        image_urls.append(url)
                except:
                    pass
            
            # Get src attribute
            if main_image.get('src'):
                image_urls.append(main_image.get('src'))
        
        # Method 2: Look for high-res image in image block data
        image_block = soup.find('div', {'id': 'imageBlock'})
        if image_block:
            all_imgs = image_block.find_all('img')
            for img in all_imgs:
                if img.get('src'):
                    image_urls.append(img.get('src'))
        
        # Method 3: Check forBook Cover image specifically
        book_cover = soup.find('img', {'class': 'book-cover-image'})
        if book_cover and book_cover.get('src'):
            image_urls.append(book_cover.get('src'))
        
        if not image_urls:
            return None
        
        # Clean and enhance URLs to get highest resolution
        best_url = None
        max_resolution_indicator = 0
        
        for url in image_urls:
            if not url or 'data:image' in url:
                continue
            
            # Remove size restrictions in Amazon URLs to get original size
            # Replace size indicators with larger ones
            enhanced_url = url
            
            # Remove common size restrictions
            size_patterns = [
                '._AC_UL320_', '._AC_UL480_', '._AC_UL640_',
                '._SX300_', '._SY300_', '._SX400_', '._SY400_',
                '._AC_US200_', '._AC_SR200,200_'
            ]
            
            for pattern in size_patterns:
                if pattern in enhanced_url:
                    # Try to get original by removing size specification
                    enhanced_url = enhanced_url.split(pattern)[0] + '.jpg'
                    break
            
            # If URL has size indicator, try larger size
            if '._' in enhanced_url and '.jpg' in enhanced_url:
                # Try to request 1500px version (very high res for CNN)
                base_url = enhanced_url.split('._')[0]
                enhanced_url = base_url + '._AC_UL1500_.jpg'
            
            # Estimate resolution from URL
            resolution = 0
            if 'UL1500' in enhanced_url or 'SX1500' in enhanced_url:
                resolution = 1500
            elif 'UL800' in enhanced_url or 'SX800' in enhanced_url:
                resolution = 800
            elif 'UL640' in enhanced_url or 'SX640' in enhanced_url:
                resolution = 640
            elif 'UL480' in enhanced_url or 'SX480' in enhanced_url:
                resolution = 480
            else:
                resolution = 1000  # Assume original/high-res if no size indicator
            
            if resolution > max_resolution_indicator:
                max_resolution_indicator = resolution
                best_url = enhanced_url
        
        return best_url
    
    except Exception as e:
        print(f"  ⚠ Error getting image URL: {e}")
        return None


def download_image(image_url, save_path, min_size=300):
    """
    Download image from URL and save it.
    
    Args:
        image_url: URL of the image
        save_path: Path to save the image
        min_size: Minimum acceptable width/height in pixels (default 300px for CNN)
    
    Returns:
        tuple: (success: bool, width: int, height: int)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Open image with PIL to check dimensions
        img = Image.open(BytesIO(response.content))
        width, height = img.size
        
        # Check if image is large enough for CNN/deep learning
        if width < min_size or height < min_size:
            print(f"  ⚠ Image too small: {width}x{height}px (minimum: {min_size}x{min_size}px)")
            return False, width, height
        
        # Convert to RGB if needed (some images are RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save image
        img.save(save_path, 'JPEG', quality=95, optimize=True)
        return True, width, height
    
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False, 0, 0


def download_all_images(
    csv_file='amazon_cleaned.csv',
    output_dir='images',
    min_image_size=300,
    max_products=None
):
    """
    Download all product images from CSV file.
    
    Args:
        csv_file: Path to CSV with product data
        output_dir: Directory to save images
        min_image_size: Minimum acceptable image dimension in pixels
        max_products: Maximum number of products to process (None for all)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("AMAZON PRODUCT IMAGE DOWNLOADER")
    print("For CNN/Deep Learning - STAT6207 Final Project")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading data from {csv_file}...")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df)} products")
    
    # Filter products with both ISBN and product_url
    df_with_data = df[df['barcode'].notna() & df['product_url'].notna()].copy()
    print(f"✓ Found {len(df_with_data)} products with ISBN and URL")
    
    if max_products:
        df_with_data = df_with_data.head(max_products)
        print(f"  Limiting to first {max_products} products for testing")
    
    # Initialize driver
    print("\nInitializing Chrome driver...")
    driver = setup_driver()
    
    # Statistics
    successful = 0
    failed = 0
    skipped = 0
    total_size = 0
    resolutions = []
    
    try:
        print(f"\nDownloading images (minimum size: {min_image_size}x{min_image_size}px)...")
        print("-" * 80)
        
        for idx, row in df_with_data.iterrows():
            isbn = str(row['barcode']).strip()
            product_url = row['product_url']
            
            # Clean ISBN (remove any non-digits)
            isbn_clean = ''.join(filter(str.isdigit, isbn))
            
            # If barcode doesn't start with 978, try isbn_13 column
            if isbn_clean and not isbn_clean.startswith('978'):
                if 'isbn_13' in df.columns and pd.notna(row['isbn_13']):
                    isbn_13 = str(row['isbn_13']).strip()
                    isbn_13_clean = ''.join(filter(str.isdigit, isbn_13))
                    if isbn_13_clean.startswith('978'):
                        print(f"[{idx + 1}/{len(df_with_data)}] ℹ Using isbn_13: {isbn_13_clean} (barcode was {isbn_clean})")
                        isbn_clean = isbn_13_clean
                    else:
                        # isbn_13 also doesn't start with 978, skip this product
                        print(f"[{idx + 1}/{len(df_with_data)}] ⚠ Skipping: Neither barcode ({isbn_clean}) nor isbn_13 starts with 978")
                        skipped += 1
                        continue
                else:
                    # No isbn_13 column or it's empty, skip this product
                    print(f"[{idx + 1}/{len(df_with_data)}] ⚠ Skipping: Barcode {isbn_clean} doesn't start with 978, no isbn_13 available")
                    skipped += 1
                    continue
            
            if not isbn_clean:
                print(f"[{idx + 1}/{len(df_with_data)}] ⚠ Invalid ISBN, skipping...")
                skipped += 1
                continue
            
            save_path = output_path / f"{isbn_clean}.jpg"
            
            # Skip if already downloaded
            if save_path.exists():
                try:
                    img = Image.open(save_path)
                    width, height = img.size
                    if width >= min_image_size and height >= min_image_size:
                        print(f"[{idx + 1}/{len(df_with_data)}] ✓ Already exists: {isbn_clean}.jpg ({width}x{height}px)")
                        successful += 1
                        resolutions.append(width * height)
                        continue
                except:
                    pass  # Re-download if file is corrupted
            
            print(f"[{idx + 1}/{len(df_with_data)}] ISBN: {isbn_clean}")
            
            # Get high-res image URL
            image_url = get_high_res_image_url(driver, product_url)
            
            if not image_url:
                print(f"  ✗ No image URL found")
                failed += 1
                continue
            
            print(f"  → Downloading from: {image_url[:80]}...")
            
            # Download image
            success, width, height = download_image(image_url, save_path, min_image_size)
            
            if success:
                file_size = save_path.stat().st_size / 1024  # KB
                total_size += file_size
                resolutions.append(width * height)
                print(f"  ✓ Saved: {isbn_clean}.jpg ({width}x{height}px, {file_size:.1f}KB)")
                successful += 1
            else:
                failed += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
            # Progress update every 10 images
            if (idx + 1) % 10 == 0:
                print(f"\n--- Progress: {idx + 1}/{len(df_with_data)} | Success: {successful} | Failed: {failed} ---\n")
    
    finally:
        print("\nClosing driver...")
        driver.quit()
    
    # Final report
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Total processed: {len(df_with_data)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"⊘ Skipped: {skipped}")
    
    if resolutions:
        avg_resolution = sum(resolutions) / len(resolutions)
        avg_width = int((avg_resolution) ** 0.5)
        print(f"\nImage Statistics:")
        print(f"  Average resolution: ~{avg_width}x{avg_width}px")
        print(f"  Total size: {total_size / 1024:.1f}MB")
        print(f"  Average size: {total_size / successful:.1f}KB per image")
    
    print(f"\n✓ Images saved to: {output_path.resolve()}/")
    print("=" * 80)


def main():
    """Main execution function."""
    # Configuration
    CSV_FILE = 'amazon_cleaned.csv'
    OUTPUT_DIR = 'images'
    MIN_IMAGE_SIZE = 300  # Minimum 300x300px for CNN/deep learning
    MAX_PRODUCTS = None  # Set to a number (e.g., 10) for testing, None for all
    
    try:
        download_all_images(
            csv_file=CSV_FILE,
            output_dir=OUTPUT_DIR,
            min_image_size=MIN_IMAGE_SIZE,
            max_products=MAX_PRODUCTS
        )
    except FileNotFoundError:
        print(f"\n❌ Error: {CSV_FILE} not found!")
        print("   Make sure the CSV file exists in the current directory.")
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
