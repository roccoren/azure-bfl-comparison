#!/usr/bin/env python3
"""Test script for person orientation detection."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from azure_bfl_compare.tasks.outfit_transfer import EnhancedOutfitTransferPipeline
from PIL import Image


def test_single_image(pipeline: EnhancedOutfitTransferPipeline, image_path: Path) -> None:
    """Test orientation detection on a single image."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {image_path.name}")
    print(f"{'=' * 70}")
    
    try:
        with Image.open(image_path) as img:
            original_rgb = img.convert("RGB")
        
        result = pipeline.detect_person_orientation(original_rgb)
        
        print(f"\n📊 Detection Results:")
        print(f"   Orientation: {result['orientation']}")
        print(f"   Confidence:  {result['confidence']}")
        print(f"   Description: {result['description']}")
        
        # Show the generated instruction
        if result['orientation']:
            instruction = pipeline._get_orientation_instruction(
                result['orientation'], 
                result['confidence']
            )
            print(f"\n📝 Generated Instruction:")
            print(f"   {instruction}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None


def test_multiple_images(pipeline: EnhancedOutfitTransferPipeline, image_dir: Path) -> None:
    """Test orientation detection on multiple images."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [
        f for f in image_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n🔍 Testing orientation detection on {len(image_files)} images...")
    
    results = []
    for image_path in sorted(image_files)[:5]:  # Test first 5 images
        result = test_single_image(pipeline, image_path)
        if result:
            results.append((image_path.name, result))
    
    # Summary
    print(f"\n{'=' * 70}")
    print("📈 SUMMARY")
    print(f"{'=' * 70}")
    
    orientation_counts = {}
    confidence_counts = {}
    
    for name, result in results:
        orientation = result.get('orientation', 'None')
        confidence = result.get('confidence', 'None')
        
        orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        
        print(f"   {name:30s} → {orientation:20s} ({confidence})")
    
    print(f"\n   Orientation Distribution:")
    for orientation, count in sorted(orientation_counts.items()):
        print(f"      {orientation:20s}: {count}")
    
    print(f"\n   Confidence Distribution:")
    for confidence, count in sorted(confidence_counts.items()):
        print(f"      {confidence:20s}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Test person orientation detection")
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a single image to test",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to test (tests first 5)",
    )
    args = parser.parse_args()
    
    # Initialize pipeline
    print("🚀 Initializing outfit transfer pipeline...")
    pipeline = EnhancedOutfitTransferPipeline()
    
    if not pipeline.azure_gpt_endpoint or not pipeline.azure_gpt_key:
        print("⚠️  WARNING: Azure GPT credentials not configured.")
        print("   Set AZURE_GPT_ENDPOINT and AZURE_GPT_API_KEY in your .env file.")
        print("   Orientation detection will be skipped.\n")
    
    if args.image:
        if not args.image.exists():
            print(f"❌ Error: Image not found: {args.image}")
            return 1
        test_single_image(pipeline, args.image)
    
    elif args.image_dir:
        if not args.image_dir.exists():
            print(f"❌ Error: Directory not found: {args.image_dir}")
            return 1
        test_multiple_images(pipeline, args.image_dir)
    
    else:
        # Default: test on sample images from the project
        default_dirs = [
            project_root / "images" / "background",
            project_root / "input" / "id",
        ]
        
        for img_dir in default_dirs:
            if img_dir.exists():
                test_multiple_images(pipeline, img_dir)
                break
        else:
            print("❌ No default image directory found.")
            print("Usage:")
            print(f"  {sys.argv[0]} --image path/to/image.jpg")
            print(f"  {sys.argv[0]} --image-dir path/to/images/")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())