#!/usr/bin/env python3
"""
Flux LoRA Dataset Preparation Script

This script:
1. Converts JPG/JPEG images to PNG format
2. Generates image descriptions using BLIP or other taggers
3. Saves descriptions in .txt and .json formats for Flux LoRA training
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import sys

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Install with: pip install torch")


class ImageTagger:
    """Base class for image tagging/captioning"""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        if device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    def generate_caption(self, image_path: str) -> str:
        """Generate caption for an image"""
        raise NotImplementedError


class BLIPTagger(ImageTagger):
    """BLIP-based image captioner with detailed tagging"""
    
    def __init__(self, device: str = "auto", detailed: bool = True):
        super().__init__(device)
        if not BLIP_AVAILABLE:
            raise ImportError("transformers library required for BLIP tagger")
        
        # Try to load on requested device, fallback to CPU if CUDA fails
        requested_device = self.device
        if requested_device == "auto":
            requested_device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        print(f"Loading BLIP model on {requested_device}...")
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            
            # Try to load on requested device
            try:
                self.model = self.model.to(requested_device)
                self.device = requested_device
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"Warning: CUDA out of memory, falling back to CPU...")
                    if TORCH_AVAILABLE:
                        torch.cuda.empty_cache()
                    self.model = self.model.to("cpu")
                    self.device = "cpu"
                else:
                    raise
            
            self.model.eval()
            self.detailed = detailed
            print(f"BLIP model loaded successfully on {self.device}!")
        except Exception as e:
            # Final fallback to CPU
            if requested_device != "cpu":
                print(f"Error loading on {requested_device}, trying CPU fallback...")
                try:
                    if TORCH_AVAILABLE:
                        torch.cuda.empty_cache()
                    self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        "Salesforce/blip-image-captioning-base"
                    ).to("cpu")
                    self.model.eval()
                    self.device = "cpu"
                    self.detailed = detailed
                    print("BLIP model loaded successfully on CPU (fallback)!")
                except Exception as e2:
                    raise Exception(f"Failed to load BLIP model on both {requested_device} and CPU: {e2}")
            else:
                raise
    
    def _deduplicate_words(self, text: str) -> str:
        """Remove consecutive repeated words from text"""
        words = text.split()
        if not words:
            return text
        
        deduplicated = []
        for word in words:
            # Only add if it's different from the last word (case-insensitive)
            # This removes consecutive duplicates like "stained stained stained"
            if not deduplicated or word.lower() != deduplicated[-1].lower():
                deduplicated.append(word)
        
        return " ".join(deduplicated)
    
    def generate_caption(self, image_path: str) -> str:
        """Generate detailed caption using BLIP with multiple passes for comprehensive tagging"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.detailed:
                # Generate multiple captions and combine them for comprehensive tagging
                captions = []
                
                # First pass: General description with high repetition penalty
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, 
                        max_length=75, 
                        num_beams=5,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.5  # Higher penalty to prevent repetition
                    )
                caption1 = self.processor.decode(out[0], skip_special_tokens=True).strip()
                # Clean up any obvious repetitions in the caption itself
                caption1 = self._deduplicate_words(caption1)
                captions.append(caption1)
                
                # Second pass: Detailed description with different parameters
                try:
                    inputs = self.processor(image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_length=80,
                            num_beams=6,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.95,
                            repetition_penalty=1.6  # Even higher penalty
                        )
                    caption2 = self.processor.decode(out[0], skip_special_tokens=True).strip()
                    caption2 = self._deduplicate_words(caption2)
                    if caption2 and caption2 not in captions:
                        captions.append(caption2)
                except Exception as e:
                    pass  # Fallback: use only first caption
                
                # Third pass: Another variation
                try:
                    inputs = self.processor(image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **inputs,
                            max_length=70,
                            num_beams=4,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.9,
                            top_p=0.9,
                            repetition_penalty=1.5
                        )
                    caption3 = self.processor.decode(out[0], skip_special_tokens=True).strip()
                    caption3 = self._deduplicate_words(caption3)
                    if caption3 and caption3 not in captions:
                        captions.append(caption3)
                except Exception as e:
                    pass  # Fallback: use available captions
                
                # Combine captions into a comprehensive tag-like description
                # Extract key phrases and important words
                important_words = set()
                phrases = []
                
                # Collect all descriptive elements
                for cap in captions:
                    if not cap:
                        continue
                    # Clean and split into meaningful phrases
                    cap_clean = cap.lower().strip()
                    # Remove common stop words for tag extraction
                    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    
                    # Extract noun phrases and important words
                    words = cap_clean.replace(',', ' ').replace('.', ' ').replace(':', ' ').split()
                    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
                    important_words.update(meaningful_words)
                    
                    # Keep full phrases that are descriptive
                    if len(cap.split()) > 3:  # Keep longer descriptive phrases
                        phrases.append(cap)
                
                # Start with the longest/most detailed caption as base
                if phrases:
                    base_caption = max(phrases, key=len)
                else:
                    base_caption = captions[0] if captions else ""
                
                # Create tag-like format: base description + additional key elements
                if len(captions) > 1:
                    # Extract unique descriptive elements from other captions
                    base_words = set(base_caption.lower().replace(',', ' ').replace('.', ' ').split())
                    additional_elements = []
                    
                    for cap in captions:
                        if cap == base_caption:
                            continue
                        # Find unique descriptive phrases/words
                        cap_words = set(cap.lower().replace(',', ' ').replace('.', ' ').split())
                        unique = cap_words - base_words
                        # Filter out stop words and very short words
                        unique_meaningful = set([w for w in unique if w not in stop_words and len(w) > 2])
                        if unique_meaningful:
                            # Try to preserve phrases if they make sense
                            cap_phrases = [p.strip() for p in cap.split(',')]
                            for phrase in cap_phrases:
                                phrase_words = set(phrase.lower().split())
                                if phrase_words & unique_meaningful and len(phrase.split()) <= 5:
                                    if phrase not in additional_elements:
                                        additional_elements.append(phrase)
            
                    # Combine: base caption + additional unique elements
                    if additional_elements:
                        result = base_caption
                        for elem in additional_elements[:3]:  # Limit to top 3 additional elements
                            if elem.lower() not in result.lower():
                                result += ", " + elem
                        return result.strip()
                    else:
                        return base_caption
                else:
                    return base_caption
            else:
                # Simple single-pass generation
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_length=75, num_beams=5)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                return caption.strip()
                
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return ""


class SimpleTagger(ImageTagger):
    """Simple tagger that extracts basic metadata"""
    
    def generate_caption(self, image_path: str) -> str:
        """Generate basic description from filename and metadata"""
        path = Path(image_path)
        # Use filename as base description
        name = path.stem.replace("_", " ").replace("-", " ")
        return name


class DatasetPreparer:
    """Main class for preparing Flux LoRA datasets"""
    
    def __init__(self, tagger: ImageTagger, output_format: str = "both"):
        self.tagger = tagger
        self.output_format = output_format  # "txt", "json", or "both"
    
    def convert_jpg_to_png(self, input_path: str, output_path: str) -> bool:
        """Convert JPG/JPEG image to PNG"""
        try:
            img = Image.open(input_path).convert("RGB")
            img.save(output_path, "PNG", optimize=True)
            return True
        except Exception as e:
            print(f"Error converting {input_path} to PNG: {e}")
            return False
    
    def process_image(self, image_path: Path, output_dir: Path, 
                     convert_to_png: bool = True) -> Dict:
        """Process a single image: convert and generate caption"""
        result = {
            "input_file": str(image_path),
            "success": False,
            "caption": "",
            "output_files": []
        }
        
        try:
            # Generate caption from original image
            caption = self.tagger.generate_caption(str(image_path))
            result["caption"] = caption
            
            # Determine output paths
            base_name = image_path.stem
            output_png = output_dir / f"{base_name}.png"
            output_txt = output_dir / f"{base_name}.txt"
            output_json = output_dir / f"{base_name}.json"
            
            # Convert to PNG if needed
            if convert_to_png and image_path.suffix.lower() in ['.jpg', '.jpeg']:
                if self.convert_jpg_to_png(str(image_path), str(output_png)):
                    result["output_files"].append(str(output_png))
                else:
                    return result
            elif convert_to_png:
                # Already PNG or other format, copy it
                img = Image.open(image_path).convert("RGB")
                img.save(output_png, "PNG", optimize=True)
                result["output_files"].append(str(output_png))
            
            # Save caption files
            if self.output_format in ["txt", "both"]:
                with open(output_txt, "w", encoding="utf-8") as f:
                    f.write(caption)
                result["output_files"].append(str(output_txt))
            
            if self.output_format in ["json", "both"]:
                json_data = {
                    "image": output_png.name,
                    "caption": caption,
                    "source": image_path.name
                }
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                result["output_files"].append(str(output_json))
            
            result["success"] = True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         convert_to_png: bool = True, 
                         extensions: List[str] = None) -> Dict:
        """Process all images in a directory"""
        if extensions is None:
            # Normalized extensions (lowercase) - we'll normalize suffixes when checking
            extensions = ['.jpg', '.jpeg', '.png']
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files - glob once and normalize suffixes to avoid duplicates
        # Use a set to track seen files (normalized by lowercase path)
        seen_files = set()
        image_files = []
        
        # Glob all potential image files at once
        for pattern in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for file_path in input_dir.glob(pattern):
                # Normalize by converting to lowercase path for comparison
                normalized_path = str(file_path).lower()
                if normalized_path not in seen_files:
                    seen_files.add(normalized_path)
                    image_files.append(file_path)
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return {"processed": 0, "failed": 0, "results": []}
        
        print(f"Found {len(image_files)} image(s) to process")
        
        results = []
        processed = 0
        failed = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            result = self.process_image(image_path, output_dir, convert_to_png)
            results.append(result)
            
            if result["success"]:
                processed += 1
                print(f"  ✓ Caption: {result['caption'][:80]}...")
            else:
                failed += 1
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        summary = {
            "processed": processed,
            "failed": failed,
            "total": len(image_files),
            "results": results
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare images and captions for Flux LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images with BLIP tagger, save both txt and json
  python prepare_flux_lora_dataset.py input_dir output_dir --tagger blip

  # Only save txt files
  python prepare_flux_lora_dataset.py input_dir output_dir --tagger blip --format txt

  # Use simple tagger (filename-based)
  python prepare_flux_lora_dataset.py input_dir output_dir --tagger simple

  # Don't convert to PNG (keep original format)
  python prepare_flux_lora_dataset.py input_dir output_dir --no-convert
        """
    )
    
    parser.add_argument("input_dir", type=str, help="Input directory containing images")
    parser.add_argument("output_dir", type=str, help="Output directory for processed images and captions")
    parser.add_argument("--tagger", type=str, default="blip", 
                       choices=["blip", "simple"],
                       help="Tagger to use (default: blip)")
    parser.add_argument("--detailed", action="store_true", default=True,
                       help="Generate detailed multi-pass descriptions (default: True)")
    parser.add_argument("--format", type=str, default="both",
                       choices=["txt", "json", "both"],
                       help="Output format for captions (default: both)")
    parser.add_argument("--no-convert", action="store_true",
                       help="Don't convert images to PNG (keep original format)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for tagging (default: auto). Use 'cpu' if GPU memory is full.")
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Initialize tagger
    print(f"\nInitializing {args.tagger} tagger...")
    try:
        if args.tagger == "blip":
            if not BLIP_AVAILABLE:
                print("Error: BLIP tagger requires transformers library.")
                print("Install with: pip install transformers torch")
                sys.exit(1)
            tagger = BLIPTagger(device=args.device, detailed=args.detailed)
        elif args.tagger == "simple":
            tagger = SimpleTagger(device=args.device)
        else:
            print(f"Error: Unknown tagger: {args.tagger}")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing tagger: {e}")
        sys.exit(1)
    
    # Create preparer
    preparer = DatasetPreparer(tagger, output_format=args.format)
    
    # Process directory
    print(f"\nProcessing images from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Convert to PNG: {not args.no_convert}")
    print(f"Output format: {args.format}\n")
    
    summary = preparer.process_directory(
        input_dir, 
        output_dir,
        convert_to_png=not args.no_convert
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total images: {summary['total']}")
    print(f"Successfully processed: {summary['processed']}")
    print(f"Failed: {summary['failed']}")
    print("="*60)
    
    if summary['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

