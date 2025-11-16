# Flux LoRA Dataset Preparation Tool

This tool prepares image datasets for Flux LoRA training by:
1. Converting JPG/JPEG images to PNG format
2. Generating image descriptions using AI taggers
3. Saving captions in `.txt` and/or `.json` formats

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process all images in a directory with BLIP captioning:

```bash
python prepare_flux_lora_dataset.py input_directory output_directory
```

### Options

- `--tagger {blip,simple}`: Choose the tagger to use
  - `blip`: Uses BLIP model for AI-generated captions (recommended)
  - `simple`: Uses filename-based descriptions (no AI needed)
  
- `--format {txt,json,both}`: Output format for captions
  - `txt`: Only `.txt` files (standard for most LoRA trainers)
  - `json`: Only `.json` files
  - `both`: Both formats (default)

- `--no-convert`: Skip PNG conversion (keep original format)

- `--device {auto,cuda,cpu}`: Device for AI tagging
  - `auto`: Automatically detect (default)
  - `cuda`: Use GPU if available
  - `cpu`: Force CPU usage

### Examples

**Process with BLIP, save both txt and json:**
```bash
python prepare_flux_lora_dataset.py ./images ./processed --tagger blip --format both
```

**Only generate txt files:**
```bash
python prepare_flux_lora_dataset.py ./images ./processed --tagger blip --format txt
```

**Use simple tagger (no AI, faster):**
```bash
python prepare_flux_lora_dataset.py ./images ./processed --tagger simple
```

**Process without converting to PNG:**
```bash
python prepare_flux_lora_dataset.py ./images ./processed --no-convert
```

## Output Format

### TXT Files
Each image gets a `.txt` file with the same name containing the caption:
```
image001.png
image001.txt  (contains: "a beautiful landscape with mountains and trees")
```

### JSON Files
Each image gets a `.json` file with structured metadata:
```json
{
  "image": "image001.png",
  "caption": "a beautiful landscape with mountains and trees",
  "source": "image001.jpg"
}
```

## Notes

- The script processes `.jpg`, `.jpeg`, and `.png` files
- BLIP model will be downloaded automatically on first use (~990MB)
- GPU is recommended for faster processing with BLIP
- For large datasets, processing may take time depending on your hardware

## Troubleshooting

**Out of memory errors:**
- Use `--device cpu` to force CPU usage
- Process images in smaller batches

**BLIP not working:**
- Ensure `transformers` and `torch` are installed: `pip install transformers torch`
- Check that you have sufficient disk space for model download

**Image conversion errors:**
- Ensure images are valid and not corrupted
- Check file permissions on input/output directories

