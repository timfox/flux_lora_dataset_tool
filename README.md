# Flux LoRA Dataset Preparation Tool

Prepare image datasets for Flux LoRA training with easy AI captioning, PNG conversion, and flexible output options.

---

## Features

- [x] **Convert JPG/JPEG images to PNG format**
- [x] **AI-powered captioning** (using BLIP or filename)
- [x] **Save captions in `.txt` and/or `.json` formats**
- [x] **Simple CLI** *(prepare_flux_lora_dataset.py)*
- [x] **Modern GUI** *(flux_lora_dataset_gui.pyw)* — with Pause, Resume, and Stop!

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Command-Line (CLI)

Process all images in a directory using BLIP captioning:

```bash
python prepare_flux_lora_dataset.py input_directory output_directory
```

#### Options

- `--tagger {blip,simple}`: Choose caption generator
    - `blip`: Uses BLIP (AI; recommended)
    - `simple`: Uses filenames (fast, no AI)
- `--format {txt,json,both}`: Caption output format
    - `txt`: `.txt` files (default for most trainers)
    - `json`: `.json` files
    - `both`: Both
- `--no-convert`: Skip PNG conversion (keep originals)
- `--device {auto,cuda,cpu}`: Device for BLIP
    - `auto` *(default)*: Use GPU if available, else CPU
    - `cuda`: Force GPU
    - `cpu`: Force CPU

#### Examples

Basic BLIP (both formats):
```bash
python prepare_flux_lora_dataset.py ./images ./processed --tagger blip --format both
```

TXT only:
```bash
python prepare_flux_lora_dataset.py ./images ./processed --format txt
```

No AI (filename tags):
```bash
python prepare_flux_lora_dataset.py ./images ./processed --tagger simple
```

Don't convert PNGs:
```bash
python prepare_flux_lora_dataset.py ./images ./processed --no-convert
```

---

### 2. Graphical User Interface (GUI)

The GUI makes dataset preparation easy and interactive. You can pause, resume, or stop processing — ideal for big folders!

#### Start the GUI

```bash
python flux_lora_dataset_gui.pyw
```

#### Key Features

- Folder selection with file browser
- Choose tagger and output format
- Optional PNG conversion
- Live log and visual progress bar
- Pause/Resume/Stop controls
- Error reporting and summaries when done

---

## Output Format

### TXT Files

Each image gets a `.txt` file with the caption:

```
image001.png
image001.txt  (contains: "a beautiful landscape with mountains and trees")
```

### JSON Files

Each image gets a `.json` file with:

```json
{
  "image": "image001.png",
  "caption": "a beautiful landscape with mountains and trees",
  "source": "image001.jpg"
}
```

---

## Notes & Tips

- Supports `.jpg`, `.jpeg`, and `.png` files.
- BLIP model downloads automatically (~990MB, first run).
- Use GPU (`cuda`) for much faster BLIP processing.
- Large datasets may take time; use the GUI for easier management.

---

## Troubleshooting

**Out of memory?**
- Use `--device cpu` or select CPU in the GUI.
- Split your dataset into smaller batches.

**BLIP doesn't work?**
- Ensure you have `transformers` and `torch`:  
  `pip install transformers torch`
- Check for enough disk space for the model.

**Image conversion errors?**
- Make sure your images are valid (not corrupted).
- Check permissions for the input/output folders.

---

*Both CLI and GUI work cross-platform. For best results, use the GUI for large or complex batches!*

