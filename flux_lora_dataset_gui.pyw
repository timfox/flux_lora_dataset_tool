#!/usr/bin/env python3
"""
Flux LoRA Dataset Preparation GUI

This script provides a PyQt6 GUI for:
1. Converting JPG/JPEG images to PNG format
2. Generating image descriptions using BLIP or other taggers
3. Saving descriptions in .txt and .json formats for Flux LoRA training
4. Ability to pause or stop the processing of the dataset
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QTextEdit, QComboBox, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

        requested_device = self.device
        if requested_device == "auto":
            requested_device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        try:
            self.model = self.model.to(requested_device)
            self.device = requested_device
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                self.model = self.model.to("cpu")
                self.device = "cpu"
            else:
                raise
        self.model.eval()
        self.detailed = detailed

    def generate_caption(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            if self.detailed:
                captions = []

                # First pass
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=75,
                        num_beams=5,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                caption1 = self.processor.decode(out[0], skip_special_tokens=True).strip()
                captions.append(caption1)

                # Second pass
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
                            repetition_penalty=1.2
                        )
                    caption2 = self.processor.decode(out[0], skip_special_tokens=True).strip()
                    if caption2 and caption2 not in captions:
                        captions.append(caption2)
                except Exception:
                    pass

                # Third pass
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
                            top_p=0.9
                        )
                    caption3 = self.processor.decode(out[0], skip_special_tokens=True).strip()
                    if caption3 and caption3 not in captions:
                        captions.append(caption3)
                except Exception:
                    pass

                # Combine
                important_words = set()
                phrases = []
                stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

                for cap in captions:
                    if not cap:
                        continue
                    cap_clean = cap.lower().strip()
                    words = cap_clean.replace(',', ' ').replace('.', ' ').replace(':', ' ').split()
                    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
                    important_words.update(meaningful_words)
                    if len(cap.split()) > 3:
                        phrases.append(cap)

                if phrases:
                    base_caption = max(phrases, key=len)
                else:
                    base_caption = captions[0] if captions else ""

                if len(captions) > 1:
                    base_words = set(base_caption.lower().replace(',', ' ').replace('.', ' ').split())
                    additional_elements = []
                    for cap in captions:
                        if cap == base_caption:
                            continue
                        cap_words = set(cap.lower().replace(',', ' ').replace('.', ' ').split())
                        unique = cap_words - base_words
                        unique_meaningful = set([w for w in unique if w not in stop_words and len(w) > 2])
                        if unique_meaningful:
                            cap_phrases = [p.strip() for p in cap.split(',')]
                            for phrase in cap_phrases:
                                phrase_words = set(phrase.lower().split())
                                if phrase_words & unique_meaningful and len(phrase.split()) <= 5:
                                    if phrase not in additional_elements:
                                        additional_elements.append(phrase)
                    if additional_elements:
                        result = base_caption
                        for elem in additional_elements[:3]:
                            if elem.lower() not in result.lower():
                                result += ", " + elem
                        return result.strip()
                    else:
                        return base_caption
                else:
                    return base_caption
            else:
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
        path = Path(image_path)
        name = path.stem.replace("_", " ").replace("-", " ")
        return name


class DatasetPreparer:
    """Main class for preparing Flux LoRA datasets"""

    def __init__(self, tagger: ImageTagger, output_format: str = "both"):
        self.tagger = tagger
        self.output_format = output_format

    def convert_jpg_to_png(self, input_path: str, output_path: str) -> bool:
        try:
            img = Image.open(input_path).convert("RGB")
            img.save(output_path, "PNG", optimize=True)
            return True
        except Exception as e:
            print(f"Error converting {input_path} to PNG: {e}")
            return False

    def process_image(self, image_path: Path, output_dir: Path,
                     convert_to_png: bool = True) -> Dict:
        result = {
            "input_file": str(image_path),
            "success": False,
            "caption": "",
            "output_files": []
        }

        try:
            caption = self.tagger.generate_caption(str(image_path))
            result["caption"] = caption

            base_name = image_path.stem
            output_png = output_dir / f"{base_name}.png"
            output_txt = output_dir / f"{base_name}.txt"
            output_json = output_dir / f"{base_name}.json"

            if convert_to_png and image_path.suffix.lower() in ['.jpg', '.jpeg']:
                if self.convert_jpg_to_png(str(image_path), str(output_png)):
                    result["output_files"].append(str(output_png))
                else:
                    return result
            elif convert_to_png:
                img = Image.open(image_path).convert("RGB")
                img.save(output_png, "PNG", optimize=True)
                result["output_files"].append(str(output_png))

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
                         extensions: List[str] = None, progress_callback=None,
                         pause_callback=None, stop_callback=None) -> Dict:
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))

        results = []
        processed = 0
        failed = 0

        total = len(image_files)
        for i, image_path in enumerate(image_files, 1):
            if stop_callback and stop_callback():
                break
            if pause_callback:
                should_pause = pause_callback()
                # If pause_callback returns True, wait until unpaused.
                # This logic will only pause before processing the next file.
                while should_pause:
                    QThread.msleep(100)
                    should_pause = pause_callback()
            result = self.process_image(image_path, output_dir, convert_to_png)
            results.append(result)
            if result["success"]:
                processed += 1
            else:
                failed += 1

            if progress_callback is not None:
                progress_callback(i, total, result)

        summary = {
            "processed": processed,
            "failed": failed,
            "total": total,
            "results": results
        }

        return summary


class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int, int, dict)  # step, total, latest_result
    finished_signal = pyqtSignal(dict)  # whole summary
    stopped_signal = pyqtSignal(dict)   # emitted if stop was called
    paused_signal = pyqtSignal(bool)    # emit whether the process is paused or not

    def __init__(self, input_dir, output_dir, tagger, output_format, convert_to_png, detailed):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.tagger = tagger
        self.convert_to_png = convert_to_png
        self.detailed = detailed
        self._should_stop = False
        self._should_pause = False
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()

    def pause(self):
        self._mutex.lock()
        self._should_pause = True
        self._mutex.unlock()
        self.paused_signal.emit(True)

    def resume(self):
        self._mutex.lock()
        self._should_pause = False
        self._wait_condition.wakeAll()
        self._mutex.unlock()
        self.paused_signal.emit(False)

    def stop(self):
        self._should_stop = True
        self.resume()  # In case we are paused, wake the thread so it can actually stop

    def is_paused(self):
        self._mutex.lock()
        paused = self._should_pause
        self._mutex.unlock()
        return paused

    def is_stopped(self):
        return self._should_stop

    def run(self):
        def pause_callback():
            # Returns True if should pause, else False
            self._mutex.lock()
            should_pause = self._should_pause
            self._mutex.unlock()
            if should_pause:
                self.paused_signal.emit(True)
                # Now wait until unpaused or stopped
                self._mutex.lock()
                try:
                    while self._should_pause and not self._should_stop:
                        self._wait_condition.wait(self._mutex, 200)
                finally:
                    self._mutex.unlock()
                self.paused_signal.emit(False)
            return should_pause

        def stop_callback():
            return self._should_stop

        preparer = DatasetPreparer(self.tagger, output_format=self.output_format)
        summary = preparer.process_directory(
            self.input_dir,
            self.output_dir,
            convert_to_png=self.convert_to_png,
            progress_callback=self.emit_progress,
            pause_callback=pause_callback,
            stop_callback=stop_callback
        )
        if self._should_stop:
            self.stopped_signal.emit(summary)
        else:
            self.finished_signal.emit(summary)

    def emit_progress(self, step, total, latest_result):
        self.progress_signal.emit(step, total, latest_result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux LoRA Dataset Preparation Tool")

        layout = QVBoxLayout()

        # Input directory
        input_layout = QHBoxLayout()
        self.input_dir_edit = QLineEdit(self)
        self.input_dir_button = QPushButton("Browse...", self)
        self.input_dir_button.clicked.connect(self.select_input_dir)
        input_layout.addWidget(QLabel("Dataset Source Folder: "))
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(self.input_dir_button)
        layout.addLayout(input_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self)
        self.output_dir_button = QPushButton("Browse...", self)
        self.output_dir_button.clicked.connect(self.select_output_dir)
        output_layout.addWidget(QLabel("Processed Output Folder: "))
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_button)
        layout.addLayout(output_layout)

        # Tagger and options
        options_layout = QHBoxLayout()
        self.tagger_combo = QComboBox(self)
        self.tagger_combo.addItems(["blip", "simple"])
        options_layout.addWidget(QLabel("Tagger: "))
        options_layout.addWidget(self.tagger_combo)

        self.detailed_checkbox = QCheckBox("Detailed BLIP (multi-pass)")
        self.detailed_checkbox.setChecked(True)
        options_layout.addWidget(self.detailed_checkbox)
        layout.addLayout(options_layout)

        # Format
        format_layout = QHBoxLayout()
        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["both", "txt", "json"])
        format_layout.addWidget(QLabel("Caption Output Format: "))
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # PNG Conversion
        self.convert_png_checkbox = QCheckBox("Convert to PNG (recommended)")
        self.convert_png_checkbox.setChecked(True)
        layout.addWidget(self.convert_png_checkbox)

        # Process, Pause/Resume, and Stop buttons
        btn_layout = QHBoxLayout()
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.process_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.pause_processing)
        btn_layout.addWidget(self.pause_button)

        self.resume_button = QPushButton("Resume")
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self.resume_processing)
        btn_layout.addWidget(self.resume_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)
        btn_layout.addWidget(self.stop_button)
        layout.addLayout(btn_layout)

        # Progress bar and log
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(140)
        layout.addWidget(self.log_text)

        # Central widget setup
        central = QWidget(self)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.processing_thread = None

        self.update_options()

        self.tagger_combo.currentTextChanged.connect(self.update_options)

        if not BLIP_AVAILABLE:
            self.tagger_combo.setItemText(
                0,
                "blip (requires transformers/torch)"
            )
            if self.tagger_combo.currentIndex() == 0:
                self.tagger_combo.setCurrentIndex(1)

    def update_options(self):
        tagger = self.tagger_combo.currentText()
        if "blip" in tagger:
            self.detailed_checkbox.setEnabled(True)
        else:
            self.detailed_checkbox.setEnabled(False)
            self.detailed_checkbox.setChecked(False)

    def select_input_dir(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Dataset Source Folder", "")
        if dir:
            self.input_dir_edit.setText(dir)

    def select_output_dir(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if dir:
            self.output_dir_edit.setText(dir)

    def start_processing(self):
        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        tagger_name = self.tagger_combo.currentText().split(' ')[0]
        output_format = self.format_combo.currentText()
        convert_to_png = self.convert_png_checkbox.isChecked()
        detailed = self.detailed_checkbox.isChecked()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.critical(self, "Error", "Please specify a valid dataset source folder.")
            return
        if not output_dir:
            QMessageBox.critical(self, "Error", "Please specify a valid output folder.")
            return

        # Check BLIP prerequisites
        if tagger_name == "blip":
            if not BLIP_AVAILABLE:
                QMessageBox.critical(self, "BLIP Not Available", 
                    "transformers/torch libraries required. Please install with:\n\npip install transformers torch")
                return
            try:
                tagger = BLIPTagger(device="auto", detailed=detailed)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize BLIP tagger:\n{e}")
                return
        else:
            tagger = SimpleTagger(device="auto")

        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.processing_thread = ProcessingThread(
            input_dir, output_dir, tagger, output_format, convert_to_png, detailed
        )
        self.processing_thread.progress_signal.connect(self.on_progress)
        self.processing_thread.finished_signal.connect(self.on_finished)
        self.processing_thread.stopped_signal.connect(self.on_stopped)
        self.processing_thread.paused_signal.connect(self.on_paused)
        self.processing_thread.start()

        self.log_text.append("Started processing dataset...")

    def pause_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.log_text.append("(Paused)")

    def resume_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.log_text.append("(Resumed)")

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.log_text.append("(Stopping...)")
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)

    def on_progress(self, step, total, latest_result):
        percent = int((step / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        fn = os.path.basename(latest_result.get("input_file", ""))
        if latest_result.get("success"):
            self.log_text.append(f'[✓] {fn} — Caption: {latest_result["caption"][:80]}')
        else:
            err = latest_result.get("error", "Unknown error")
            self.log_text.append(f'[X] {fn} — Error: {err}')

    def on_finished(self, summary):
        self.progress_bar.setValue(100)
        self.process_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        msg = (
            "<b>Processing complete!</b><br/>"
            f"Total images: {summary['total']}<br/>"
            f"Successfully processed: {summary['processed']}<br/>"
            f"Failed: {summary['failed']}<br/>"
        )
        self.log_text.append("\n" + "="*32 + "\n" + msg + "="*32)
        QMessageBox.information(self, "Processing Complete", msg)

    def on_stopped(self, summary):
        self.process_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        msg = (
            "<b>Processing stopped.</b><br/>"
            f"Total attempted: {summary['total']}<br/>"
            f"Successfully processed: {summary['processed']}<br/>"
            f"Failed: {summary['failed']}<br/>"
        )
        self.log_text.append("\n" + "="*32 + "\n" + msg + "="*32)
        QMessageBox.information(self, "Process Stopped", msg)

    def on_paused(self, is_paused):
        # This slot can be used to update UI, if needed
        if is_paused:
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
        else:
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(650, 500)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
