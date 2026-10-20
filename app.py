"""Local desktop application. Images are processed on this computer."""
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from PIL import Image, ImageOps, ImageTk
from core import LABELS, load_bundle, predict, preprocess

ROOT = Path(__file__).resolve().parent


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MammoScope | BI-RADS research application")
        self.geometry("1080x760")
        self.minsize(900, 680)
        self.configure(bg="#eef3f7")
        self.model = self.meta = self.image_path = self.result = None
        self.busy = False
        self.events = queue.Queue()
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef3f7")
        style.configure("TLabel", background="#eef3f7", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"), foreground="#14354a")
        style.configure("Result.TLabel", font=("Segoe UI", 18, "bold"), foreground="#176c78")
        shell = ttk.Frame(self, padding=24)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="MammoScope", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, text="CNN / ResNet50 • BI-RADS 1–5 • Local research application").pack(anchor="w", pady=(0, 12))
        banner = tk.Label(shell, text="RESEARCH ONLY | Not clinically validated. Do not use for diagnosis or treatment decisions.",
                          bg="#fff2d5", fg="#73511b", padx=12, pady=12, font=("Segoe UI", 10))
        banner.pack(fill="x", pady=(0, 14))
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        analysis, training, about = [ttk.Frame(self.tabs, padding=18) for _ in range(3)]
        for frame, label in ((analysis, "  Image analysis  "), (training, "  Model training  "), (about, "  Methods and help  ")):
            self.tabs.add(frame, text=label)
        self.status = tk.StringVar(value="No model loaded. Load a trained model to enable predictions.")
        ttk.Label(analysis, textvariable=self.status, wraplength=940).pack(anchor="w", pady=(0, 12))
        toolbar = ttk.Frame(analysis)
        toolbar.pack(fill="x")
        self.load_button = ttk.Button(toolbar, text="Load model folder", command=self.choose_model)
        self.load_button.pack(side="left", padx=(0, 8))
        self.image_button = ttk.Button(toolbar, text="Upload image", command=self.choose_image)
        self.image_button.pack(side="left", padx=(0, 8))
        self.predict_button = ttk.Button(toolbar, text="Predict", command=self.run_prediction, state="disabled")
        self.predict_button.pack(side="left", padx=(0, 8))
        self.export_button = ttk.Button(toolbar, text="Save result", command=self.export_result, state="disabled")
        self.export_button.pack(side="left")
        columns = ttk.Frame(analysis)
        columns.pack(fill="both", expand=True, pady=18)
        self.preview = tk.Label(columns, text="Upload a mammogram\nPNG / JPG / JPEG / BMP", bg="#dfe8ee", fg="#587080", width=42)
        self.preview.pack(side="left", fill="both", expand=True, padx=(0, 24))
        results = ttk.Frame(columns, width=400)
        results.pack(side="left", fill="both", expand=True)
        self.result_label = ttk.Label(results, text="No prediction yet", style="Result.TLabel")
        self.result_label.pack(anchor="w", pady=(0, 12))
        self.score_vars, self.bars = [], []
        for label in LABELS:
            var = tk.StringVar(value=f"{label}   —")
            ttk.Label(results, textvariable=var).pack(anchor="w", pady=(6, 3))
            bar = ttk.Progressbar(results, maximum=1)
            bar.pack(fill="x")
            self.score_vars.append(var)
            self.bars.append(bar)
        ttk.Label(results, text="Scores describe model outputs; they are not cancer probabilities.\nNot intended for ordinary photos, ultrasound or MRI.\nThe application cannot verify that an image is a mammogram.",
                  wraplength=390).pack(anchor="w", pady=14)
        self.confirmed = tk.BooleanVar(value=False)
        ttk.Checkbutton(analysis, text="This is a de-identified mammogram of the same modality as the training data.",
                        variable=self.confirmed, command=self.refresh).pack(anchor="w")
        ttk.Label(training, text="1. Select a labeled CSV manifest  →  2. Validate  →  3. Train", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 12))
        ttk.Label(training, text="CSV: path,birads,patient_id,split • Classes: 1–5 • Splits: train, val, test\nEach split must include every class. Patients must not overlap across splits.").pack(anchor="w")
        row = ttk.Frame(training)
        row.pack(fill="x", pady=12)
        self.manifest = tk.StringVar(value=str(ROOT / "data" / "manifest.csv"))
        ttk.Entry(row, textvariable=self.manifest).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Select CSV", command=self.choose_manifest).pack(side="left", padx=8)
        options = ttk.Frame(training)
        options.pack(fill="x")
        self.architecture = tk.StringVar(value='ResNet50')
        chooser = ttk.Combobox(options, textvariable=self.architecture, values=['ResNet50', 'Paper CNN'], state='readonly', width=12)
        chooser.pack(side='left', padx=(0,8))
        chooser.bind('<<ComboboxSelected>>', lambda event: self.epochs.set('40' if self.architecture.get() == 'ResNet50' else '10'))
        ttk.Label(options, text="Epoch:").pack(side="left")
        self.epochs = tk.StringVar(value="40")
        ttk.Spinbox(options, from_=1, to=200, textvariable=self.epochs, width=5).pack(side="left", padx=8)
        self.validate_button = ttk.Button(options, text="Validate data", command=self.validate_data)
        self.validate_button.pack(side="left", padx=8)
        self.train_button = ttk.Button(options, text="Start training", command=self.start_training)
        self.train_button.pack(side="left")
        self.log_box = tk.Text(training, height=14, bg="#132f41", fg="#e5f1f4", font=("Consolas", 10), wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(14, 0))
        self.append_log("See data/cdd_cesm/preparation_report.json for data preparation and RESULTS.md for experiment results.")
        info = tk.Text(about, wrap="word", font=("Segoe UI", 11), bg="#eef3f7", relief="flat")
        info.pack(fill="both", expand=True)
        info.insert("1.0", (ROOT / "README.md").read_text(encoding="utf-8"))
        info.configure(state="disabled")
        ttk.Label(shell, text="Images stay on your computer. Predictions require a trained model.", foreground="#597181").pack(anchor="w", pady=(12, 0))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.poll)
        self.after(250, self.load_default)

    def load_default(self):
        active = ROOT / 'models' / 'active_model.json'
        folder = 'cdd_cesm_cnn'
        if active.is_file():
            try:
                folder = json.loads(active.read_text(encoding='utf-8'))['folder']
                if Path(folder).name != folder:
                    raise ValueError('Invalid model folder')
            except (ValueError, KeyError):
                self.status.set('Invalid default model configuration; use Load model folder.')
                return
        path = ROOT / 'models' / folder
        if (path / 'metadata.json').is_file():
            self.status.set('Loading the default model…')
            self.work(lambda: load_bundle(path), self.model_ready)

    def append_log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def refresh(self):
        for button in (self.load_button, self.image_button, self.validate_button, self.train_button):
            button.configure(state="disabled" if self.busy else "normal")
        ready = self.model is not None and self.image_path is not None and self.confirmed.get() and not self.busy
        self.predict_button.configure(state="normal" if ready else "disabled")
        self.export_button.configure(state="normal" if self.result and not self.busy else "disabled")

    def clear_result(self):
        self.result = None
        self.result_label.configure(text="No prediction yet")
        for label, var, bar in zip(LABELS, self.score_vars, self.bars):
            var.set(label + "   —")
            bar["value"] = 0
        self.refresh()

    def work(self, fn, done):
        if self.busy:
            return
        self.busy = True
        self.refresh()
        def worker():
            try:
                self.events.put(("done", done, fn()))
            except Exception as error:
                self.events.put(("error", str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "log":
                    self.append_log(event[1])
                elif event[0] == "error":
                    self.busy = False
                    if self.model is None:
                        self.status.set("No model loaded. A valid training bundle is required.")
                    self.append_log("ERROR: " + event[1])
                    messagebox.showerror("Operation failed", event[1])
                    self.refresh()
                else:
                    self.busy = False
                    event[1](event[2])
                    self.refresh()
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def choose_manifest(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.manifest.set(path)

    def choose_model(self):
        path = filedialog.askdirectory(title="Select a trusted model bundle created with this application", initialdir=ROOT / "models")
        if path:
            self.model = self.meta = None
            self.status.set("Loading model…")
            self.clear_result()
            self.work(lambda: load_bundle(path), self.model_ready)

    def model_ready(self, bundle):
        self.model, self.meta = bundle
        metrics = self.meta.get('metrics', {})
        acc = metrics.get('test_accuracy')
        accuracy_text = f" • Held-out test accuracy: {100*acc:.1f}%" if acc is not None else ''
        family = 'ResNet50' if 'resnet50' in self.meta.get('architecture','') else 'CNN'
        self.status.set(f"{family} ready • BI-RADS 1–5" + accuracy_text + " • Limited performance; no clinical validation")
        self.clear_result()

    def choose_image(self):
        path = filedialog.askopenfilename(initialdir=ROOT / 'examples', filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        self.clear_result()
        try:
            preprocess(path)
            with Image.open(path) as source:
                image = ImageOps.contain(source.convert("RGB"), (390, 310))
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="", width=390)
            self.image_path = path
            self.confirmed.set(False)
        except Exception as error:
            self.image_path = None
            self.preview.configure(image="", text="Upload imagenemedi")
            messagebox.showerror("Invalid image", str(error))
        self.refresh()

    def run_prediction(self):
        if self.model is None or not self.image_path or not self.confirmed.get() or self.busy:
            return
        self.clear_result()
        self.work(lambda: predict(self.model, self.image_path), self.show_result)

    def show_result(self, result):
        self.result = result | {"model_sha256": self.meta["model_sha256"],
                                "test_metrics": self.meta.get('metrics'),
                                "created_local": datetime.now().astimezone().isoformat()}
        self.result_label.configure(text="Model prediction: " + result["label"])
        for label, var, bar in zip(LABELS, self.score_vars, self.bars):
            score = result["scores"][label]
            var.set(f"{label}    {score * 100:.2f}%")
            bar["value"] = score

    def export_result(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="research_result.json", filetypes=[("JSON", "*.json")])
        if path:
            Path(path).write_text(json.dumps(self.result, ensure_ascii=False, indent=2), encoding="utf-8")

    def validate_data(self):
        from core import read_manifest
        path = self.manifest.get()
        self.work(lambda: read_manifest(path), lambda rows: self.append_log(
            "Validation passed: " + str({s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')})))

    def start_training(self):
        try:
            epochs = int(self.epochs.get())
            if not 1 <= epochs <= 200:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Epoch", "Enter an integer from 1 to 200.")
            return
        path = self.manifest.get()
        architecture = self.architecture.get()
        output = ROOT / "models" / datetime.now().strftime("training_%Y%m%d_%H%M%S_%f")
        def run():
            if architecture == 'ResNet50':
                from train_resnet import train_resnet
                train_resnet(path, output, epochs, log=lambda text: self.events.put(("log", text)))
            else:
                from train import train
                train(path, output, epochs=epochs, balanced=True, log=lambda text: self.events.put(("log", text)))
            return load_bundle(output)
        self.work(run, self.model_ready)

    def close(self):
        if self.busy:
            if not messagebox.askyesno("Operation in progress", "Closing will interrupt the running operation. Close anyway?"):
                return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
