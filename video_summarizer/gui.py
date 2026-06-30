"""Desktop GUI for video lesson summarizer."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from video_summarizer.config import Settings, get_settings
from video_summarizer.runner import process_video
from video_summarizer.transcribe import format_transcript_plain, load_transcript


class VideoSummarizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Lesson Summarizer")
        self.root.minsize(960, 640)
        self.root.geometry("1100x760")

        self.settings = get_settings()
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(self.settings.output_dir))
        self.max_duration = tk.StringVar()
        self.status = tk.StringVar(value="Pronto")
        self.work_dir: Path | None = None
        self._worker: threading.Thread | None = None
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_ui()
        self._poll_events()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        input_row = ttk.LabelFrame(frame, text="Input video", padding=8)
        input_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Entry(input_row, textvariable=self.video_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(input_row, text="Sfoglia…", command=self._browse_video).pack(side=tk.LEFT)

        options_row = ttk.Frame(frame)
        options_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(options_row, text="Cartella output:").pack(side=tk.LEFT)
        ttk.Entry(options_row, textvariable=self.output_dir, width=36).pack(
            side=tk.LEFT, padx=(6, 8)
        )
        ttk.Button(options_row, text="Sfoglia…", command=self._browse_output).pack(side=tk.LEFT)

        ttk.Label(options_row, text="Max secondi:").pack(side=tk.LEFT, padx=(16, 6))
        ttk.Entry(options_row, textvariable=self.max_duration, width=8).pack(side=tk.LEFT)

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(0, 8))

        self.process_btn = ttk.Button(
            actions, text="Elabora video", command=self._start_processing
        )
        self.process_btn.pack(side=tk.LEFT)

        ttk.Button(actions, text="Carica job esistente", command=self._load_existing).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="Apri cartella output", command=self._open_output).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.transcript_view = scrolledtext.ScrolledText(
            notebook, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.appunti_view = scrolledtext.ScrolledText(
            notebook, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.log_view = scrolledtext.ScrolledText(
            notebook, wrap=tk.WORD, font=("Consolas", 9), height=8
        )

        notebook.add(self.transcript_view, text="Trascrizione")
        notebook.add(self.appunti_view, text="Appunti (appunti.md)")
        notebook.add(self.log_view, text="Log")

        status_bar = ttk.Label(frame, textvariable=self.status, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(8, 0))

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[
                ("Video", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("Tutti i file", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Cartella output")
        if path:
            self.output_dir.set(path)

    def _open_output(self) -> None:
        target = self.work_dir or Path(self.output_dir.get())
        if not target.exists():
            messagebox.showinfo("Output", "Nessuna cartella output disponibile.")
            return
        import os
        import subprocess
        import sys

        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target)], check=False)
        else:
            subprocess.run(["xdg-open", str(target)], check=False)

    def _load_existing(self) -> None:
        path = filedialog.askdirectory(title="Seleziona cartella job (output/<video>)")
        if not path:
            return
        self._load_results(Path(path))

    def _set_text(self, widget: scrolledtext.ScrolledText, content: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.configure(state=tk.DISABLED)

    def _append_log(self, message: str) -> None:
        self.log_view.configure(state=tk.NORMAL)
        self.log_view.insert(tk.END, message + "\n")
        self.log_view.see(tk.END)
        self.log_view.configure(state=tk.DISABLED)

    def _load_results(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        transcript_path = work_dir / "transcript.json"
        transcript_md_path = work_dir / "transcript.md"
        appunti_path = work_dir / "appunti.md"

        if transcript_path.exists():
            transcript = load_transcript(transcript_path)
            content = (
                transcript_md_path.read_text(encoding="utf-8")
                if transcript_md_path.exists()
                else format_transcript_plain(transcript)
            )
            self._set_text(self.transcript_view, content)
        else:
            self._set_text(self.transcript_view, "Trascrizione non ancora disponibile.")

        if appunti_path.exists():
            self._set_text(self.appunti_view, appunti_path.read_text(encoding="utf-8"))
        else:
            self._set_text(
                self.appunti_view,
                "Appunti non ancora disponibili. Attendi il completamento della summarization.",
            )

        self.status.set(f"Output: {work_dir}")

    def _parse_max_duration(self) -> int | None:
        raw = self.max_duration.get().strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            raise ValueError("Max secondi deve essere un numero intero") from None
        if value <= 0:
            raise ValueError("Max secondi deve essere maggiore di zero")
        return value

    def _start_processing(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("In corso", "Elaborazione già in esecuzione.")
            return

        video = Path(self.video_path.get().strip())
        if not video.exists():
            messagebox.showerror("Errore", "Seleziona un file video valido.")
            return

        try:
            max_duration = self._parse_max_duration()
        except ValueError as exc:
            messagebox.showerror("Errore", str(exc))
            return

        output_dir = Path(self.output_dir.get().strip() or "output")
        settings = Settings(**{**self.settings.model_dump(), "output_dir": output_dir})

        self._set_text(self.transcript_view, "")
        self._set_text(self.appunti_view, "")
        self._set_text(self.log_view, "")
        self.process_btn.configure(state=tk.DISABLED)
        self.status.set("Elaborazione in corso…")
        self._append_log(f"Avvio: {video}")

        def worker() -> None:
            try:
                work_dir = process_video(
                    video,
                    settings=settings,
                    max_duration=max_duration,
                    force=False,
                    output_dir=output_dir,
                )
                self._events.put(("done", work_dir))
            except Exception as exc:
                self._events.put(("error", str(exc)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "done":
                    self._load_results(payload)
                    self._append_log(f"Completato: {payload}")
                    self.status.set(f"Completato — {payload}")
                elif kind == "error":
                    self._append_log(f"Errore: {payload}")
                    self.status.set("Errore durante l'elaborazione")
                    messagebox.showerror("Errore", str(payload))
                self.process_btn.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_events)


def run_gui() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass
    VideoSummarizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
