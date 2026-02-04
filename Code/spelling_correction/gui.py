"""
Spelling Correction GUI — tkinter two-pane application.

Layout:
  LEFT:   Text editor (500 char limit) with red-underlined error highlighting
  RIGHT:  Dictionary search panel
  BOTTOM: Status bar

Run:
    python Code/spelling_correction/gui.py
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)                        # Code/
sys.path.insert(0, _CODE_ROOT)

from spelling_correction.corpus import CorpusProcessor
from spelling_correction.dictionary import Dictionary
from spelling_correction.bigram_model import BigramModel
from spelling_correction.corrector import SpellingCorrector, tokenize_with_positions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CHARS = 500
DICT_LIST_MAX = 1000


# ---------------------------------------------------------------------------
# Helper: character offset → tkinter "line.col" index
# ---------------------------------------------------------------------------
def _offset_to_index(text_widget, offset):
    """Convert a 0-based character offset to a tkinter Text index string."""
    content = text_widget.get("1.0", "end-1c")
    line = 1
    col = 0
    for i, ch in enumerate(content):
        if i == offset:
            return f"{line}.{col}"
        if ch == '\n':
            line += 1
            col = 0
        else:
            col += 1
    # offset == len(content)  →  end of text
    return f"{line}.{col}"


# ===========================================================================
# Main application class
# ===========================================================================
class SpellingApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Spelling Correction System — CT052-3-M-NLP")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f0f0f0")

        # Backend objects (initialised after splash)
        self.corrector = None
        self.dictionary = None

        # Active error tags — tracked so we can remove them on re-scan
        self._error_tags = []

        # --- Show loading splash BEFORE heavy init ---
        self._show_splash()
        self.root.update()          # force repaint
        self._init_backend()
        self._destroy_splash()

        # --- Build UI ---
        self._build_ui()

    # ----------------------------------------------------------------------
    # Splash / loading
    # ----------------------------------------------------------------------
    def _show_splash(self):
        self._splash_label = tk.Label(
            self.root, text="Loading corpus … please wait …",
            font=("Helvetica", 16), bg="#f0f0f0", fg="#333333"
        )
        self._splash_label.place(relx=0.5, rely=0.5, anchor="center")

    def _destroy_splash(self):
        self._splash_label.destroy()

    # ----------------------------------------------------------------------
    # Backend initialisation
    # ----------------------------------------------------------------------
    def _init_backend(self):
        corpus = CorpusProcessor()
        corpus.build()
        self.dictionary = Dictionary(corpus)
        bigram_model = BigramModel(corpus)
        self.corrector = SpellingCorrector(self.dictionary, corpus, bigram_model)

    # ----------------------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------------------
    def _build_ui(self):
        # --- Top-level frames ---
        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # LEFT pane
        left_frame = tk.LabelFrame(top_frame, text="Text Editor",
                                   font=("Helvetica", 11, "bold"),
                                   bg="#f0f0f0", fg="#333333", bd=2)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.text_editor = tk.Text(
            left_frame, font=("Courier", 12), width=55, height=28,
            wrap="word", undo=True, bg="white", fg="#222222",
            insertbackground="#333333", relief="flat", bd=8
        )
        self.text_editor.pack(fill="both", expand=True)

        # Character counter
        self.char_label = tk.Label(
            left_frame, text="0 / 500 characters",
            font=("Helvetica", 9), bg="#f0f0f0", fg="#666666", anchor="w"
        )
        self.char_label.pack(fill="x", padx=8, pady=(0, 4))

        # RIGHT pane — dictionary
        right_frame = tk.LabelFrame(top_frame, text="Dictionary",
                                    font=("Helvetica", 11, "bold"),
                                    bg="#f0f0f0", fg="#333333", bd=2,
                                    width=300)
        right_frame.pack(side="right", fill="y", expand=False, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Search entry
        search_row = tk.Frame(right_frame, bg="#f0f0f0")
        search_row.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(search_row, text="Search:", font=("Helvetica", 10),
                 bg="#f0f0f0", fg="#444444").pack(side="left")
        self.dict_search_var = tk.StringVar()
        self.dict_search_entry = tk.Entry(search_row, textvariable=self.dict_search_var,
                                          font=("Helvetica", 10), bd=2, relief="groove")
        self.dict_search_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.dict_search_var.trace_add("write", self._on_dict_search)

        # Word count label
        self.dict_count_label = tk.Label(
            right_frame, text=f"Words: {self.dictionary.get_word_count():,}",
            font=("Helvetica", 9), bg="#f0f0f0", fg="#666666", anchor="w"
        )
        self.dict_count_label.pack(fill="x", padx=8)

        # Listbox + scrollbar
        list_frame = tk.Frame(right_frame, bg="#f0f0f0")
        list_frame.pack(fill="both", expand=True, padx=6, pady=4)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.dict_listbox = tk.Listbox(
            list_frame, font=("Courier", 10), yscrollcommand=scrollbar.set,
            bd=2, relief="groove", activestyle="none"
        )
        self.dict_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.dict_listbox.yview)

        # Populate dictionary listbox (first 1000 words)
        self._populate_dict_list("")

        # --- Status bar ---
        self.status_bar = tk.Label(
            self.root, text="Type to begin …",
            font=("Helvetica", 10), bg="#e8e8e8", fg="#444444",
            anchor="w", relief="flat", bd=0
        )
        self.status_bar.pack(fill="x", side="bottom", padx=10, pady=(0, 8))

        # --- Bind events ---
        self.text_editor.bind("<KeyRelease>", self._on_key_release)

    # ----------------------------------------------------------------------
    # Dictionary panel helpers
    # ----------------------------------------------------------------------
    def _populate_dict_list(self, prefix):
        words = self.dictionary.search(prefix)
        self.dict_listbox.delete(0, tk.END)
        for w in words[:DICT_LIST_MAX]:
            self.dict_listbox.insert(tk.END, w)
        self.dict_count_label.config(
            text=f"Showing {len(words[:DICT_LIST_MAX]):,} / {self.dictionary.get_word_count():,}"
        )

    def _on_dict_search(self, *_args):
        self._populate_dict_list(self.dict_search_var.get().strip())

    # ----------------------------------------------------------------------
    # Text-editor event handlers
    # ----------------------------------------------------------------------
    def _on_key_release(self, _event=None):
        # Enforce 500-char limit
        content = self.text_editor.get("1.0", "end-1c")
        if len(content) > MAX_CHARS:
            # Keep cursor roughly where it was
            idx = self.text_editor.index("insert")
            content = content[:MAX_CHARS]
            self.text_editor.delete("1.0", "end")
            self.text_editor.insert("1.0", content)
            self.text_editor.mark_set("insert", idx)

        # Update char counter
        content = self.text_editor.get("1.0", "end-1c")
        self.char_label.config(text=f"{len(content)} / {MAX_CHARS} characters")

        # Re-scan for errors
        self._scan_errors()

    # ----------------------------------------------------------------------
    # Error scanning & highlighting
    # ----------------------------------------------------------------------
    def _scan_errors(self):
        # Remove previous error tags
        for tag in self._error_tags:
            self.text_editor.tag_delete(tag)
        self._error_tags.clear()

        content = self.text_editor.get("1.0", "end-1c")
        if not content.strip():
            self.status_bar.config(text="Type to begin …")
            return

        errors = self.corrector.detect_errors(content)
        error_count = 0

        tokens = tokenize_with_positions(content)
        for i, (token, error) in enumerate(zip(tokens, errors)):
            if error is None:
                continue
            error_count += 1

            tag_name = f"error_{i}"
            self._error_tags.append(tag_name)

            start_idx = _offset_to_index(self.text_editor, token['start'])
            end_idx   = _offset_to_index(self.text_editor, token['end'])

            # Style: red foreground + underline
            self.text_editor.tag_configure(
                tag_name, foreground="red", underline=True
            )
            self.text_editor.tag_add(tag_name, start_idx, end_idx)

            # Bind click — use default-arg closure to capture current values
            self.text_editor.tag_bind(
                tag_name, "<Button-1>",
                lambda evt, w=error['word'], s=token['start'],
                       e=token['end'], et=error['error_type'], idx=i:
                    self._open_suggestion_popup(w, s, e, et, idx)
            )

        # Status bar
        if error_count == 0:
            self.status_bar.config(text="No errors found")
        else:
            self.status_bar.config(text=f"{error_count} error(s) detected — click a red word for suggestions")

    # ----------------------------------------------------------------------
    # Suggestion popup
    # ----------------------------------------------------------------------
    def _open_suggestion_popup(self, word, start_offset, end_offset, error_type, token_index):
        """Open a Toplevel with ranked correction suggestions."""
        # Build context word list for ranking
        content = self.text_editor.get("1.0", "end-1c")
        tokens = tokenize_with_positions(content)
        context_words = [t['word'] for t in tokens]

        # Get ranked suggestions
        candidates = self.corrector.get_candidates(word)
        suggestions = self.corrector.rank_candidates(
            word, candidates, context_words, token_index
        )

        # --- Popup window ---
        popup = tk.Toplevel(self.root)
        popup.title(f"Corrections for '{word}'")
        popup.geometry("380x320")
        popup.grab_set()                    # modal
        popup.resizable(False, False)
        popup.configure(bg="#f5f5f5")

        tk.Label(popup, text=f"Corrections for  \"{word}\"",
                 font=("Helvetica", 13, "bold"), bg="#f5f5f5", fg="#222222"
                 ).pack(pady=(12, 2))

        # Error-type badge
        badge_color = "#d9534f" if error_type == "non_word" else "#f0ad4e"
        badge_text  = "Non-word error" if error_type == "non_word" else "Real-word error (context)"
        tk.Label(popup, text=badge_text, font=("Helvetica", 9, "italic"),
                 bg=badge_color, fg="white", padx=8, pady=2
                 ).pack(pady=(0, 8))

        if not suggestions:
            tk.Label(popup, text="No suggestions available.",
                     font=("Helvetica", 11), bg="#f5f5f5", fg="#888888"
                     ).pack(pady=20)
        else:
            # Scrollable frame for suggestions
            container = tk.Frame(popup, bg="#f5f5f5")
            container.pack(fill="both", expand=True, padx=12, pady=4)

            canvas = tk.Canvas(container, bg="#f5f5f5", highlightthickness=0)
            scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas, bg="#f5f5f5")
            scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for sug in suggestions:
                d = sug['distances']
                label_text = (
                    f"{sug['word']:<18}"
                    f"Lev: {d['levenshtein']}   "
                    f"D-L: {d['damerau_levenshtein']}   "
                    f"Wtd: {d['weighted']:.1f}"
                )
                btn = tk.Button(
                    scroll_frame, text=label_text,
                    font=("Courier", 11), anchor="w",
                    bg="white", fg="#222222", activebackground="#e2e2e2",
                    relief="groove", bd=2, padx=8, pady=4, cursor="hand2",
                    command=lambda s=sug['word'], p=popup:
                        self._apply_correction(start_offset, end_offset, s, p)
                )
                btn.pack(fill="x", pady=2)

        # Close button
        tk.Button(popup, text="Close", font=("Helvetica", 10),
                  bg="#cccccc", fg="#333333", relief="flat", bd=0,
                  padx=16, pady=4, cursor="hand2",
                  command=popup.destroy
                  ).pack(pady=(8, 12))

    # ----------------------------------------------------------------------
    # Apply a correction
    # ----------------------------------------------------------------------
    def _apply_correction(self, start_offset, end_offset, replacement, popup):
        """Replace the word at [start_offset, end_offset) and re-scan."""
        start_idx = _offset_to_index(self.text_editor, start_offset)
        end_idx   = _offset_to_index(self.text_editor, end_offset)
        self.text_editor.delete(start_idx, end_idx)
        self.text_editor.insert(start_idx, replacement)
        popup.destroy()
        self._scan_errors()


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    root = tk.Tk()
    app = SpellingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
