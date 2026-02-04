---
skill: run-spell-gui
description: Launch the spelling correction GUI application
---

# Run Spelling Correction GUI

Launch the spelling correction system with its graphical user interface.

## Steps

1. Check if the GUI file exists at Code/spelling_correction/gui.py
2. Verify virtual environment is activated
3. Check if required dependencies are installed
4. Run the GUI application
5. If the file doesn't exist, inform user and provide guidance on implementation status

## Notes

- The GUI should support up to 500 characters of text
- Should provide clickable misspelled words
- Should display correction suggestions with edit distances
- Requires corpus data to be available

## Error Handling

- If gui.py doesn't exist, check implementation status
- If dependencies missing, prompt to run setup-env skill
- If corpus data missing, guide user to prepare corpus
