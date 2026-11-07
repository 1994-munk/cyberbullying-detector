# Cyberbully Detection Python Project

## Build
- Replace the home placeholder with a polished, beginner-friendly interactive lab named **Cyberbully Detection**.
- Create five guided stages: inspect the uploaded dataset, clean and audit labels, split without duplicate leakage, train a TF-IDF/logistic-regression baseline, and evaluate/save/use the model.
- Show measured facts from the uploaded 20,001-row JSON-lines file, explain labels as cyber-aggressive vs. non-aggressive, and flag that this is not a definitive bullying judgment.
- Provide clearly commented, runnable Python project files and setup instructions for download, without redistributing the uploaded dataset.
- Use the existing design system with a distinct research-lab visual style and add page-specific metadata.

## Technical details
- Add downloadable Python script, requirements, and README under the app's public project files; the script reads the supplied JSON-lines format, audits and removes exact duplicates and conflicting labels, trains a stratified baseline, prints classification metrics, saves the model, and includes a prediction example.
- Build the lesson navigation and code-copy/download interactions on `/`, style through semantic tokens in `src/styles.css`, update `AGENTS.md` with the structural choice, and verify the app and existing routing test.
