Completed run: bbc_12masks_20261101T171020Z
700 articles: 350 training and 350 excluded controls. 5,582 common windows.
All six result CSVs have exactly: document,window_number,mask,p_z.
LLaDA has separate threshold and low-confidence files; 10 draws per estimate.
Each LLaDA file has 66,984 rows; each AR model has 5,582 rows.
document = group_filteredIndex, e.g. train_100; the index refers to the frozen
March-June 2025 filtered dataset, not the selected-cohort row number.
document_manifest.csv maps group + filtered_index to source metadata and hashes.
window_number retains the original ZERO-BASED word-start window_index.
Windows were sampled from each article; do not renumber them consecutively.
mask strings are the user-supplied labels in notebook order; abbreviated labels
represent the original full 50-position masks, not truncated literal masks.
AR always uses mask 12 (51-100): exact suffix probability conditioned on 1-50.
p_z is copied verbatim from verified source CSVs; no rounding or averaging.
Zero p_z may reflect floating-point underflow; original Drive results retain log_p_z.
All source files passed completion-manifest SHA256 checks. Unique tasks and
identical window identities across all five models were verified.
