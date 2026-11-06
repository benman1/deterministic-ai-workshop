# Colab notebooks (GPU)

Two experiments that are too slow for a laptop CPU. Open each in Google Colab (File, Upload notebook), choose Runtime, Change runtime type, T4 GPU, and run the cells top to bottom.

| Notebook | Question | Time on a T4 (estimate) |
|---|---|---|
| `18b_finetune_student_colab.ipynb` | Does fine-tuning all the weights of the student, instead of freezing them as notebook 18 does, change the gap between human labels and language-model labels on Banking77? 18 runs: 2 sizes, 3 label sources, 3 seeds. | about 10 minutes |
| `19_loop_repair_colab.ipynb` | A small model that thinks before answering can loop and never finish. How much does a decoding setting remove, and how much does a LoRA trained on the model's own finished traces remove? Qwen3-0.6B on MATH-500. | about 1 hour with the defaults |

**Status.** Both notebooks passed a CPU smoke test (`SMOKE = True`, tiny sizes) that runs every cell; neither has been run on a Colab GPU, and the times above are estimates from model size. The first printed lines of each run show the real speed. Nothing here is a result until you run it and download the results file.

**Files for 18b.** Upload `sample_idx.json`, `teacher_labels.jsonl` and `teacher_labels_opus.jsonl` from `data/results/18_distillation/` when the notebook asks (it tries GitHub first, so once those files are pushed the upload step disappears).

**Bring the numbers back.** Each notebook writes a small JSON file next to itself. Download it from the Colab file browser into `data/results/` here, then the book's documents and slides can quote it.
