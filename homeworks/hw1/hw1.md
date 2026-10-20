Report Builder

Build your final report block by block — screenshots, descriptions, guided answers — then export one file for Blackboard. Everything stays in this browser until you export.
⬇ Export report (HTML → print to PDF)
Start over
Back to the brief

How to submit: export the HTML file, open it in a browser, press Ctrl/Cmd+P, choose Save as PDF, name it Codename_Report.pdf, upload to Blackboard. Your code is embedded in the export as an appendix.
1 · Report header
Codename (leaderboard)
Your name + email
One-paragraph overview

In 3–5 sentences: what did you build, what representation did you choose for the images, and what is the single most important thing you learned going from the linear baseline to the MLP?
2 · Dataset & collection

Upload a screenshot of your Copybook coverage panel (trace.html) and describe your collection.
Coverage screenshot
Click (or drop) the coverage-panel screenshot here

How many raw captures per character did you collect, and how did you keep the classes balanced? Any capture you redid and why?
3 · Assignment — linear softmax baseline
＋ Add a linear-baseline evidence block (screenshot + description)

Suggested evidence: the training-loss/accuracy printout, and the Evaluation Site result page for linear_model.onnx (with your submission ID visible).
4 · Project — the MLP
＋ Add an experiment / training block (screenshot + description)

Rubric asks for at least 3 controlled experiments (pooling factor, width, depth, initialization, …). One block each: change one thing, show the numbers, say what you concluded. Include your final training curve/printout.
5 · Evaluation Site results
Submission ID screenshot
Screenshot showing your submission ID(s) on the Evaluation Site

List both submission IDs (linear and MLP) and their scores.
Hidden-dataset result screenshot
Screenshot of the full report page for your best MLP (slices, confusion, latency)

Analysis (graded): which font slice or confusion pair is weakest, and why do you think that happens? What would you try next?
6 · Comparison & conclusion

Compare linear vs MLP on the hidden set (accuracy, parameters, latency). Was the MLP worth its extra parameters everywhere, or only on some slices?

What is the single change you made that improved hidden accuracy the most — and what did you try that did NOT help?

Optional (worth up to 5 bonus points) — Is your dataset linearly separable?

With 11 classes in your chosen feature space (16×16 = 256 dims, or 8×8 = 64 dims, depending on your pooling factor), can a set of 11 linear decision boundaries (one per class, as softmax computes) perfectly partition all classes? What does the hidden-set result of your linear model tell you? A useful lens: Linear Discriminant Analysis (LDA) projects data onto directions that maximize between-class separation — if classes overlap in that projection, no linear boundary will separate them perfectly.
＋ Add a separability evidence block (screenshot + description)
7 · Your code

Upload the completed sample_mlp.py you actually ran. It is embedded in the exported report as Appendix A
sample_mlp.py
Click (or drop) your final .py file here
Export checklist

Green = auto-detected as complete. Check the boxes yourself as you review — unchecked items are exported with a warning.
Codename + your name are filled in ✗ not yet
Overview paragraph (≥ 80 characters) ✗ not yet
Coverage screenshot attached + collection answers written ✗ not yet
≥ 1 linear evidence block with screenshot + answers ✗ not yet
≥ 3 MLP experiment blocks (rubric requirement) ✗ not yet
Every experiment block has a screenshot + both answers ✗ not yet
Submission-ID screenshot attached + both IDs listed ✗ not yet
Hidden-result screenshot + analysis written ✗ not yet
Comparison + conclusion (≥ 60 characters) ✗ not yet
Most-helpful-change + what-did-not-help ✗ not yet
sample_mlp.py uploaded ✗ not yet