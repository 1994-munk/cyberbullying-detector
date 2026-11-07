import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCheck,
  ChevronRight,
  CircleDot,
  Clipboard,
  Code2,
  Database,
  FlaskConical,
  Gauge,
  GitBranch,
  Layers3,
  Terminal,
  TriangleAlert,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const stages = [
  { id: "inspect", label: "Inspect data", short: "01", Icon: Database },
  { id: "clean", label: "Clean & audit", short: "02", Icon: FlaskConical },
  { id: "split", label: "Split fairly", short: "03", Icon: GitBranch },
  { id: "train", label: "Train a model", short: "04", Icon: Layers3 },
  { id: "evaluate", label: "Evaluate & save", short: "05", Icon: Gauge },
] as const;

const stageDetails = [
  {
    id: "inspect",
    eyebrow: "START WITH THE DATA",
    title: "Know what you’re learning from.",
    description:
      "Read one JSON record per line. Find the message in `content` and its label inside `annotation.label` before building anything.",
    code: `import pandas as pd

# Each line is one record in this dataset.
raw = pd.read_json(
    "Dataset_for_Detection_of_Cyber-Trolls_1.json",
    lines=True
)

print(f"Rows: {len(raw):,}")
print(raw[["content", "annotation"]].head())

# The source has 20,001 labeled tweets.
# Label 0 = non cyber-aggressive
# Label 1 = cyber-aggressive`,
    note: "These source labels describe aggressive language, not proof of repeated bullying, intent, or harm.",
  },
  {
    id: "clean",
    eyebrow: "MAKE THE LABELS TRUSTWORTHY",
    title: "Clean text. Keep the meaning.",
    description:
      "Normalize spacing and HTML, remove blank or unknown-label records, and audit repeated text. Conflicting examples should not quietly teach the model two answers.",
    code: `import html

# Unpack the nested labels before cleaning.
data = pd.DataFrame({
    "text": raw["content"],
    "label": raw["annotation"].map(
        lambda item: item["label"][0]
        if isinstance(item, dict)
        and item.get("label") else None
    )
})

data["text"] = data["text"].fillna("").map(
    lambda text: " ".join(
        html.unescape(str(text)).split()
    )
)
data["label"] = pd.to_numeric(
    data["label"], errors="coerce"
)
data = data.dropna(subset=["label"])
data = data[data["text"].str.strip().ne("")]

# Check repeats; the complete starter audits and removes
# repeated messages whose labels disagree.`,
    note: "The supplied file includes thousands of repeated messages, including 15 normalized texts with conflicting labels.",
  },
  {
    id: "split",
    eyebrow: "PREVENT DATA LEAKAGE",
    title: "Keep identical tweets together.",
    description:
      "Remove duplicates and ambiguous repeated labels before splitting. A stratified holdout preserves class proportions and gives a fairer test of unseen text.",
    code: `from sklearn.model_selection import train_test_split

# Split unique, non-conflicting messages only.
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"].astype(int),
    test_size=0.20,
    random_state=42,
    stratify=data["label"],
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows:     {len(X_test):,}")`,
    note: "The original file has 5,373 exact repeats. After whitespace and case normalization, 15 repeated messages have conflicting labels.",
  },
  {
    id: "train",
    eyebrow: "BUILD A BASELINE",
    title: "Turn words into useful signals.",
    description:
      "TF–IDF emphasizes words and two-word phrases that help distinguish messages. Logistic regression makes a fast first model with a useful class-level evaluation.",
    code: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1_200,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X_train, y_train)`,
    note: "Balanced class weights help the model learn despite the uneven class counts; they do not remove dataset bias.",
  },
  {
    id: "evaluate",
    eyebrow: "CHECK BEFORE YOU TRUST",
    title: "Look beyond a single score.",
    description:
      "Compare a majority-only baseline, then inspect precision, recall, F1, and the confusion matrix. Save the entire pipeline so future text uses the learned vocabulary.",
    code: `from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import joblib

predictions = model.predict(X_test)
print(classification_report(
    y_test,
    predictions,
    target_names=["non-aggressive", "aggressive"],
    zero_division=0,
))
print(confusion_matrix(y_test, predictions))

# Save the vectorizer and classifier together.
joblib.dump(model, "cyberbully_baseline.joblib")

# A prediction is a cue for human review—not a verdict.
print(model.predict(["Example message to review."]))`,
    note: "Message-level text cannot prove bullying. Review human, social, and cultural context before taking action.",
  },
] as const;

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Cyberbully Detection — Python Text Classification Lab" },
      {
        name: "description",
        content:
          "A five-step, beginner-friendly Python project for inspecting tweet labels, cleaning text, training a baseline, and evaluating cyber-aggressive language classification.",
      },
      { property: "og:title", content: "Cyberbully Detection — Python Lab" },
      {
        property: "og:description",
        content:
          "Explore the uploaded tweet dataset and follow a clear, five-step Python machine-learning walkthrough.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

function Index() {
  const [activeStage, setActiveStage] = useState(0);
  const [copied, setCopied] = useState(false);
  const stage = stageDetails[activeStage];

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(stage.code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }

  return (
    <main className="lab-shell min-h-screen bg-background text-foreground">
      <header className="lab-topbar">
        <a className="lab-brand" href="#top" aria-label="Cyberbully Detection home">
          <span className="brand-mark"><Activity size={17} strokeWidth={2.2} /></span>
          <span className="brand-name">FIELDNOTES <span>/ ML LAB</span></span>
        </a>
        <div className="topbar-right">
          <span className="dataset-status"><span className="status-dot" /> DATASET LOADED</span>
          <a className="topbar-download" href="/project/README.md" download>
            <ArrowDownToLine size={15} /> <span>PROJECT FILES</span>
          </a>
        </div>
      </header>

      <div id="top" className="lab-layout">
        <aside className="lab-sidebar" aria-label="Project overview">
          <div className="sidebar-label">PROJECT / 001</div>
          <h1 className="sidebar-title">Cyberbully<br />Detection<span className="sidebar-period">.</span></h1>
          <p className="sidebar-description">A practical Python study of language, labels, and what a model can—and can’t—tell us.</p>

          <div className="sidebar-rule" />
          <p className="sidebar-label">THE STUDY</p>
          <div className="study-stat"><Database size={15} /><span>20,001</span><small>labeled messages</small></div>
          <div className="study-stat"><CircleDot size={15} /><span>2</span><small>source labels</small></div>
          <div className="label-key">
            <p><i className="key-dot key-aggressive" /> <span>1</span><small>cyber-aggressive</small></p>
            <p><i className="key-dot key-neutral" /> <span>0</span><small>non-aggressive</small></p>
          </div>

          <div className="sidebar-rule" />
          <p className="sidebar-label">THE METHOD</p>
          <div className="method-list">
            <span><i /> JSON Lines</span>
            <span><i /> TF–IDF features</span>
            <span><i /> Logistic regression</span>
            <span><i /> Stratified holdout</span>
          </div>

          <div className="sidebar-caution">
            <TriangleAlert size={15} />
            <p><strong>Not a verdict.</strong> Aggressive language is not the same as proven bullying. Human context matters.</p>
          </div>
          <a className="dataset-source" href="https://www.kaggle.com/datasets/dataturks/dataset-for-detection-of-cybertrolls" target="_blank" rel="noreferrer">SOURCE DATASET <ArrowRight size={13} /></a>
        </aside>

        <section className="lab-main" aria-label="Python project walkthrough">
          <div className="lab-eyebrow"><span>PYTHON / NATURAL LANGUAGE PROCESSING</span><span>GUIDED PROJECT <b>↗</b></span></div>
          <div className="main-heading-row">
            <div>
              <h2>From raw tweets<br /><span>to a tested model.</span></h2>
              <p className="intro-copy">A clear, step-by-step machine-learning project using your uploaded dataset. Start with the data; finish with a measured, human-reviewed baseline.</p>
            </div>
            <div className="heading-index" aria-hidden="true"><span>FIELD</span><b>01</b><span>OF 05</span></div>
          </div>

          <div className="dataset-strip">
            <div className="strip-item"><span className="strip-icon"><Database size={17} /></span><div><b>20,001</b><small>labeled tweet records</small></div></div>
            <div className="strip-divider" />
            <div className="strip-item"><span className="strip-icon strip-icon-coral"><Activity size={17} /></span><div><b>7,822</b><small>aggressive · label 1</small></div></div>
            <div className="strip-divider" />
            <div className="strip-item"><span className="strip-icon strip-icon-green"><CheckCheck size={17} /></span><div><b>12,179</b><small>non-aggressive · label 0</small></div></div>
            <div className="strip-note">Counts from the uploaded file<br />before cleaning or deduplication.</div>
          </div>

          <div className="lesson-label"><span><span className="lesson-pulse" /> THE WORKFLOW</span><span>FIVE PRACTICAL STAGES</span></div>
          <nav className="stage-nav" aria-label="Project stages">
            {stages.map(({ id, label, short, Icon }, index) => (
              <button
                className={`stage-tab ${index === activeStage ? "stage-tab-active" : ""}`}
                key={id}
                onClick={() => { setActiveStage(index); setCopied(false); }}
                aria-current={index === activeStage ? "step" : undefined}
              >
                <span className="stage-icon"><Icon size={17} /></span>
                <span className="stage-label"><small>STEP {short}</small><b>{label}</b></span>
                {index < activeStage && <Check size={14} className="stage-complete" />}
              </button>
            ))}
          </nav>

          <article className="lesson-panel" key={stage.id}>
            <div className="lesson-copy">
              <div className="step-kicker"><span>{stage.eyebrow}</span><span>0{activeStage + 1} / 05</span></div>
              <h3>{stage.title}</h3>
              <p className="lesson-description">{stage.description}</p>
              <div className="insight-note"><span className="insight-symbol">i</span><p>{stage.note}</p></div>
              <div className="lesson-controls">
                <Button variant="outline" size="sm" onClick={() => { setActiveStage(Math.max(0, activeStage - 1)); setCopied(false); }} disabled={activeStage === 0} aria-label="Previous stage">
                  <ArrowLeft size={15} /> <span>PREVIOUS</span>
                </Button>
                <span className="step-count">{String(activeStage + 1).padStart(2, "0")} <i>/</i> 05</span>
                <Button size="sm" onClick={() => { setActiveStage(Math.min(4, activeStage + 1)); setCopied(false); }} disabled={activeStage === 4}>
                  <span>NEXT STEP</span> <ArrowRight size={15} />
                </Button>
              </div>
            </div>

            <div className="code-window">
              <div className="code-header"><span><Terminal size={14} /> <b>PYTHON</b> <i>·</i> {activeStage === 0 ? "inspect_data.py" : `${stage.id}.py`}</span>
                <Button variant="ghost" size="sm" onClick={copyCode} aria-label={copied ? "Copied code" : "Copy code"} title="Copy this step’s code">
                  {copied ? <Check size={14} /> : <Clipboard size={14} />}<span>{copied ? "COPIED" : "COPY"}</span>
                </Button>
              </div>
              <pre className="code-body"><code>{stage.code}</code></pre>
              <div className="code-footer"><span><span className="code-live-dot" /> READY TO RUN</span><span><Code2 size={13} /> PYTHON 3.10+</span></div>
            </div>
          </article>

          <div className="finish-band">
            <div className="finish-symbol"><CheckCheck size={19} /></div>
            <div className="finish-copy"><strong>Run the complete project on your machine.</strong><span>Get the commented Python script and setup guide. Add your dataset locally to run the full analysis.</span></div>
            <a className="download-button" href="/project/cyberbully_detection.py" download><ArrowDownToLine size={16} /> DOWNLOAD PYTHON</a>
            <a className="readme-link" href="/project/README.md" download>SETUP GUIDE <ChevronRight size={14} /></a>
          </div>

          <footer className="lab-footer"><span>FIELDNOTES <b>·</b> A SMALL STUDY IN TEXT CLASSIFICATION</span><span>MODEL OUTPUT IS A SIGNAL, NOT A JUDGMENT.</span></footer>
        </section>
      </div>
    </main>
  );
}