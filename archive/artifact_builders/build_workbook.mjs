import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";
import { deepReview } from "./deep_review_data.mjs";

const outputDir = new URL("../outputs/", import.meta.url).pathname;
const previewDir = new URL("./workbook-previews/", import.meta.url).pathname;
const corpusPath = new URL("./screened_corpus.csv", import.meta.url).pathname;
const font = "Arial";
const navy = "#17324D";
const blue = "#DCEAF7";
const pale = "#F4F7FA";
const amber = "#FFF1CC";
const green = "#DDEFE2";
const red = "#F8DFDF";
const border = "#C9D3DD";

function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ""; }
    else if (c === '\n') { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  const header = rows.shift();
  return rows.filter(r => r.some(v => v !== "")).map(r => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function styleTitle(sheet, title, subtitle, width) {
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: font, size: 16, bold: true, color: navy };
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${width}2`).format.font = { name: font, size: 10, italic: true, color: "#506274" };
  sheet.getRange(`A3:${width}3`).format.borders = { bottom: { style: "thin", color: navy } };
}

function styleHeader(range) {
  range.format = {
    fill: navy,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
  range.format.rowHeight = 30;
}

function styleBody(range) {
  range.format.font = { name: font, size: 9, color: "#1F2933" };
  range.format.verticalAlignment = "top";
  range.format.wrapText = true;
  range.format.borders = { insideHorizontal: { style: "thin", color: border } };
}

const workbook = Workbook.create();

// Overview
const overview = workbook.worksheets.add("Overview");
styleTitle(overview, "LLM fine-tuning and catastrophic forgetting", "State of the art through 4 September 2026; designed for an eight-week, 3-4 person course project", "H");
overview.getRange("A5:B9").values = [
  ["Review measure", "Count"],
  ["Index records returned", 1000],
  ["Unique records after cross-query deduplication", 746],
  ["Records retained for title/abstract screening", 180],
  ["Papers closely reviewed", 60],
];
styleHeader(overview.getRange("A5:B5"));
styleBody(overview.getRange("A6:B9"));
overview.getRange("B6:B9").format.numberFormat = "#,##0";

overview.getRange("D5:H5").values = [["Recommended project", "Core question", "Novelty confidence", "Essential runs", "Best hardware fit"]];
styleHeader(overview.getRange("D5:H5"));
overview.getRange("D6:H6").values = [[
  "Replay or Re-alignment? Damage-Aware KL Anchoring",
  "Which small unlabeled prompt set best protects capabilities under a fixed KL-anchor budget, and does it reduce residual rather than only prompt-recoverable loss?",
  "Medium",
  "12-18",
  "24GB; 16GB MVP",
]];
styleBody(overview.getRange("D6:H6"));
overview.getRange("D6:H6").format.fill = green;
overview.getRange("D6:H6").format.rowHeight = 74;

overview.getRange("A12:H12").values = [["Rank", "Direction", "Why it is promising", "Main risk", "16GB", "24GB", "48GB", "Recommendation"]];
styleHeader(overview.getRange("A12:H12"));
overview.getRange("A13:H15").values = [
  [1, "Damage-aware KL anchors plus recovery evaluation", "Algorithmic and measurement contribution; fixed-budget question", "Adjacent work means novelty is medium, not certain", "MVP", "Recommended", "Full version", "Best overall"],
  [2, "Recoverable versus durable LoRA/full-FT forgetting", "Strong controls and useful null result", "Mostly an evaluation contribution", "Recommended", "Full version", "Cross-scale", "Safest"],
  [3, "Unified geometric predictors of LoRA forgetting", "Mechanistic and PhD-relevant", "SVD/activation analysis and statistical leakage risk", "Reduced", "Recommended", "Full version", "Most ambitious"],
];
styleBody(overview.getRange("A13:H15"));
overview.getRange("A13:A15").format.horizontalAlignment = "center";

overview.getRange("A18:H18").values = [["Key conclusions", "Evidence status", "What it means for the project", "", "", "", "", ""]];
styleHeader(overview.getRange("A18:H18"));
overview.getRange("A19:C24").values = [
  ["Plain LoRA versus full FT is no longer novel", "Established", "Use both as baselines and match target learning"],
  ["Plain KD/KL anchoring is no longer novel", "Established", "Make anchor selection and recovery-aware measurement the contribution"],
  ["Raw benchmark drop mixes access, format and durable loss", "Strong evidence", "Use multiple prompts, forced-choice scoring and item transitions"],
  ["No universal SOTA exists across incompatible regimes", "Established", "Compare only within the same model, data, budget and metric protocol"],
  ["Magnitude, spectrum and layer-location explanations conflict", "Open", "A unified predictor study is defensible"],
  ["Random replay remains a strong baseline", "Established", "Include it before implementing complex continual-learning methods"],
];
styleBody(overview.getRange("A19:C24"));
overview.getRange("A26:H27").values = [["Caution", "Novelty is a literature-backed risk judgment, not proof. Run one final focused search immediately before proposal submission and do not call non-recovery under a finite probe set 'knowledge erasure'.", "", "", "", "", "", ""], ["Companion report", "The PDF explains definitions, trends, SOTA by comparable regime, gaps, evaluation design, 11 ideas, and three complete proposals.", "", "", "", "", "", ""]];
overview.getRange("A26:H27").format.fill = amber;
styleBody(overview.getRange("A26:H27"));
overview.getRange("A:A").format.columnWidth = 24;
overview.getRange("B:B").format.columnWidth = 18;
overview.getRange("C:C").format.columnWidth = 32;
overview.getRange("D:D").format.columnWidth = 28;
overview.getRange("E:H").format.columnWidth = 18;

// Deep review
const evidence = workbook.worksheets.add("Deep review");
styleTitle(evidence, "Close review of 60 papers", "Author claims are recorded separately from limitations; URLs point to primary or official sources when available", "M");
const evidenceHeaders = ["Citation", "Venue and status", "Research question", "Models and sizes", "Datasets and task sequences", "Fine-tuning method", "Baselines", "Metrics", "Main result", "Important limitations", "Code", "Approximate compute", "Paper URL"];
evidence.getRange("A5:M5").values = [evidenceHeaders];
styleHeader(evidence.getRange("A5:M5"));
const evidenceRows = deepReview.map(x => [x.citation, x.venueStatus, x.question, x.models, x.datasets, x.method, x.baselines, x.metrics, x.result, x.limitations, x.code, x.compute, x.url]);
evidence.getRange(`A6:M${5 + evidenceRows.length}`).values = evidenceRows;
styleBody(evidence.getRange(`A6:M${5 + evidenceRows.length}`));
evidence.getRange(`A6:M${5 + evidenceRows.length}`).format.rowHeight = 76;
const evWidths = [36, 22, 34, 29, 34, 28, 26, 28, 42, 42, 28, 26, 42];
evWidths.forEach((w, i) => evidence.getRangeByIndexes(0, i, 65, 1).format.columnWidth = w);
evidence.freezePanes.freezeRows(5);
evidence.freezePanes.freezeColumns(1);
const evTable = evidence.tables.add(`A5:M${5 + evidenceRows.length}`, true, "DeepReviewTable");
evTable.style = "TableStyleMedium2";

// Screened corpus
const corpusText = await fs.readFile(corpusPath, "utf8");
const corpusRows = parseCsv(corpusText);
const screened = workbook.worksheets.add("Screened corpus");
styleTitle(screened, "Bounded title and abstract screening log", "180 high-relevance records retained from 746 deduplicated index records; numeric score bounded discovery but did not determine evidence quality", "I");
const screenedHeaders = ["Title", "Year", "Venue", "Type", "First-pass decision", "Relevance score", "Citation count", "Screen note", "Source URL"];
screened.getRange("A5:I5").values = [screenedHeaders];
styleHeader(screened.getRange("A5:I5"));
const screenedRows = corpusRows.map(x => [x.title, Number(x.year) || null, x.venue, x.type, x.screen_decision === "Include" ? "Candidate" : "Not promoted", Number(x.relevance_score) || 0, Number(x.citations) || 0, x.screen_note, x.url]);
screened.getRange(`A6:I${5 + screenedRows.length}`).values = screenedRows;
styleBody(screened.getRange(`A6:I${5 + screenedRows.length}`));
screened.getRange(`A6:I${5 + screenedRows.length}`).format.rowHeight = 50;
screened.getRange("A:A").format.columnWidth = 48;
screened.getRange("B:B").format.columnWidth = 9;
screened.getRange("C:C").format.columnWidth = 28;
screened.getRange("D:D").format.columnWidth = 14;
screened.getRange("E:E").format.columnWidth = 16;
screened.getRange("F:G").format.columnWidth = 12;
screened.getRange("H:H").format.columnWidth = 46;
screened.getRange("I:I").format.columnWidth = 44;
screened.freezePanes.freezeRows(5);
screened.freezePanes.freezeColumns(1);
const corpusTable = screened.tables.add(`A5:I${5 + screenedRows.length}`, true, "ScreenedCorpusTable");
corpusTable.style = "TableStyleMedium2";

// Project ideas
const ideas = workbook.worksheets.add("Project ideas");
styleTitle(ideas, "Project idea scoring", "Scores are 1-5. For complexity, compute and null-result risk, lower is better. Priority is formula-driven from the visible rubric.", "K");
ideas.getRange("A5:K5").values = [["Direction", "Novelty", "Scientific value", "Feasibility", "Complexity", "Compute", "Data and code", "Null-result risk", "PhD value", "Priority score", "Recommendation"]];
styleHeader(ideas.getRange("A5:K5"));
const ideaRows = [
  ["Damage-aware KL anchors plus recovery evaluation",4,5,4,3,3,5,3,5,null,"Best overall"],
  ["Recoverable versus durable target-matched LoRA/full FT",4,5,5,2,3,5,2,5,null,"Safest"],
  ["Unified geometric predictor of LoRA forgetting",4,5,3,4,3,4,4,5,null,"Most ambitious"],
  ["Prompt-robust forgetting benchmark for small LLMs",3,4,5,2,1,5,2,4,null,"Good measurement project"],
  ["Active, disabled, scaled and merged adapter interference",4,4,5,2,2,5,2,4,null,"Strong compact study"],
  ["Damage-aware labeled replay for decoder instruction tuning",3,4,5,3,2,5,2,4,null,"Good extension"],
  ["Optimizer reality check: SFT, SAM, FINCH, replay and KL",3,4,4,3,3,4,3,4,null,"Replication/extension"],
  ["Task-distance map for orthogonal LoRA",4,5,3,4,3,4,4,5,null,"High-risk mechanism study"],
  ["Adapter composition order and routing interference",3,4,4,3,2,4,3,4,null,"Systems option"],
  ["Real versus self-generated replay at equal total budget",2,4,3,3,3,4,3,4,null,"Crowded but useful"],
  ["Source-importance freezing versus simple masks",2,3,4,3,3,4,3,3,null,"Lower novelty"],
];
ideas.getRange("A6:K16").values = ideaRows;
ideas.getRange("J6").formulas = [["=0.2*B6+0.2*C6+0.2*D6+0.1*G6+0.1*I6+0.1*(6-E6)+0.05*(6-F6)+0.05*(6-H6)"]];
ideas.getRange("J6:J16").fillDown();
ideas.getRange("J6:J16").format.numberFormat = "0.00";
styleBody(ideas.getRange("A6:K16"));
ideas.getRange("A6:K16").format.rowHeight = 42;
ideas.getRange("A:A").format.columnWidth = 46;
ideas.getRange("B:J").format.columnWidth = 12;
ideas.getRange("K:K").format.columnWidth = 24;
ideas.getRange("B6:J16").format.horizontalAlignment = "center";
ideas.freezePanes.freezeRows(5);
ideas.getRange("A19:B27").values = [
  ["Priority formula", "20% novelty + 20% scientific value + 20% feasibility + 10% data/code + 10% PhD value + 10% ease + 5% low compute + 5% low null risk"],
  ["Novelty", "Literature-backed likelihood that the exact controlled question is not already settled"],
  ["Scientific value", "Ability to clarify a mechanism, measurement problem or generalizable method"],
  ["Feasibility", "Probability of a complete, interpretable project in eight weeks"],
  ["Complexity", "Engineering and analysis difficulty; 1 is easiest"],
  ["Compute", "Relative GPU demand; 1 is lowest"],
  ["Data and code", "Availability of public datasets, baselines and reproducible tooling"],
  ["Null-result risk", "Risk that the study is too noisy or underpowered; 1 is lowest"],
  ["PhD value", "Potential to develop into a deeper research program"],
];
styleBody(ideas.getRange("A19:B27"));
ideas.getRange("A19:A27").format.font = { name: font, size: 9, bold: true, color: navy };
ideas.getRange("B:B").format.columnWidth = 26;

// Starting idea verdicts
const verdicts = workbook.worksheets.add("Starting ideas");
styleTitle(verdicts, "Verdict on the team's starting ideas", "The recommendation challenges the original assumptions and preserves useful ideas as baselines or narrower extensions", "D");
verdicts.getRange("A5:D5").values = [["Starting idea", "Verdict", "Reason", "Best use"]];
styleHeader(verdicts.getRange("A5:D5"));
verdicts.getRange("A6:D12").values = [
  ["Plain LoRA versus full FT", "Abandon as novelty", "The basic learning-forgetting trade-off is already directly studied", "Keep as target-matched baselines"],
  ["LoRA/full FT plus KD anchoring", "Plain version occupied", "KL, self-distillation and small-prompt anchoring already exist", "Make anchor selection and recovery evaluation the contribution"],
  ["Rank, frozen backbone and update magnitude", "Crowded as stated", "Recent work separately tests all three and adds spectral/activation explanations", "Compare multiple predictors on identical checkpoints"],
  ["Recoverable versus genuine forgetting", "Pursue", "No accepted recovery taxonomy is routinely used in LoRA/full-FT comparisons", "Use operational categories and avoid claims of proven erasure"],
  ["Layer-wise or adaptive rank", "Crowded", "Many layer- and rank-aware LoRA variants now exist", "Allocate rank only from a tested predictive signal and fixed budget"],
  ["Gradient or subspace prediction", "Pursue as diagnosis", "Signals conflict and simple similarities can be weak", "Benchmark against magnitude, target loss and activation drift"],
  ["Adapter composition and interference", "Viable systems study", "Preservation can be routing/isolation rather than internal retention", "Report storage, latency, task IDs and routing errors"],
];
styleBody(verdicts.getRange("A6:D12"));
verdicts.getRange("A6:D12").format.rowHeight = 58;
verdicts.getRange("A:A").format.columnWidth = 36;
verdicts.getRange("B:B").format.columnWidth = 22;
verdicts.getRange("C:D").format.columnWidth = 52;

// Methodology and evaluation checklist
const method = workbook.worksheets.add("Methodology");
styleTitle(method, "Search and evaluation methodology", "Counts describe a bounded course-project review, not a registered meta-analysis", "F");
method.getRange("A5:B5").values = [["Search measure", "Value"]];
styleHeader(method.getRange("A5:B5"));
method.getRange("A6:B11").values = [
  ["Coverage", "January 2023 to 4 September 2026; foundational earlier work retained selectively"],
  ["Sources", "OpenAlex, arXiv, OpenReview, ICLR, ICML/PMLR, NeurIPS, ACL Anthology, AAAI, TMLR, ACM surveys, official code pages"],
  ["Records returned", 1000], ["Unique after deduplication", 746], ["Title/abstract screened", 180], ["Close reviewed", 60],
];
styleBody(method.getRange("A6:B11"));
method.getRange("B8:B11").format.numberFormat = "#,##0";
method.getRange("D5:F5").values = [["Required evaluation control", "Minimum implementation", "Why"]];
styleHeader(method.getRange("D5:F5"));
method.getRange("D6:F15").values = [
  ["Target learning", "Target accuracy/EM plus loss", "Retention without learning is not useful"],
  ["Target matching", "Compare checkpoints within a preset score tolerance", "Equal steps do not imply equal adaptation"],
  ["Item transitions", "Correct->incorrect and incorrect->correct", "Aggregate scores hide movement"],
  ["Prompt battery", "Five equivalent templates; mean/median/worst", "One prompt can create a false conclusion"],
  ["Likelihood scoring", "Forced-choice log-likelihood for MCQ", "Separates knowledge from output formatting"],
  ["Format audit", "Parser/refusal/schema-failure rate", "Exact match can mislabel access as loss"],
  ["Recovery", "Fixed few-shot prompts; optional tiny repair", "Separates prompt access from residual loss"],
  ["Seeds", "Two central seeds; three for primary claim if possible", "Training variance can exceed small retention gains"],
  ["Statistics", "Paired bootstrap over items; show seed points", "Uses paired before/after structure"],
  ["Resources", "Tokens, forwards/backwards, wall time, memory, storage", "Methods make different cost assumptions"],
];
styleBody(method.getRange("D6:F15"));
method.getRange("A:A").format.columnWidth = 28;
method.getRange("B:B").format.columnWidth = 78;
method.getRange("D:D").format.columnWidth = 28;
method.getRange("E:E").format.columnWidth = 48;
method.getRange("F:F").format.columnWidth = 48;
method.getRange("A18:F18").values = [["Limitations", "Index coverage and 2026 publication status can change. Some papers omit seeds, compute, or compatible baselines. Missing a paper is not proof of novelty. The screening score bounded discovery and is not an evidence-quality rating.", "", "", "", ""]];
method.getRange("A18:F18").format.fill = amber;
styleBody(method.getRange("A18:F18"));

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

// Verification inspections before export.
const overviewCheck = await workbook.inspect({ kind: "table", range: "Overview!A1:H27", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 10 });
console.log(overviewCheck.ndjson);
const ideaCheck = await workbook.inspect({ kind: "table", range: "Project ideas!A5:K16", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 12 });
console.log(ideaCheck.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errors.ndjson);

const renderSpecs = [
  ["Overview", "A1:H27", "overview"],
  ["Deep review", "A1:M14", "deep-review-top"],
  ["Deep review", "A30:M38", "deep-review-middle"],
  ["Deep review", "A56:M65", "deep-review-bottom"],
  ["Screened corpus", "A1:I15", "screened-top"],
  ["Screened corpus", "A176:I185", "screened-bottom"],
  ["Project ideas", "A1:K27", "project-ideas"],
  ["Starting ideas", "A1:D12", "starting-ideas"],
  ["Methodology", "A1:F18", "methodology"],
];
for (const [sheetName, range, name] of renderSpecs) {
  const img = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await img.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/llm_forgetting_sota_evidence.xlsx`);
console.log(`Saved ${outputDir}/llm_forgetting_sota_evidence.xlsx`);
