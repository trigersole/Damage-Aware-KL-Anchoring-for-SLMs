from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

import build_report_pdf as core


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "damage-aware-kl-study-guide.md"
OUTPUT = ROOT / "output" / "pdf" / "damage-aware-kl-anchoring-study-guide.pdf"


def step_style(name: str, color) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        parent=core.BODY,
        fontName=core.BOLD,
        fontSize=10.2,
        leading=13.2,
        alignment=TA_CENTER,
        textColor=color,
        spaceAfter=0,
    )


def flow_diagram() -> Table:
    stages = [
        ("1. Original base model", "Broad abilities before specialization", core.PALE_BLUE),
        ("2. Short pilot LoRA run", "Reveal which general prompts change most", core.PALE_TEAL),
        ("3. Score candidate prompts", "Base-versus-pilot KL becomes the damage signal", core.PALE_BLUE),
        ("4. Select a fixed anchor set", "Compare random, diverse, damaged, and diverse-damaged prompts", core.PALE_TEAL),
        ("5. Reset to a fresh base copy", "Every main condition begins from the same starting model", core.PALE_BLUE),
        ("6. Main LoRA training", "Target-task loss plus teacher-student KL on the selected anchors", core.PALE_TEAL),
        ("7. Evaluate", "Target learning, general retention, item transitions, and recovery", core.PALE_BLUE),
    ]
    data = []
    backgrounds = []
    for index, (title, subtitle, background) in enumerate(stages):
        text = f"<b>{title}</b><br/><font size='8' color='#5C6B78'>{subtitle}</font>"
        data.append([Paragraph(text, step_style(f"Step{index}", core.NAVY))])
        backgrounds.append(background)
        if index < len(stages) - 1:
            data.append([Paragraph("↓", ParagraphStyle(
                f"Arrow{index}", parent=core.BODY, fontName=core.BOLD,
                fontSize=15, leading=17, alignment=TA_CENTER,
                textColor=core.TEAL, spaceAfter=0,
            ))])
    table = Table(data, colWidths=[core.CONTENT_W * 0.82], hAlign="CENTER")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]
    stage_row = 0
    for background in backgrounds:
        commands.extend([
            ("BACKGROUND", (0, stage_row), (0, stage_row), background),
            ("BOX", (0, stage_row), (0, stage_row), 0.65, core.GRID),
            ("TOPPADDING", (0, stage_row), (0, stage_row), 7),
            ("BOTTOMPADDING", (0, stage_row), (0, stage_row), 7),
        ])
        stage_row += 2
    table.setStyle(TableStyle(commands))
    return table


def study_cover_story() -> list:
    big = ParagraphStyle(
        "StudyBig",
        parent=core.TITLE,
        fontSize=22,
        leading=25,
        alignment=TA_CENTER,
    )
    small = ParagraphStyle(
        "StudySmall",
        parent=core.CAPTION,
        alignment=TA_CENTER,
    )
    return [
        Spacer(1, 22 * mm),
        Paragraph("BEGINNER-FIRST PROJECT STUDY GUIDE", core.KICKER),
        Paragraph("Damage-Aware KL Anchoring for Small Language Models", core.TITLE),
        Paragraph(
            "From AI and LLM fundamentals to the complete experiment, evaluation, team plan, and novelty claim",
            core.SUBTITLE,
        ),
        Spacer(1, 11 * mm),
        Table(
            [[
                Paragraph("0.5B-1.5B", big),
                Paragraph("10", big),
                Paragraph("8 weeks", big),
            ], [
                Paragraph("recommended model scale", small),
                Paragraph("central training runs", small),
                Paragraph("course-project plan", small),
            ]],
            colWidths=[core.CONTENT_W / 3] * 3,
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), core.PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, core.GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, core.GRID),
                ("TOPPADDING", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
            ]),
        ),
        Spacer(1, 13 * mm),
        Paragraph(
            "Core idea: use a short pilot fine-tuning run to find the general prompts that are most vulnerable, "
            "then spend a small fixed KL-protection budget on those prompts during the real LoRA training run.",
            core.CALLOUT,
        ),
        Spacer(1, 27 * mm),
        Paragraph("Prepared as a self-contained learning resource", core.CAPTION),
        Paragraph("Project framing: research-oriented LLM course project for a 3-4 person team", core.CAPTION),
        PageBreak(),
        Paragraph("The complete project in one picture", core.H1),
        Paragraph(
            "The pilot diagnoses vulnerability; it is then discarded. Every final condition restarts from the same base model.",
            core.BODY,
        ),
        Spacer(1, 3 * mm),
        flow_diagram(),
        PageBreak(),
    ]


class StudyDoc(core.ReportDoc):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.title = "Damage-Aware KL Anchoring for Small Language Models"
        self.author = "Beginner-first project study guide"
        self.subject = "LLM fine-tuning, catastrophic forgetting, KL anchoring, and experimental design"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    core.SOURCE = SOURCE
    core.OUTPUT = OUTPUT
    core.cover_story = study_cover_story
    core.ReportDoc = StudyDoc
    core.main()


if __name__ == "__main__":
    main()
