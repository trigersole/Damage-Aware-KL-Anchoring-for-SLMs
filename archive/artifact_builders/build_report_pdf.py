from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "report-source.md"
OUTPUT = ROOT / "outputs" / "llm_forgetting_sota_report.pdf"

PAGE_W, PAGE_H = A4
LEFT = 17 * mm
RIGHT = 17 * mm
TOP = 17 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT

NAVY = colors.HexColor("#173652")
BLUE = colors.HexColor("#2F6B91")
TEAL = colors.HexColor("#3A7D78")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_TEAL = colors.HexColor("#EAF5F2")
PALE_GOLD = colors.HexColor("#FFF4D6")
INK = colors.HexColor("#24313B")
MUTED = colors.HexColor("#5C6B78")
GRID = colors.HexColor("#B8C8D3")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReportSans-Italic", str(italic)))
            return "ReportSans", "ReportSans-Bold", "ReportSans-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


REGULAR, BOLD, ITALIC = register_fonts()


def sanitize(value: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2192": "->",
        "\u2194": "<->",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def inline_markup(text: str) -> str:
    text = sanitize(text.strip())
    links: list[tuple[str, str]] = []

    def keep_link(match: re.Match[str]) -> str:
        links.append((match.group(1), match.group(2)))
        return f"@@LINK{len(links) - 1}@@"

    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", keep_link, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    for idx, (label, url) in enumerate(links):
        safe_label = html.escape(sanitize(label), quote=False)
        safe_url = html.escape(url, quote=True)
        text = text.replace(
            f"@@LINK{idx}@@",
            f"<link href='{safe_url}' color='#2F6B91'><u>{safe_label}</u></link>",
        )
    return text


styles = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontName=BOLD,
    fontSize=28,
    leading=32,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=12,
)
SUBTITLE = ParagraphStyle(
    "ReportSubtitle",
    parent=styles["Normal"],
    fontName=REGULAR,
    fontSize=15,
    leading=20,
    textColor=BLUE,
    spaceAfter=8,
)
KICKER = ParagraphStyle(
    "Kicker",
    parent=styles["Normal"],
    fontName=BOLD,
    fontSize=9,
    leading=12,
    textColor=TEAL,
    uppercase=True,
    spaceAfter=5,
)
H1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontName=BOLD,
    fontSize=17,
    leading=21,
    textColor=NAVY,
    spaceBefore=12,
    spaceAfter=7,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontName=BOLD,
    fontSize=12.5,
    leading=16,
    textColor=BLUE,
    spaceBefore=9,
    spaceAfter=5,
    keepWithNext=True,
)
H3 = ParagraphStyle(
    "H3",
    parent=styles["Heading3"],
    fontName=BOLD,
    fontSize=10.4,
    leading=13,
    textColor=TEAL,
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True,
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName=REGULAR,
    fontSize=8.8,
    leading=12.2,
    textColor=INK,
    alignment=TA_LEFT,
    spaceAfter=4.5,
)
BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=12,
    firstLineIndent=-7,
    bulletIndent=2,
    spaceAfter=2.8,
)
NUMBERED = ParagraphStyle(
    "Numbered",
    parent=BODY,
    leftIndent=16,
    firstLineIndent=-12,
    spaceAfter=2.8,
)
CALLOUT = ParagraphStyle(
    "Callout",
    parent=BODY,
    fontName=BOLD,
    fontSize=10.2,
    leading=14,
    textColor=NAVY,
    borderColor=TEAL,
    borderWidth=1.2,
    borderPadding=9,
    backColor=PALE_TEAL,
    spaceBefore=6,
    spaceAfter=9,
)
CAPTION = ParagraphStyle(
    "Caption",
    parent=BODY,
    fontName=ITALIC,
    fontSize=7.7,
    leading=10,
    textColor=MUTED,
)


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="Catastrophic Forgetting and Capability Degradation in LLM Fine-Tuning",
            author="Systematic course-project review",
            subject="State of the art, research gaps, and eight-week project proposals",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="main")
        self.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page > 1:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.35)
            canvas.line(LEFT, PAGE_H - 11 * mm, PAGE_W - RIGHT, PAGE_H - 11 * mm)
            canvas.setFont(REGULAR, 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, PAGE_H - 8.4 * mm, "LLM fine-tuning and catastrophic forgetting")
            canvas.drawRightString(PAGE_W - RIGHT, 8.5 * mm, f"{page}")
        canvas.restoreState()


def table_widths(rows: list[list[str]]) -> list[float]:
    n = len(rows[0])
    if n >= 9:
        first = 25 if n == 10 else 38
        second = 145 if n == 10 else 130
        remaining = (CONTENT_W - first - second) / (n - 2)
        return [first, second] + [remaining] * (n - 2)
    if n == 8:
        return [60, 80, 72, 62, 58, 58, 55, CONTENT_W - 445]
    if n == 4:
        return [118, 92, 60, CONTENT_W - 270]
    if n == 3:
        return [145, 105, CONTENT_W - 250]
    if n == 2:
        return [155, CONTENT_W - 155]
    return [CONTENT_W / n] * n


def make_table(raw_rows: list[list[str]]) -> Table:
    ncols = len(raw_rows[0])
    if ncols >= 9:
        font_size, leading = 5.5, 7.0
    elif ncols >= 6:
        font_size, leading = 6.1, 7.6
    elif ncols == 4:
        font_size, leading = 6.8, 8.5
    else:
        font_size, leading = 7.2, 9.0
    cell = ParagraphStyle(
        f"TableCell{ncols}",
        parent=BODY,
        fontName=REGULAR,
        fontSize=font_size,
        leading=leading,
        spaceAfter=0,
    )
    header = ParagraphStyle(
        f"TableHeader{ncols}",
        parent=cell,
        fontName=BOLD,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    data = []
    for ridx, row in enumerate(raw_rows):
        normalized = row + [""] * (ncols - len(row))
        data.append([
            Paragraph(inline_markup(value), header if ridx == 0 else cell)
            for value in normalized[:ncols]
        ])
    table = Table(
        data,
        colWidths=table_widths(raw_rows),
        repeatRows=1,
        hAlign="LEFT",
        spaceBefore=4,
        spaceAfter=8,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for ridx in range(1, len(data)):
        if ridx % 2 == 1:
            commands.append(("BACKGROUND", (0, ridx), (-1, ridx), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def parse_table(lines: list[str], start: int) -> tuple[Table, int]:
    rows: list[list[str]] = []
    idx = start
    while idx < len(lines) and lines[idx].lstrip().startswith("|"):
        row = [part.strip() for part in lines[idx].strip().strip("|").split("|")]
        rows.append(row)
        idx += 1
    if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        rows.pop(1)
    return make_table(rows), idx


def cover_story() -> list:
    return [
        Spacer(1, 26 * mm),
        Paragraph("SYSTEMATIC STATE-OF-THE-ART REVIEW", KICKER),
        Paragraph("Catastrophic Forgetting and Capability Degradation in LLM Fine-Tuning", TITLE),
        Paragraph("Research gaps and realistic eight-week projects for a 3-4 person student team", SUBTITLE),
        Spacer(1, 11 * mm),
        Table(
            [[
                Paragraph("180", ParagraphStyle("Big1", parent=TITLE, fontSize=22, leading=24, alignment=TA_CENTER)),
                Paragraph("60", ParagraphStyle("Big2", parent=TITLE, fontSize=22, leading=24, alignment=TA_CENTER)),
                Paragraph("11", ParagraphStyle("Big3", parent=TITLE, fontSize=22, leading=24, alignment=TA_CENTER)),
            ], [
                Paragraph("papers screened", ParagraphStyle("Small1", parent=CAPTION, alignment=TA_CENTER)),
                Paragraph("papers close-reviewed", ParagraphStyle("Small2", parent=CAPTION, alignment=TA_CENTER)),
                Paragraph("project directions scored", ParagraphStyle("Small3", parent=CAPTION, alignment=TA_CENTER)),
            ]],
            colWidths=[CONTENT_W / 3] * 3,
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
                ("TOPPADDING", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
            ]),
        ),
        Spacer(1, 13 * mm),
        Paragraph(
            "Recommended project: <b>Replay or Re-alignment? Damage-Aware KL Anchoring for Small Language Models</b>",
            CALLOUT,
        ),
        Paragraph(
            "Main conclusion: a plain LoRA-versus-full-fine-tuning comparison is now a replication. "
            "The strongest course-sized contribution is to make anchor selection and recovery-aware measurement the research question.",
            BODY,
        ),
        Spacer(1, 29 * mm),
        Paragraph("Coverage through 4 September 2026", CAPTION),
        Paragraph("Designed for 0.5B-3B open models, 16/24/48 GB GPU plans, and roughly 10-20 training runs", CAPTION),
        PageBreak(),
    ]


def build_story(markdown: str) -> list:
    lines = markdown.splitlines()
    story = cover_story()
    idx = 0
    skipped_title = False
    while idx < len(lines):
        raw = lines[idx].rstrip()
        stripped = raw.strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("|"):
            table, idx = parse_table(lines, idx)
            story.append(table)
            continue
        if stripped.startswith("# ") and not skipped_title:
            skipped_title = True
            idx += 1
            continue
        if stripped == "## State of the art, research gaps, and eight-week project proposals":
            idx += 1
            continue
        if stripped.startswith("## "):
            heading = stripped[3:]
            if heading.startswith("Proposal ") or heading in {"Selected evidence notes", "Final cautions"}:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(heading), H1))
            idx += 1
            continue
        if stripped.startswith("### "):
            heading = stripped[4:]
            style = H2 if not heading.startswith("Candidate title") else H3
            story.append(Paragraph(inline_markup(heading), style))
            idx += 1
            continue
        bullet = re.match(r"^-\s+(.*)$", stripped)
        if bullet:
            story.append(Paragraph(inline_markup(bullet.group(1)), BULLET, bulletText="•"))
            idx += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            story.append(
                Paragraph(f"<b>{numbered.group(1)}.</b> {inline_markup(numbered.group(2))}", NUMBERED)
            )
            idx += 1
            continue
        paragraph_lines = [stripped]
        idx += 1
        while idx < len(lines):
            nxt = lines[idx].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or re.match(r"^-\s+", nxt)
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            paragraph_lines.append(nxt)
            idx += 1
        content = " ".join(paragraph_lines)
        style = CALLOUT if content.startswith("**Recommended project:") else BODY
        story.append(Paragraph(inline_markup(content), style))
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    markdown = SOURCE.read_text(encoding="utf-8")
    doc = ReportDoc(str(OUTPUT))
    story = build_story(markdown)
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
