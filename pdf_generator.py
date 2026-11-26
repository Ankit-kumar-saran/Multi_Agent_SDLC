# pdf_generator.py
"""Automated PDF Generator for Financial Requirements Engineering & SDLC Analysis."""

import html
import os
from typing import Any, Dict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to draw dynamic page numbers, headers, and footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        if self._pageNumber > 1:
            self.drawString(54, 750, "AGENTIC RE-SDLC ADVISOR — SOFTWARE REQUIREMENTS SPECIFICATION")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        self.drawRightString(558, 36, f"Page {self._pageNumber} of {page_count}")
        self.drawString(54, 36, "Financial Software Requirements Engineering & SDLC Governance")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()


def _esc(val: Any) -> str:
    return html.escape(str(val or ""))


def generate_pdf_report(
    analysis_data: Dict[str, Any],
    output_path: str = "output/requirements_analysis.pdf",
) -> str:
    """Compile a multi-page SRS PDF from dynamically extracted analysis_data."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    primary_color = colors.HexColor("#0F172A")
    secondary_color = colors.HexColor("#1E3A8A")
    accent_color = colors.HexColor("#0284C7")
    text_color = colors.HexColor("#1E293B")
    border_color = colors.HexColor("#CBD5E1")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, textColor=primary_color, spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10, leading=14, textColor=secondary_color, spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "Heading1", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=12, leading=16, textColor=primary_color, spaceBefore=12, spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "Heading2", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=10, leading=14, textColor=secondary_color, spaceBefore=8, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyText", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=12, textColor=text_color,
    )
    table_hdr_style = ParagraphStyle(
        "TableHdr", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=colors.white,
    )
    table_cell_style = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7.5, leading=10, textColor=text_color,
    )

    risk_lvl = str(analysis_data.get("risk_level", "MEDIUM")).upper()
    risk_score = analysis_data.get("risk_score_pct", 65)
    risk_summary = _esc(analysis_data.get("risk_summary", "Multi-factor project risk evaluation."))
    risk_color = {
        "CRITICAL": colors.HexColor("#B91C1C"),
        "HIGH": colors.HexColor("#C2410C"),
        "MEDIUM": colors.HexColor("#B45309"),
        "LOW": colors.HexColor("#047857"),
    }.get(risk_lvl, colors.HexColor("#B45309"))

    dynamic_tag_style = ParagraphStyle(
        "DynamicTagStyle", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=risk_color,
    )

    story = []
    story.append(Paragraph(_esc(analysis_data.get("title", "Software Requirements Specification")), title_style))
    story.append(Paragraph(f"Domain: {_esc(analysis_data.get('domain', 'Financial Software Systems'))}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=0, spaceAfter=10))

    reqs_list = analysis_data.get("requirements_list", [])
    categories_count = len(analysis_data.get("categories_breakdown", {}))

    meta_data = [
        [
            Paragraph("<b>Parameter</b>", table_hdr_style),
            Paragraph("<b>Value</b>", table_hdr_style),
            Paragraph("<b>Rationale & Scope</b>", table_hdr_style),
        ],
        [
            Paragraph("<b>MoSCoW Baseline</b>", table_cell_style),
            Paragraph(f"<b>{_esc(analysis_data.get('moscow_priority', 'Must Have'))}</b>", table_cell_style),
            Paragraph(_esc(analysis_data.get("priority_justification", "")), table_cell_style),
        ],
        [
            Paragraph("<b>Risk Classification</b>", table_cell_style),
            Paragraph(f"<b>{_esc(risk_lvl)} ({_esc(risk_score)}/100)</b>", dynamic_tag_style),
            Paragraph(risk_summary, table_cell_style),
        ],
        [
            Paragraph("<b>Extracted Scope</b>", table_cell_style),
            Paragraph(f"<b>{len(reqs_list)} Requirements</b>", table_cell_style),
            Paragraph(f"Organized across {categories_count} engineering categories", table_cell_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[110, 130, 264])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 1. Stakeholder Request & Summary
    story.append(Paragraph("1. System Scope & Stakeholder Request", h1_style))
    raw_req = _esc(analysis_data.get("original_requirement", "")).strip()
    callout_text = raw_req[:500] + ("..." if len(raw_req) > 500 else "")
    req_box = Table([[Paragraph(f'<i>"{callout_text}"</i>', body_style)]], colWidths=[504])
    req_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINELEFT", (0, 0), (0, -1), 3, accent_color),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
    ]))
    story.append(req_box)
    if analysis_data.get("system_summary"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Summary:</b> {_esc(analysis_data.get('system_summary'))}", body_style))
    story.append(Spacer(1, 8))

    # 2. Modular Requirements
    if reqs_list:
        story.append(Paragraph(f"2. Software Requirements Specification ({len(reqs_list)} Requirements)", h1_style))
        categories: Dict[str, list] = {}
        for r in reqs_list:
            categories.setdefault(r.get("category", "General"), []).append(r)

        for cat_name, cat_reqs in categories.items():
            story.append(Paragraph(f"<b>{_esc(cat_name)} ({len(cat_reqs)})</b>", h2_style))
            cat_rows = [[
                Paragraph("<b>ID</b>", table_hdr_style),
                Paragraph("<b>Title & Specification</b>", table_hdr_style),
                Paragraph("<b>Priority</b>", table_hdr_style),
                Paragraph("<b>Standard / Acceptance Criteria</b>", table_hdr_style),
            ]]
            for cr in cat_reqs:
                desc_text = f"<b>{_esc(cr.get('title', ''))}</b><br/>{_esc(cr.get('description', ''))}"
                rule_text = (
                    f"<b>Ref:</b> {_esc(cr.get('regulation', ''))}<br/>"
                    f"<b>Acceptance:</b> <i>{_esc(cr.get('acceptance_criteria', ''))}</i>"
                )
                cat_rows.append([
                    Paragraph(f"<b>{_esc(cr.get('id', ''))}</b>", table_cell_style),
                    Paragraph(desc_text, table_cell_style),
                    Paragraph(f"<b>{_esc(cr.get('priority', 'Must Have'))}</b>", table_cell_style),
                    Paragraph(rule_text, table_cell_style),
                ])

            cat_table = Table(cat_rows, colWidths=[58, 185, 55, 206])
            cat_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, border_color),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
            ]))
            story.append(cat_table)
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    # 3. Applicable Regulations & Policies
    regs = analysis_data.get("regulations", [])
    if regs:
        story.append(Paragraph("3. Applicable Regulations, Policies & Standards", h1_style))
        reg_rows = [[
            Paragraph("<b>Standard / Regulation</b>", table_hdr_style),
            Paragraph("<b>Relevance</b>", table_hdr_style),
            Paragraph("<b>Citation / Source</b>", table_hdr_style),
        ]]
        for r in regs:
            reg_rows.append([
                Paragraph(f"<b>{_esc(r.get('regulation', ''))}</b>", table_cell_style),
                Paragraph(_esc(r.get("relevance", "")), table_cell_style),
                Paragraph(_esc(r.get("statutory_citation", "")), table_cell_style),
            ])
        reg_table = Table(reg_rows, colWidths=[140, 160, 204])
        reg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
        ]))
        story.append(reg_table)
        story.append(Spacer(1, 10))

    # 4. Technical & Security Controls
    controls = analysis_data.get("technical_controls", [])
    if controls:
        story.append(Paragraph("4. Security & Technical Controls", h1_style))
        ctrl_rows = [[
            Paragraph("<b>Control ID</b>", table_hdr_style),
            Paragraph("<b>Specification</b>", table_hdr_style),
            Paragraph("<b>Verification Method</b>", table_hdr_style),
        ]]
        for c in controls:
            spec_text = f"<b>{_esc(c.get('title', ''))}</b><br/>{_esc(c.get('specification', ''))}"
            ctrl_rows.append([
                Paragraph(f"<b>{_esc(c.get('control_id', ''))}</b>", table_cell_style),
                Paragraph(spec_text, table_cell_style),
                Paragraph(_esc(c.get("verification", "")), table_cell_style),
            ])
        ctrl_table = Table(ctrl_rows, colWidths=[75, 245, 184])
        ctrl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
        ]))
        story.append(ctrl_table)
        story.append(Spacer(1, 10))

    # 5. Clarification Questions / Open Ambiguities
    clarifications = analysis_data.get("clarification_questions", [])
    if clarifications:
        story.append(Paragraph("5. Open Ambiguities & Clarification Questions", h1_style))
        for idx, q in enumerate(clarifications, 1):
            story.append(Paragraph(f"{idx}. {_esc(q)}", body_style))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 8))

    # 6. Traceability Matrix
    trace_rows_data = analysis_data.get("traceability_matrix", [])
    if trace_rows_data:
        story.append(Paragraph("6. Requirements Traceability Matrix", h1_style))
        trace_rows = [[
            Paragraph("<b>Req IDs</b>", table_hdr_style),
            Paragraph("<b>Module / Category</b>", table_hdr_style),
            Paragraph("<b>SDLC Phase</b>", table_hdr_style),
            Paragraph("<b>Verification Deliverable</b>", table_hdr_style),
        ]]
        for t in trace_rows_data:
            trace_rows.append([
                Paragraph(f"<b>{_esc(t.get('req_id', ''))}</b>", table_cell_style),
                Paragraph(_esc(t.get("sub_module", "")), table_cell_style),
                Paragraph(_esc(t.get("sdlc_phase", "")), table_cell_style),
                Paragraph(
                    f"{_esc(t.get('target_artefact', ''))}<br/><i>{_esc(t.get('verification_method', ''))}</i>",
                    table_cell_style,
                ),
            ])
        trace_table = Table(trace_rows, colWidths=[80, 135, 115, 174])
        trace_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
        ]))
        story.append(trace_table)

    try:
        doc.build(story, canvasmaker=NumberedCanvas)
    except Exception as exc:
        print(f"[WARN] PDF canvas fallback: {exc}")
        doc.build(story)
    return output_path
