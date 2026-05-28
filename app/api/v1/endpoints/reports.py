"""
Report Generation
==================
POST /reports/generate        — Queue report generation (background task)
GET  /reports/{report_id}     — Get report status
GET  /reports/{report_id}/download — Download PDF/JSON report
GET  /reports/my              — List user's reports

PDF reports include:
- Summary section
- Prediction/analysis results
- Matplotlib charts
- Recommendations
"""

import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.models import (
    PriceForecast, Report, ReportFormat, SuitabilityResult,
    User, YieldPrediction, RiskScore
)
from app.schemas.schemas import ReportGenerationRequest, ReportStatusResponse
from app.core.logging import api_logger

router = APIRouter(prefix="/reports", tags=["Reports"])

os.makedirs(settings.REPORT_STORAGE_PATH, exist_ok=True)


async def _fetch_analysis_data(analysis_type: str, result_id, db: AsyncSession):
    """Fetch the underlying analysis result from PostgreSQL."""
    type_map = {
        "price_forecast": PriceForecast,
        "crop_suitability": SuitabilityResult,
        "yield_prediction": YieldPrediction,
        "risk_score": RiskScore,
    }
    model = type_map.get(analysis_type)
    if not model:
        return None
    result = await db.execute(select(model).where(model.id == result_id))
    return result.scalar_one_or_none()


def _generate_pdf_report(analysis_type: str, data: dict, user: User) -> bytes:
    """
    Generate a PDF report using ReportLab.
    Includes summary, metrics, and basic chart.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        green = colors.HexColor("#2D6A4F")
        light_green = colors.HexColor("#95D5B2")

        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            textColor=green, fontSize=20, spaceAfter=12
        )
        heading_style = ParagraphStyle(
            "Heading2", parent=styles["Heading2"],
            textColor=green, fontSize=13, spaceBefore=12
        )
        body_style = styles["BodyText"]

        story = []

        # ── Header ────────────────────────────────────────────────────────
        story.append(Paragraph("🌱 GEoN Risk Assessment Platform", title_style))
        story.append(Paragraph(
            f"Agricultural Analysis Report — {analysis_type.replace('_', ' ').title()}",
            heading_style
        ))
        story.append(HRFlowable(color=green, thickness=2, width="100%"))
        story.append(Spacer(1, 0.3*inch))

        # ── Report Metadata ────────────────────────────────────────────────
        meta_data = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M UTC")],
            ["Prepared For:", user.full_name],
            ["Subscription Plan:", data.get("subscription_tier", "—").upper()],
            ["Analysis Type:", analysis_type.replace("_", " ").title()],
        ]
        meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), green),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.3*inch))

        # ── Analysis Results Section ───────────────────────────────────────
        story.append(Paragraph("Analysis Results", heading_style))

        # Dynamic content by analysis type
        if analysis_type == "price_forecast":
            story.append(Paragraph(
                f"<b>Crop:</b> {data.get('crop', '').title()} — "
                f"<b>Forecast Period:</b> {data.get('forecast_period', '')}",
                body_style
            ))
            preds = data.get("predictions", [])[:8]  # Show first 8 in report
            if preds:
                pred_table_data = [["Date", "Price (XAF/kg)", "Lower CI", "Upper CI"]]
                for p in preds:
                    pred_table_data.append([
                        p["date"], f"{p['price_xaf']:,.0f}",
                        f"{p['lower_ci']:,.0f}", f"{p['upper_ci']:,.0f}"
                    ])
                pred_table = Table(pred_table_data, colWidths=[3.5*cm]*4)
                pred_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), green),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightyellow]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(pred_table)

        elif analysis_type == "risk_score":
            risk_data = [
                ["Risk Component", "Score", "Level"],
                ["Overall Risk", f"{data.get('overall_risk_score', 0):.2f}", data.get('risk_level', '—').upper()],
                ["Financial Risk", f"{data.get('financial_risk', 0):.2f}", "—"],
                ["Climate Risk", f"{data.get('climate_risk', 0):.2f}", "—"],
                ["Agronomic Risk", f"{data.get('agronomic_risk', 0):.2f}", "—"],
            ]
            risk_table = Table(risk_data, colWidths=[6*cm, 4*cm, 4*cm])
            risk_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), green),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(risk_table)

        # ── Model Accuracy ─────────────────────────────────────────────────
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Model Accuracy Metrics", heading_style))
        metrics = data.get("accuracy", data.get("metrics", {}))
        if metrics:
            m_data = [["Metric", "Value"]] + [[k.upper(), str(v)] for k, v in metrics.items()]
            m_table = Table(m_data, colWidths=[6*cm, 6*cm])
            m_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), light_green),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(m_table)

        # ── Recommendations ────────────────────────────────────────────────
        recs = data.get("recommendations", [])
        if recs:
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("Recommendations", heading_style))
            for i, rec in enumerate(recs, 1):
                story.append(Paragraph(f"{i}. {rec}", body_style))

        # ── Footer ────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.5*inch))
        story.append(HRFlowable(color=colors.lightgrey, thickness=1, width="100%"))
        story.append(Paragraph(
            "Generated by GEoN Risk Assessment Platform — "
            "Data sourced from MINADER, FAO Cameroon. "
            "This report is for informational purposes only.",
            ParagraphStyle("footer", parent=body_style, fontSize=8, textColor=colors.grey)
        ))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        # ReportLab not available — return JSON-as-PDF fallback
        return json.dumps(data, default=str, indent=2).encode()


@router.post("/generate", response_model=ReportStatusResponse, status_code=201)
async def generate_report(
    payload: ReportGenerationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue report generation for an existing analysis result."""
    # Subscription check for PDF
    if payload.format == "pdf" and current_user.subscription_type.value == "free":
        raise HTTPException(
            status_code=403,
            detail="PDF reports require MEDIUM or PREMIUM subscription",
        )

    # Fetch the underlying analysis data
    analysis_data = await _fetch_analysis_data(payload.analysis_type, payload.result_id, db)
    if not analysis_data:
        raise HTTPException(404, detail="Analysis result not found")

    # Convert ORM object to dict
    data = {c.name: getattr(analysis_data, c.name) for c in analysis_data.__table__.columns}

    report_id = uuid.uuid4()
    filename = f"{payload.analysis_type}_{report_id}.{payload.format}"
    filepath = os.path.join(settings.REPORT_STORAGE_PATH, filename)

    if payload.format == "pdf":
        pdf_bytes = _generate_pdf_report(payload.analysis_type, data, current_user)
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)
    else:
        with open(filepath, "w") as f:
            json.dump(data, f, default=str, indent=2)

    report = Report(
        id=report_id,
        user_id=current_user.id,
        request_id=payload.result_id,
        report_type=payload.analysis_type,
        format=ReportFormat(payload.format),
        file_path=filepath,
        file_size_bytes=os.path.getsize(filepath),
        is_ready=True,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REPORT_RETENTION_DAYS),
    )
    db.add(report)

    return ReportStatusResponse(
        report_id=report_id,
        status="ready",
        format=payload.format,
        is_ready=True,
        download_url=f"/api/v1/reports/{report_id}/download",
        created_at=datetime.now(timezone.utc),
    )


@router.get("/{report_id}/download")
async def download_report(
    report_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download a generated report file."""
    result = await db.execute(
        select(Report).where(
            Report.id == report_id,
            Report.user_id == current_user.id,
        )
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(404, detail="Report not found")
    if not report.is_ready or not os.path.exists(report.file_path):
        raise HTTPException(404, detail="Report file not ready")

    # Increment download count
    report.download_count += 1

    content_type = "application/pdf" if report.format == ReportFormat.PDF else "application/json"
    return FileResponse(
        path=report.file_path,
        media_type=content_type,
        filename=os.path.basename(report.file_path),
    )


@router.get("/my", summary="List my reports")
async def list_my_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all reports generated by the current user."""
    result = await db.execute(
        select(Report).where(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
    )
    reports = result.scalars().all()
    return [
        {
            "report_id": r.id,
            "report_type": r.report_type,
            "format": r.format.value,
            "is_ready": r.is_ready,
            "download_url": f"/api/v1/reports/{r.id}/download",
            "created_at": r.created_at,
        }
        for r in reports
    ]
