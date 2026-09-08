from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY

doc = SimpleDocTemplate(
    "concept_summary.pdf",
    pagesize=letter,
    topMargin=0.4 * inch,
    bottomMargin=0.4 * inch,
    leftMargin=0.6 * inch,
    rightMargin=0.6 * inch,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=13, leading=15.5, spaceAfter=3)
sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=7)
h_style = ParagraphStyle("HeadX", parent=styles["Heading3"], fontSize=10, spaceBefore=5, spaceAfter=2)
body_style = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=8.8, leading=11.3, alignment=TA_JUSTIFY, spaceAfter=4)
small_style = ParagraphStyle("SmallX", parent=styles["Normal"], fontSize=7.5, leading=9.5, textColor=colors.grey, spaceBefore=3)

story = []

story.append(Paragraph(
    "Sparse Codes Make Linear Attention Trustworthy: A Graph-Structured View of BDH-GPU's Attention Mechanism",
    title_style,
))
story.append(Paragraph(
    "Concepts: Linear Attention &middot; Sparse Non-Negative Activations &middot; Local Neural Computation, on graph-structured data &middot; connected to BDH-GPU (Dragon Hatchling)",
    sub_style,
))

story.append(Paragraph("The design pressure", h_style))
story.append(Paragraph(
    "Standard softmax attention compares every pair of tokens directly, giving robust retrieval at the cost of a "
    "key-value cache that grows with context and O(n) work per new query against everything stored so far. "
    "Linear attention removes the softmax nonlinearity and replaces pairwise comparison with an incrementally "
    "updated, fixed-size matrix &mdash; a fast-weight associative memory queried in O(1) per token (Katharopoulos "
    "et al., 2020). The saving is real, but so is the cost: compressing many stored associations into one fixed-size "
    "matrix means they begin to interfere with each other, a capacity limit long known for linear associative "
    "memories. BDH-GPU (Kosowski et al., 2025) takes a specific, unusual position on this trade-off: rather than "
    "compressing keys into a small hidden dimension before applying linear attention, it runs linear attention "
    "directly in a very large neuron dimension n, forcing activations positive via ReLU, and reports them "
    "empirically sparse at roughly 5% active (Kosowski et al., 2025, Sec. 4.1).",
    body_style,
))

story.append(Paragraph("What changes, and the trade-off it introduces", h_style))
story.append(Paragraph(
    "The mechanistic difference is where crosstalk gets controlled. Dense, low-dimensional linear attention accepts "
    "interference risk in exchange for a compact state. BDH-GPU instead expands into a high-dimensional but sparse "
    "code, so different stored associations are more likely to land on largely disjoint sets of active units &mdash; "
    "the same argument used to explain capacity in sparse, high-dimensional associative memories, echoing sparse "
    "coding accounts of the visual cortex (Olshausen and Field, 1997, cited in Kosowski et al., 2025). The cost is a "
    "larger state footprint per neuron, and a sparsity level that must land in a workable range: too dense and the "
    "scheme reduces to the interference-prone case; too sparse and there is not enough active capacity to represent "
    "much of anything.",
    body_style,
))

story.append(Paragraph("Representative architectures, compared", h_style))

table_data = [
    ["", "Softmax attention\n(Transformer)", "Linear attention,\nlow-dim (Katharopoulos\net al., 2020 lineage)", "BDH-GPU\n(Kosowski et al., 2025)"],
    ["Per-token cost", "O(context length)", "O(1) after O(1) update", "O(1) after O(1) update"],
    ["State", "Raw KV-cache\n(grows with context)", "Fixed small dense\nmatrix", "Fixed large sparse\n(~5% active) matrix"],
    ["Interference risk", "None (recomputed\nfrom raw keys)", "High at scale\n(shown in our toy test)", "Low at same scale\n(shown in our toy test)"],
    ["Interpretability", "Post-hoc only", "Post-hoc only", "Reported monosemantic\nsynapses (Sec. 6.3)"],
    ["Evidence status", "Industry standard", "Established literature", "Single-group reported\n(see below)"],
]
tbl = Table(table_data, colWidths=[0.95*inch, 1.55*inch, 1.75*inch, 1.75*inch])
tbl.setStyle(TableStyle([
    ("FONTSIZE", (0,0), (-1,-1), 7.0),
    ("LEADING", (0,0), (-1,-1), 8.2),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("TOPPADDING", (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
]))
story.append(tbl)
story.append(Spacer(1, 3))

story.append(Paragraph("Evidence, labeled correctly", h_style))
story.append(Paragraph(
    "BDH-GPU's headline result &mdash; matching GPT2-class Transformer scaling laws from 10M to 1B parameters on "
    "language and translation tasks &mdash; is a developer-reported result in the primary paper (Kosowski et al., "
    "2025, Sec. 4.2), not yet an independent third-party reproduction. Its widely cited 97.4% Sudoku Extreme result "
    "is explicitly flagged, including in Pathway's own open-source repository, as coming from an internal "
    "implementation distinct from the released code, and is not independently reproduced. On the specific "
    "sparsity-versus-interference link this summary is about, our own small synthetic retrieval experiment (not a "
    "BDH reproduction) shows retrieval accuracy falling from about 100% to about 5% as memory load exceeds the "
    "embedding dimension for a dense linear-attention code, while a sparse (~5% active) code stays above 95% "
    "accuracy at the same load &mdash; a controlled illustration of the mechanism, not a claim about trained BDH "
    "models.",
    body_style,
))

story.append(Paragraph("Roles of BDH and BDH-CQ", h_style))
story.append(Paragraph(
    "BDH-GPU is the direct, substantive anchor for this concept: its attention state is described by the primary "
    "paper itself as functioning like an associative memory (\"like KV-cache, but organized differently,\" Sec. 4.1), "
    "built and queried exactly as a fast-weight linear-attention mechanism. BDH-CQ, a later system in the same "
    "family focused on gradient-free adaptation from demonstrations, has no direct mechanistic role in the "
    "linear-attention/sparsity story specifically &mdash; we note this rather than inventing a connection the "
    "source material does not support.",
    body_style,
))

story.append(Paragraph("The most important open question", h_style))
story.append(Paragraph(
    "Our demonstration uses synthetic, independent random codes, for which near-orthogonality is guaranteed by "
    "construction. Whether the capacity benefit survives when codes emerge from training on real, correlated data "
    "&mdash; where keys are not independent and random &mdash; is not established here and is, to our knowledge, "
    "not fully settled by BDH-GPU's own results either. A second gap: no outside group has, to our knowledge, "
    "independently reproduced BDH-GPU's headline scaling-law or Sudoku findings, so confidence in the practical "
    "performance claims currently rests on a single research group's own reporting.",
    body_style,
))

story.append(Paragraph(
    "Primary sources: Kosowski, Uznanski, Chorowski, Stamirowska, Bartoszkiewicz, \"The Dragon Hatchling: The "
    "Missing Link between the Transformer and Models of the Brain,\" arXiv:2509.26507 (2025) &middot; Katharopoulos "
    "et al., \"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention,\" ICML 2020 &middot; "
    "Haziza et al. (2025) &middot; You et al., \"Spark Transformer\" (2025) &middot; Buckman et al. (2024).",
    small_style,
))

doc.build(story)
print("PDF built.")
