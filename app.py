import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader

# ========== PATHS ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLIENT_DIR = os.path.join(BASE_DIR, "clients")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

for d in [DATA_DIR, CLIENT_DIR, ASSETS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

st.set_page_config(page_title="Surf Evaluation", layout="wide")

# ========== HELPERS ==============
def load_topics(path):
    try:
        df = pd.read_csv(path, usecols=[0,1,2])
        df.columns = ["Category","Topic","Ranking"]
        df["Ranking"] = pd.to_numeric(df["Ranking"], errors="coerce").fillna(1).astype(int)
        return df
    except:
        return pd.DataFrame(columns=["Category","Topic","Ranking"])

def client_filename(n):
    return os.path.join(CLIENT_DIR, f"{n.replace(' ','_')}.csv")

def save_client_file(client, email, phone, form, df, videos=None):
    fn = client_filename(client)
    with open(fn,"w",encoding="utf-8") as f:
        f.write(f"# ClientName:{client}\n# Email:{email}\n# Phone:{phone}\n# Form:{form}\n# SavedAt:{datetime.datetime.now()}\n")
        if df is not None and not df.empty:
            df.to_csv(f,index=False)
    if videos:
        for v in videos:
            with open(os.path.join(CLIENT_DIR, f"{client}_{v.name}"),"wb") as o:
                o.write(v.getbuffer())

def load_client_file(path):
    meta = {}
    data=[]
    with open(path) as f:
        for line in f:
            if line.startswith("# "):
                k,v=line[2:].split(":",1)
                meta[k]=v.strip()
            else:
                data.append(line)

    if data:
        try:
            df = pd.read_csv(pd.io.common.StringIO("".join(data)))
        except:
            df = pd.DataFrame(columns=["Category","Topic","Ranking","Report","Day1","Day2","Day3"])
    else:
        df = pd.DataFrame(columns=["Category","Topic","Ranking","Report","Day1","Day2","Day3"])

    # Ensure columns exist and types are sane
    if "Report" not in df.columns: df["Report"] = "No"
    for dcol in ["Day1","Day2","Day3"]:
        if dcol not in df.columns:
            df[dcol] = 0
        df[dcol] = pd.to_numeric(df[dcol], errors="coerce").fillna(0).astype(int)
    return meta,df

def list_clients():
    return sorted([x.replace(".csv","") for x in os.listdir(CLIENT_DIR) if x.endswith(".csv")])

# ========== RADAR CHART (FIXED FOR 0 VALUES) ==========
# ========== RADAR CHART (OUTLINE ONLY, NO FILL) ==========
def draw_radar(df, title):
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Sicherheitscheck ---
    if df is None or df.empty or not all(c in df.columns for c in ["Day1", "Day2", "Day3"]):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = df.index.tolist()
    N = len(labels)
    if N == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return fig

    # --- Radar Geometrie ---
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Schleife schließen

    # --- Plot erstellen ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors_map = {"Day1": "#e11d48", "Day2": "#2563eb", "Day3": "#16a34a"}  # rot, blau, grün

    # Nur Linien zeichnen (keine Füllung)
    for day in ["Day1", "Day2", "Day3"]:
        vals = df[day].fillna(0).astype(float).tolist()
        vals += vals[:1]  # Kreis schließen
        ax.plot(angles, vals, linewidth=2, label=day, color=colors_map[day])

    # --- Layout & Beschriftung ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, fontweight="regular")
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title(title, fontsize=12, pad=25)

    # Legende außerhalb
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8, frameon=False)

    # Weniger Rand
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.08)
    return fig

def fig_png(fig):
    b = BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    b.seek(0)
    return b

# ========== TEXT MAPPING (FLUENT, VARIED & KEYWORD-PRESERVING) ==========
import random

def coach_sentence(topic, score):
    """Create fluid, human coaching text using exact keywords."""
    topic = str(topic).strip()
    if not topic:
        topic = "this skill"
    topic_lc = topic[0].lower() + topic[1:] if topic else topic

    try:
        score = int(pd.to_numeric(score, errors="coerce"))
    except:
        score = 0
    score = max(0, min(5, score))

    kw_map = {
        0: "not yet performed",
        1: "never",
        2: "sometimes",
        3: "usually",
        4: "mostly",
        5: "always",
    }
    kw = kw_map.get(score, "sometimes")

    # Sentence structures that avoid always starting with "You"
    patterns = {
        0: [
            f"The skill '{topic_lc}' was {kw}. We'll introduce this movement step by step to build confidence.",
            f"'{topic_lc}' has {kw}. We'll start practicing it in the next sessions."
        ],
        1: [
            f"This aspect is {kw} {topic_lc}. Let’s bring awareness and begin developing this habit.",
            f"Currently, the movement '{topic_lc}' is {kw}. We'll establish it with clear guidance."
        ],
        2: [
            f"'{topic_lc}' appears {kw}. With focused repetition, it will become more natural.",
            f"The action '{topic_lc}' happens {kw}. Let's work on making it more consistent."
        ],
        3: [
            f"'{topic_lc}' is {kw}, showing solid understanding. We'll refine precision and timing next.",
            f"The way '{topic_lc}' is performed is {kw}. Great progress — now it's about small improvements."
        ],
        4: [
            f"'{topic_lc}' is {kw} and looks confident. Small adjustments can make it even smoother.",
            f"The execution of '{topic_lc}' is {kw}, which shows control. We'll polish the finer details."
        ],
        5: [
            f"'{topic_lc}' is {kw}, demonstrating strong mastery. Keep reinforcing this high standard.",
            f"An excellent performance — '{topic_lc}' is {kw} and stable across sessions."
        ]
    }

    return random.choice(patterns[score])

# ===== UI =====

FORM_MAP = {
    "Outdoor: Beginner": os.path.join(DATA_DIR, "outdoor_beginner.csv"),
    "Outdoor: Intermediate": os.path.join(DATA_DIR, "outdoor_intermediate.csv"),
    "Outdoor: Advanced": os.path.join(DATA_DIR, "outdoor_advanced.csv"),
    "Indoor": os.path.join(DATA_DIR, "indoor.csv"),
}

# Top buttons to switch form
b1, b2, b3, b4 = st.columns(4)
if b1.button("Outdoor: Beginner"): st.session_state["form_choice"] = "Outdoor: Beginner"
if b2.button("Outdoor: Intermediate"): st.session_state["form_choice"] = "Outdoor: Intermediate"
if b3.button("Outdoor: Advanced"): st.session_state["form_choice"] = "Outdoor: Advanced"
if b4.button("Indoor"): st.session_state["form_choice"] = "Indoor"

form_choice = st.session_state.get("form_choice", "Outdoor: Beginner")

title_col, logo_col = st.columns([5, 1])
with title_col:
    st.header(f"Surf Evaluation Form for {form_choice}")
with logo_col:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

# Load topics
topics_df = load_topics(FORM_MAP.get(form_choice, ""))
if topics_df.empty:
    st.error(f"CSV missing for {form_choice}. Expected columns: Category,Topic,Ranking")
    st.stop()

# ---------- Sidebar: Load / Create client ----------
with st.sidebar.expander("Load Client", True):
    names = list_clients()
    selected_name = st.selectbox("Select client", [""] + names, key="client_select")
    if selected_name:
        meta, df_loaded = load_client_file(client_filename(selected_name))

        # Put into session (wir upgraden beim Speichern sanft)
        st.session_state["name"]  = meta.get("ClientName", selected_name)
        st.session_state["email"] = meta.get("Email", "")
        st.session_state["phone"] = meta.get("Phone", "")
        st.session_state["loaded_df"] = df_loaded

st.sidebar.markdown("### Current Client")
name  = st.sidebar.text_input("Name",  st.session_state.get("name",  ""))
email = st.sidebar.text_input("Email", st.session_state.get("email", ""))
phone = st.sidebar.text_input("Phone", st.session_state.get("phone", ""))

# Persist video uploads in session
uploaded_videos = st.sidebar.file_uploader("Videos", type=["mp4", "mov"], accept_multiple_files=True)
if "uploaded_videos" not in st.session_state:
    st.session_state["uploaded_videos"] = []
if uploaded_videos:
    st.session_state["uploaded_videos"] = uploaded_videos
uploaded_videos = st.session_state["uploaded_videos"]

if st.sidebar.button("Save Client Info"):
    df_tmp = st.session_state.get("loaded_df", pd.DataFrame())
    save_client_file(name, email, phone, form_choice, df_tmp, videos=uploaded_videos)
    st.success("Client info saved ✅")

if not name:
    st.info("Please enter/select a client and click 'Save Client Info'.")
    st.stop()

# ---------- Ratings UI ----------
loaded_df = st.session_state.get("loaded_df", pd.DataFrame())
# ensure columns exist in loaded_df to avoid key errors later
for col in ["Category", "Topic", "Ranking", "Report", "Day1", "Day2", "Day3"]:
    if col not in loaded_df.columns:
        if col == "Report":
            loaded_df[col] = "No"
        elif col in ["Day1", "Day2", "Day3"]:
            loaded_df[col] = 0
        else:
            loaded_df[col] = ""

st.subheader(f"Ratings — {name}")

rows = []

def _safe_prev_int(df_row, col):
    """Return int 0 if NaN/empty/missing."""
    try:
        if col in df_row.columns and not df_row.empty:
            v = pd.to_numeric(df_row[col].iloc[0], errors="coerce")
            if pd.isna(v): 
                return 0
            return int(float(v))
    except:
        return 0
    return 0

def _safe_prev_str(df_row, col, default="No"):
    try:
        if col in df_row.columns and not df_row.empty:
            v = str(df_row[col].iloc[0])
            return v if v else default
    except:
        return default
    return default

for cat in topics_df["Category"].unique():
    with st.expander(f"📂 {cat}", expanded=False):
        sub = topics_df[topics_df["Category"] == cat]
        for i, (_, r) in enumerate(sub.iterrows()):
            top = str(r["Topic"])
            rk  = int(r["Ranking"])

            prev = loaded_df[(loaded_df["Category"] == cat) & (loaded_df["Topic"] == top)]

            d1 = _safe_prev_int(prev, "Day1")
            d2 = _safe_prev_int(prev, "Day2")
            d3 = _safe_prev_int(prev, "Day3")
            rep = _safe_prev_str(prev, "Report", default="No")

            c = st.columns([3, 1, 1, 1, 1])
            c[0].markdown(f"**{top}** ({rk})")

            # ✅ Report as one-click toggle (checkbox)
            chk = c[1].checkbox("Report", value=(rep == "Yes"), key=f"rep_{cat}_{top}_{i}")
            E   = "Yes" if chk else "No"

            D1 = c[2].number_input("Day1", min_value=0, max_value=5, value=int(d1), key=f"d1_{cat}_{top}_{i}")
            D2 = c[3].number_input("Day2", min_value=0, max_value=5, value=int(d2), key=f"d2_{cat}_{top}_{i}")
            D3 = c[4].number_input("Day3", min_value=0, max_value=5, value=int(d3), key=f"d3_{cat}_{top}_{i}")

            # ✅ Visual highlighting (green border for Report=Yes; red border if day <=3)
            report_style = "border:2px solid #22c55e; border-radius:6px; padding:2px 6px; display:inline-block; margin-top:4px;"
            report_style_off = "border:1px solid #999; border-radius:6px; padding:2px 6px; display:inline-block; margin-top:4px;"
            c[1].markdown(
                f"<div style='{report_style if E=='Yes' else report_style_off}'>Report: {E}</div>",
                unsafe_allow_html=True
            )
            def _badge(val):
                return "border:2px solid #ef4444; border-radius:6px; padding:2px 6px; display:inline-block; margin-top:4px;" if int(val) <= 3 and int(val) > 0 else "border:1px solid #999; border-radius:6px; padding:2px 6px; display:inline-block; margin-top:4px;"

            # --- Farbcode für Bewertungen 1–5 ---
            color_map = {
                1: "#e11d48",     # Red (kräftig)
                2: "#ff6f00",     # Darker orange 🍊
                3: "#111111",     # Black
                4: "#2563eb",     # Blue
                5: "#16a34a"      # Green
            }

            def color_style(val):
                col = color_map.get(int(val), "gray")
                border = "2px solid #ef4444" if int(val) <= 3 and int(val) > 0 else "1px solid #999"
                return f"border:{border}; border-radius:6px; padding:2px 6px; display:inline-block; margin-top:4px; color:{col}; font-weight:600;"

            c[2].markdown(f"<div style='{color_style(D1)}'>Day1: {int(D1)}</div>", unsafe_allow_html=True)
            c[3].markdown(f"<div style='{color_style(D2)}'>Day2: {int(D2)}</div>", unsafe_allow_html=True)
            c[4].markdown(f"<div style='{color_style(D3)}'>Day3: {int(D3)}</div>", unsafe_allow_html=True)

            rows.append({
                "Category": cat,
                "Topic": top,
                "Ranking": rk,
                "Report": E,
                "Day1": int(D1),
                "Day2": int(D2),
                "Day3": int(D3),
            })

ratings = pd.DataFrame(rows)

# strong typing
for col in ["Day1", "Day2", "Day3"]:
    ratings[col] = pd.to_numeric(ratings[col], errors="coerce").fillna(0).astype(int)
if "Report" not in ratings.columns:
    ratings["Report"] = "No"

# ---------- Save ALL days & reports ----------
if st.button("💾 Save ALL days & reports"):
    base = st.session_state.get("loaded_df", pd.DataFrame()).copy()
    if base is None or base.empty:
        base = pd.DataFrame(columns=ratings.columns)

    for _, r in ratings.iterrows():
        mask = (base["Category"] == r["Category"]) & (base["Topic"] == r["Topic"])
        if mask.any():
            for col in ratings.columns:
                base.loc[mask, col] = r[col]
        else:
            base = pd.concat([base, pd.DataFrame([r])], ignore_index=True)

    save_client_file(name, email, phone, form_choice, base, videos=st.session_state.get("uploaded_videos", []))
    st.session_state["loaded_df"] = base
    st.success("✅ All days & reports saved")
    st.balloons()

# ---------- Summary & Radar ----------
if not ratings.empty:
    summary = ratings.groupby("Category")[["Day1", "Day2", "Day3"]].mean().round(2)
else:
    summary = pd.DataFrame()

st.subheader("📊 Progress Radar Chart")
st.pyplot(draw_radar(summary, name))

st.subheader("📋 Category Summary")
st.dataframe(summary, use_container_width=True)

# Store for next part (coaching + PDF)
st.session_state["ratings_now"] = ratings
st.session_state["summary_now"] = summary

# ===== Coaching (Report = Yes; Day3 als „neuester Wert“) =====
st.subheader("🏄‍♂️ Coaching Focus (Report = Yes)")

coach_lines = []
if not ratings.empty:
    rep = ratings[ratings["Report"] == "Yes"].copy()
    if rep.empty:
        st.info("No items selected for the report yet. Toggle 'Report' to Yes on the topics you want to include.")
    else:
        for cat in rep["Category"].unique():
            st.markdown(f"**{cat}**")
            group = rep[rep["Category"] == cat]
            for _, row in group.iterrows():
               # ---- Smart scoring fallback ----
                d3 = int(pd.to_numeric(row.get("Day3", 0), errors="coerce")) if pd.notna(row.get("Day3", 0)) else 0
                d2 = int(pd.to_numeric(row.get("Day2", 0), errors="coerce")) if pd.notna(row.get("Day2", 0)) else 0
                d1 = int(pd.to_numeric(row.get("Day1", 0), errors="coerce")) if pd.notna(row.get("Day1", 0)) else 0

                # ✅ pick first non-zero (Day3 → Day2 → Day1)
                s = d3 if d3 > 0 else (d2 if d2 > 0 else d1)

                sentence = coach_sentence(row["Topic"], s)
                st.write(f"- {sentence}")
                coach_lines.append(f"{cat}: {sentence}")
else:
    st.info("No ratings available yet.")

# keep for PDF
coach_text_for_pdf = "\n".join(coach_lines) if coach_lines else "No report items selected."
st.session_state["coach_text_for_pdf"] = coach_text_for_pdf

# ===== PDF (radar left, coaching text right, summary table bottom) =====
def _fig_png(fig):
    b = BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    b.seek(0)
    return b

from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

def create_pdf(name, email, phone, form, summary_df, coach_text):
    """Final clean version — no missing last category, tighter spacing."""
    out = f"eval_{name}.pdf"
    c = canvas.Canvas(out, pagesize=landscape(A4))
    W, H = landscape(A4)
    m = 25 * mm
    right_margin = 10 * mm  # 1 cm Sicherheitsrand

    # ===== HEADER =====
    c.setFont("Helvetica-Bold", 24)
    c.drawString(m, H - m, f"Surf Evaluation — {form}")

     # ===== LOGOS (logo – powered by – logo2) =====
    logo1 = LOGO_PATH
    logo2 = os.path.join(ASSETS_DIR, "logo2.png")

    # Größen nochmals um ~20 % erhöht (insgesamt +40 % zum Original)
    logo1_w, logo1_h = 36 * mm, 14.5 * mm
    logo2_w, logo2_h = 26 * mm, 11.5 * mm
    spacing = 4 * mm

    # Gesamtbreite berechnen
    total_w = logo1_w + logo2_w + spacing + c.stringWidth("powered by", "Helvetica", 13)
    x_start = W - m - total_w
    y_logo = H - m - 5 * mm  # leicht tiefer für gute Ausrichtung zur Headline

    try:
        # Erstes Logo (links)
        if os.path.exists(logo1):
            c.drawImage(logo1, x_start, y_logo, width=logo1_w, height=logo1_h, mask="auto")

        # Text „powered by“
        c.setFont("Helvetica", 13)
        x_text = x_start + logo1_w + spacing / 2
        c.drawString(x_text, y_logo + 3.8 * mm, "powered by")

        # Zweites Logo (rechts)
        if os.path.exists(logo2):
            x_logo2 = x_text + c.stringWidth("powered by", "Helvetica", 13) + spacing
            c.drawImage(logo2, x_logo2, y_logo + 1.4 * mm, width=logo2_w, height=logo2_h, mask="auto")
    except Exception as e:
        print("⚠️ Logo placement error:", e)

    # ===== CLIENT INFO =====
    c.setFont("Helvetica", 12)
    y = H - m - 20 * mm
    for k, v in [("Client", name), ("Email", email), ("Phone", phone), ("Date", datetime.date.today())]:
        c.drawString(m, y, f"{k}: {v}")
        y -= 5 * mm

    # ===== LEFT COLUMN =====
    left_x = m
    y_top = y - 8 * mm
    radar_size = 95 * mm
    table_h = 70 * mm
    table_y_offset = 35 * mm

    # Radar Chart
    fig = draw_radar(summary_df, name if name else "Progress")
    try:
        c.drawImage(ImageReader(_fig_png(fig)), left_x, y_top - radar_size, width=radar_size, height=radar_size)
    except:
        pass

    # Summary Table
    if summary_df is not None and not summary_df.empty:
        data = [["Category", "Day1", "Day2", "Day3"]]
        for cat, row in summary_df.iterrows():
            data.append([
                cat,
                f"{row['Day1']:.1f}" if pd.notna(row['Day1']) else "-",
                f"{row['Day2']:.1f}" if pd.notna(row['Day2']) else "-",
                f"{row['Day3']:.1f}" if pd.notna(row['Day3']) else "-"
            ])
        tbl = Table(data, colWidths=[42 * mm, 16 * mm, 16 * mm, 16 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0077C8")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.black),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ]))
        tbl.wrapOn(c, 0, 0)
        tbl.drawOn(c, left_x, y_top - radar_size - table_h + table_y_offset)

    # ===== RIGHT COLUMN =====
    right_x = left_x + radar_size + 5 * mm
    y_right = H - m - 28 * mm
    max_text_width = W - right_x - right_margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(right_x, y_right, "Selected Coaching Focus")
    y_right -= 8 * mm

    # Gruppieren der Coaching-Zeilen
    grouped = {}
    for line in (coach_text or "").split("\n"):
        if ":" in line:
            cat, txt = line.split(":", 1)
            grouped.setdefault(cat.strip(), []).append(txt.strip())

        # Paragraph Style (tighter spacing)
    style = ParagraphStyle(
        name="Normal",
        fontName="Helvetica",
        fontSize=12,
        leading=14,  # angenehmer Zeilenabstand
        alignment=TA_LEFT,
    )

    for cat, sentences in grouped.items():
        if y_right < 25 * mm:
            # letzte Kategorie trotzdem zeigen (auch wenn wenig Platz)
            pass  

        # 🟦 Abstand vor neuer Kategorie
        y_right -= 6 * mm

        # Kategorie-Titel
        c.setFont("Helvetica-Bold", 14)
        c.drawString(right_x, y_right, cat)
        y_right -= 2 * mm  # kleiner Abstand direkt darunter

        # Textblock (Paragraph)
        text_block = "<br/>".join([f"• {s}" for s in sentences])
        p = Paragraph(text_block, style)
        w, h = p.wrap(max_text_width, 9999)

        # Falls Platz knapp — wenigstens Rest sichtbar anzeigen
        if y_right - h < 10 * mm:
            h = y_right - 15 * mm  # leicht stauchen statt weglassen

        p.drawOn(c, right_x, y_right - h)
        y_right -= h + 6  # kleinerer Abstand zum nächsten Block

    c.save()
    return out

# PDF export UI
col_pdf1, col_pdf2 = st.columns([1, 3])
with col_pdf1:
    if st.button("📄 Generate PDF"):
        pdf_file = create_pdf(
            name=name,
            email=email,
            phone=phone,
            form=form_choice,
            summary_df=st.session_state.get("summary_now", pd.DataFrame()),
            coach_text=st.session_state.get("coach_text_for_pdf", "No report items selected.")
        )
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name=pdf_file, mime="application/pdf")
with col_pdf2:
    st.caption("PDF includes: radar (left), coaching text (right), and category summary at the bottom.")