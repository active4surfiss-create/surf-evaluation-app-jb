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
        df = pd.read_csv(pd.io.common.StringIO("".join(data)))
    else:
        df = pd.DataFrame(columns=["Category","Topic","Ranking","Report","Day1","Day2","Day3"])

    if "Report" not in df.columns: df["Report"] = "No"
    for d in ["Day1","Day2","Day3"]:
        if d not in df.columns: df[d] = 0
            
    return meta,df

def list_clients():
    return sorted([x.replace(".csv","") for x in os.listdir(CLIENT_DIR) if x.endswith(".csv")])

# ========== RADAR CHART ==========
def draw_radar(df, title):
    # handle empty or invalid df
    if df is None or df.empty or not all(c in df.columns for c in ["Day1","Day2","Day3"]):
        fig, ax = plt.subplots()
        ax.text(.5,.5,"No Data",ha="center")
        return fig
    
    labels = df.index.tolist()
    values1 = df["Day1"].replace(0, np.nan).tolist()
    values2 = df["Day2"].replace(0, np.nan).tolist()
    values3 = df["Day3"].replace(0, np.nan).tolist()

    # Close radar loop
    values1 += values1[:1]
    values2 += values2[:1]
    values3 += values3[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))

    # Colors for each day
    color_map = {"Day1":"red", "Day2":"green", "Day3":"blue"}

    ax.plot(angles, values1, linewidth=2, label="Day1", color=color_map["Day1"])
    ax.fill(angles, values1, alpha=0.2, color=color_map["Day1"])

    ax.plot(angles, values2, linewidth=2, label="Day2", color=color_map["Day2"])
    ax.fill(angles, values2, alpha=0.2, color=color_map["Day2"])

    ax.plot(angles, values3, linewidth=2, label="Day3", color=color_map["Day3"])
    ax.fill(angles, values3, alpha=0.2, color=color_map["Day3"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 5)  # allow 0 now!
    ax.set_yticks(range(0,6))
    ax.set_yticklabels(["0","1","2","3","4","5"])

    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    return fig

# ========== TEXT MAPPING ==========
score_word = {1:"never",2:"sometimes",3:"usually",4:"mostly",5:"always"}
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

        # Detect "old format" (missing Report or Day cols or empty strings)
        needs_upgrade = False
        for col in ["Report", "Day1", "Day2", "Day3"]:
            if col not in df_loaded.columns:
                needs_upgrade = True
        # empty strings → NaN → will be defaulted to 0/"No" after upgrade
        if not needs_upgrade:
            empties = False
            for col in ["Day1", "Day2", "Day3"]:
                if df_loaded[col].isna().any():
                    empties = True
                if df_loaded[col].astype(str).eq("").any():
                    empties = True
            if "Report" in df_loaded.columns and df_loaded["Report"].astype(str).eq("").any():
                empties = True
            needs_upgrade = empties

        if needs_upgrade:
            st.warning("This client file uses an older format (missing or empty fields). Upgrade now?")
            u_col1, u_col2 = st.columns(2)
            if u_col1.button("✅ Upgrade now", key=f"upgrade_{selected_name}"):
                # Fill defaults: Report="No", Day1-3 = 0
                if "Report" not in df_loaded.columns:
                    df_loaded["Report"] = "No"
                for dcol in ["Day1", "Day2", "Day3"]:
                    if dcol not in df_loaded.columns:
                        df_loaded[dcol] = 0
                # coerce and fill empties
                for dcol in ["Day1", "Day2", "Day3"]:
                    df_loaded[dcol] = pd.to_numeric(df_loaded[dcol], errors="coerce").fillna(0).astype(int)
                df_loaded["Report"] = df_loaded["Report"].replace("", "No")

                # Save back immediately
                save_client_file(
                    meta.get("ClientName", selected_name),
                    meta.get("Email", ""),
                    meta.get("Phone", ""),
                    meta.get("Form", form_choice),
                    df_loaded
                )
                st.success("Client file upgraded.")
            else:
                st.info("Continuing without upgrade…")

        # Put into session
        st.session_state["name"] = meta.get("ClientName", selected_name)
        st.session_state["email"] = meta.get("Email", "")
        st.session_state["phone"] = meta.get("Phone", "")
        st.session_state["loaded_df"] = df_loaded

st.sidebar.markdown("### Current Client")
name = st.sidebar.text_input("Name", st.session_state.get("name", ""))
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
            E  = c[1].selectbox("Report", ["No", "Yes"], index=["No", "Yes"].index(rep), key=f"rep_{cat}_{top}_{i}")
            D1 = c[2].number_input("D1", min_value=0, max_value=5, value=d1, key=f"d1_{cat}_{top}_{i}")
            D2 = c[3].number_input("D2", min_value=0, max_value=5, value=d2, key=f"d2_{cat}_{top}_{i}")
            D3 = c[4].number_input("D3", min_value=0, max_value=5, value=d3, key=f"d3_{cat}_{top}_{i}")

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
    base = loaded_df.copy() if isinstance(loaded_df, pd.DataFrame) else pd.DataFrame(columns=ratings.columns)
    if base.empty:
        base = pd.DataFrame(columns=ratings.columns)

    for _, r in ratings.iterrows():
        mask = (base["Category"] == r["Category"]) & (base["Topic"] == r["Topic"])
        if mask.any():
            for col in ratings.columns:
                base.loc[mask, col] = r[col]
        else:
            base = pd.concat([base, pd.DataFrame([r])], ignore_index=True)

    save_client_file(name, email, phone, form_choice, base, videos=uploaded_videos)
    st.session_state["loaded_df"] = base
    st.success("✅ All days & reports saved")

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
# ===== Coaching text (uses Day3 as "newest value") =====

# Friendly/professional phrasing
_word_map = {0: "have not yet started to",
             1: "never",
             2: "sometimes",
             3: "usually",
             4: "mostly",
             5: "always"}

def _coach_sentence(topic: str, score: int) -> str:
    t = topic.strip()
    t_lc = t[0].lower() + t[1:] if t else t
    score = int(pd.to_numeric(score, errors="coerce")) if pd.notna(score) else 0
    score = max(0, min(5, score))
    w = _word_map.get(score, "sometimes")

    # professional & friendly tone (Option 2)
    if score == 0:
        return f"You have not yet started to {t_lc}. We'll introduce this step together and build confidence."
    if score == 1:
        return f"You currently never {t_lc}. We'll establish this habit with clear, simple reps."
    if score == 2:
        return f"You sometimes {t_lc}. Let's make this more consistent with focused practice."
    if score == 3:
        return f"You usually {t_lc}. Great work — we'll refine timing and consistency."
    if score == 4:
        return f"You mostly {t_lc}. Small refinements will make this feel even smoother."
    # score == 5
    return f"You always {t_lc}. Keep reinforcing this strong habit."

ratings = st.session_state.get("ratings_now", pd.DataFrame())
summary = st.session_state.get("summary_now", pd.DataFrame())

st.subheader("🏄‍♂️ Coaching Focus (Report = Yes)")
coach_lines = []
if not ratings.empty:
    # only items marked for report
    rep = ratings[ratings["Report"] == "Yes"].copy()

    if rep.empty:
        st.info("No items selected for the report yet. Switch 'Report' to 'Yes' on the topics you want to include.")
    else:
        # show grouped by Category
        for cat in rep["Category"].unique():
            st.markdown(f"**{cat}**")
            group = rep[rep["Category"] == cat]

            # newest value = Day3 (per your choice)
            # if Day3 == 0 we still generate a helpful intro line
            for _, row in group.iterrows():
                s = int(pd.to_numeric(row["Day3"], errors="coerce")) if pd.notna(row["Day3"]) else 0
                sentence = _coach_sentence(row["Topic"], s)
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

def create_pdf(name, email, phone, form, summary_df, coach_text):
    out = f"eval_{name}.pdf"
    c = canvas.Canvas(out, pagesize=landscape(A4))
    W, H = landscape(A4)
    m = 20 * mm

    # Header
    c.setFont("Helvetica-Bold", 22)
    c.drawString(m, H - m, f"Surf Evaluation — {form}")
    if os.path.exists(LOGO_PATH):
        try:
            c.drawImage(LOGO_PATH, W - m - 40 * mm, H - m - 15 * mm, width=40 * mm, height=12 * mm, mask="auto")
        except:
            pass

    # Client info
    c.setFont("Helvetica", 12)
    y = H - m - 10 * mm
    for k, v in [("Client", name), ("Email", email), ("Phone", phone), ("Date", datetime.date.today())]:
        c.drawString(m, y, f"{k}: {v}")
        y -= 5 * mm

    # Radar (left)
    fig = draw_radar(summary_df, name if name else "Progress")
    RAD = 120 * mm
    try:
        c.drawImage(ImageReader(_fig_png(fig)), m, y - RAD, width=RAD, height=RAD)
    except:
        pass

    # Coaching text (right)
    cx = m + RAD + 10 * mm
    cy = y
    c.setFont("Helvetica-Bold", 14)
    c.drawString(cx, cy, "Selected Coaching Focus")
    cy -= 8 * mm
    c.setFont("Helvetica", 10)

    # wrap text manually
    def _draw_wrapped(text, x, y, width_mm, line_h_mm):
        max_chars = 95  # simple wrap; ReportLab Paragraph could be used later
        line_h = line_h_mm * mm
        for raw in text.split("\n"):
            line = raw.strip()
            while len(line) > max_chars:
                c.drawString(x, y, line[:max_chars])
                line = line[max_chars:]
                y -= line_h
                if y < 20 * mm:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = H - m - 20 * mm
            if line:
                c.drawString(x, y, line)
                y -= line_h
                if y < 20 * mm:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = H - m - 20 * mm
        return y

    cy = _draw_wrapped(coach_text, cx, cy, width_mm=80, line_h_mm=5)

    # Summary table at bottom
    if summary_df is not None and not summary_df.empty:
        data = [["Category", "Day1", "Day2", "Day3"]]
        for cat, row in summary_df.iterrows():
            d1 = f"{row['Day1']:.2f}" if pd.notna(row["Day1"]) else "0.00"
            d2 = f"{row['Day2']:.2f}" if pd.notna(row["Day2"]) else "0.00"
            d3 = f"{row['Day3']:.2f}" if pd.notna(row["Day3"]) else "0.00"
            data.append([cat, d1, d2, d3])

        tbl = Table(data, colWidths=[90 * mm, 30 * mm, 30 * mm, 30 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0077C8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        tbl.wrapOn(c, 0, 0)
        tbl.drawOn(c, m, 10 * mm)

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