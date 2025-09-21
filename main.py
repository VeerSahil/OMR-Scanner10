import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import sqlite3

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f9fafb;
    }
    .stAlert {
        border-radius: 12px !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .score-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .score-title {
        font-size: 18px;
        font-weight: 600;
        color: #374151;
    }
    .score-value {
        font-size: 24px;
        font-weight: 700;
        color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Setup (SQLite) ---
DB_FILE = "omr_results.db"

def get_db_connection():
    return sqlite3.connect(DB_FILE)

def create_table_if_not_exists(conn, cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS omr_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        student_id TEXT NOT NULL,
        total_score INTEGER NOT NULL,
        python_score INTEGER,
        data_analysis_score INTEGER,
        mysql_score INTEGER,
        power_bi_score INTEGER,
        adv_stats_score INTEGER,
        image_path TEXT
    );
    """)
    conn.commit()

def insert_result(conn, cursor, student_id, total_score, subject_scores, image_path):
    insert_query = """
    INSERT INTO omr_results 
    (student_id, total_score, python_score, data_analysis_score, mysql_score, power_bi_score, adv_stats_score, image_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """
    data = (
        student_id,
        total_score,
        subject_scores[0],
        subject_scores[1],
        subject_scores[2],
        subject_scores[3],
        subject_scores[4],
        image_path
    )
    cursor.execute(insert_query, data)
    conn.commit()
    st.success("✅ Result saved successfully!")

# --- Answer Key Reader ---
def read_answer_key_from_file(file):
    try:
        df = pd.read_csv(file, header=None)
        start_row = 0
        for i, row in df.iterrows():
            if '1 - ' in str(row.iloc[0]):
                start_row = i
                break
        df = df.iloc[start_row:].copy()
        answer_pairs = []
        for col in df.columns:
            for item in df[col].dropna():
                match = re.search(r'^(\d+)\s*[-.]\s*([a-z])', str(item).lower())
                if match:
                    q_num = int(match.group(1))
                    ans_val = ord(match.group(2)) - ord('a')
                    answer_pairs.append((q_num, ans_val))
        answer_pairs.sort(key=lambda x: x[0])
        return [ans for _, ans in answer_pairs]
    except Exception as e:
        st.error(f"⚠️ Error reading answer key: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_omr_sheet(student_image):
    return student_image

def detect_and_score_omr(aligned_image, answer_key):
    gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    roi_coords = [
        (65, 200, 150, 560),  # Python
        (255, 200, 150, 560), # Data Analysis
        (445, 200, 150, 560), # MySQL
        (635, 200, 150, 560), # Power BI
        (825, 200, 150, 560), # Adv Stats
    ]

    subject_scores = [0, 0, 0, 0, 0]
    student_answers = []
    bubble_locations = []

    filled_ratio_threshold = 0.25

    for col_idx, (x_offset, y_offset, w, h) in enumerate(roi_coords):
        subject_roi = thresh[y_offset:y_offset+h, x_offset:x_offset+w]
        if subject_roi.size == 0:
            continue

        contours, _ = cv2.findContours(subject_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 6 or bh < 6:
                continue
            boxes.append((bx, by, bw, bh))

        if len(boxes) >= 8:
            # contour-based detection (same as before)
            median_w = int(np.median([b[2] for b in boxes]))
            median_h = int(np.median([b[3] for b in boxes]))
            centers = [(bx+bw//2, by+bh//2, bx, by, bw, bh) for bx,by,bw,bh in boxes]
            centers.sort(key=lambda t: t[1])
            rows = []
            if centers:
                cur_row = [centers[0]]
                cur_mean_y = centers[0][1]
                row_thresh = max( int(median_h * 0.8), 8 )
                for item in centers[1:]:
                    if abs(item[1] - cur_mean_y) <= row_thresh:
                        cur_row.append(item)
                        cur_mean_y = int(np.mean([r[1] for r in cur_row]))
                    else:
                        rows.append(cur_row)
                        cur_row = [item]
                        cur_mean_y = item[1]
                rows.append(cur_row)

            rows_sorted = [sorted(r, key=lambda t: t[0]) for r in rows]
            rows_sorted.sort(key=lambda r: np.mean([it[1] for it in r]) if r else 0)

            for row in rows_sorted:
                selected = []
                if len(row) >= 4:
                    idxs = np.linspace(0, len(row)-1, 4).astype(int)
                    selected = [row[i] for i in idxs]
                else:
                    selected = row[:]

                selected = sorted(selected, key=lambda t: t[0])
                while len(selected) < 4:
                    selected.append(None)

                q_options = []
                for opt in selected:
                    if opt is None:
                        q_options.append({'filled_ratio': 0.0, 'bbox': None, 'center': None})
                    else:
                        cx, cy, bx, by, bw, bh = opt
                        by1, by2 = max(0, by), min(subject_roi.shape[0], by+bh)
                        bx1, bx2 = max(0, bx), min(subject_roi.shape[1], bx+bw)
                        bubble_roi = subject_roi[by1:by2, bx1:bx2]
                        area = max(1, (bx2-bx1)*(by2-by1))
                        filled = cv2.countNonZero(bubble_roi)
                        ratio = filled / area
                        global_bbox = (x_offset + bx1, y_offset + by1, bx2 - bx1, by2 - by1)
                        global_center = (x_offset + cx, y_offset + cy)
                        q_options.append({'filled_ratio': ratio, 'bbox': global_bbox, 'center': global_center})

                ratios = [o['filled_ratio'] for o in q_options]
                max_ratio = max(ratios)
                marked = int(np.argmax(ratios)) if max_ratio > filled_ratio_threshold else None

                student_answers.append(marked)
                bubble_locations.append({'q_num': len(student_answers), 'options': q_options, 'subject': col_idx})

        else:
            # fallback grid slicing (fixed)
            num_questions = 20
            for q_idx in range(num_questions):
                y_start = int(q_idx * h / num_questions)
                y_end = int((q_idx + 1) * h / num_questions) if q_idx < num_questions - 1 else h

                q_options = []
                for opt in range(4):
                    x_start = int(opt * w / 4)
                    x_end = int((opt + 1) * w / 4) if opt < 3 else w

                    bubble_roi = subject_roi[y_start:y_end, x_start:x_end]
                    area = max(1, (x_end - x_start) * (y_end - y_start))
                    filled = cv2.countNonZero(bubble_roi)
                    ratio = filled / area

                    global_bbox = (x_offset + x_start, y_offset + y_start, x_end - x_start, y_end - y_start)
                    global_center = (x_offset + x_start + (x_end - x_start)//2, y_offset + y_start + (y_end - y_start)//2)

                    q_options.append({'filled_ratio': ratio, 'bbox': global_bbox, 'center': global_center})

                ratios = [o['filled_ratio'] for o in q_options]
                max_ratio = max(ratios)
                marked = int(np.argmax(ratios)) if max_ratio > filled_ratio_threshold else None

                # For last 3 questions (18,19,20) do not highlight
                if q_idx >= 17:
                    for o in q_options:
                        o['bbox'] = None
                        o['center'] = None
                    marked = None

                student_answers.append(marked)
                bubble_locations.append({'q_num': len(student_answers), 'options': q_options, 'subject': col_idx})

    # scoring
    for i in range(min(len(answer_key), len(student_answers))):
        student_mark = student_answers[i]
        if student_mark is not None and student_mark == answer_key[i]:
            subj_idx = bubble_locations[i]['subject']
            if 0 <= subj_idx < len(subject_scores):
                subject_scores[subj_idx] += 1

    total_score = sum(subject_scores)
    return total_score, subject_scores, student_answers, bubble_locations

# --- Visual Overlay ---
def generate_visual_overlay(image, bubble_locations, student_answers, answer_key):
    overlay = image.copy()
    for i, loc in enumerate(bubble_locations):
        correct = answer_key[i] if i < len(answer_key) else None
        marked = student_answers[i] if i < len(student_answers) else None

        if marked is not None:
            opt = loc['options'][marked]
            bbox = opt.get('bbox')
            if bbox:
                bx, by, bw, bh = bbox
                pad_x = max(1, int(bw * 0.12))
                pad_y = max(1, int(bh * 0.12))
                tl = (bx + pad_x, by + pad_y)
                br = (bx + bw - pad_x, by + bh - pad_y)
                color = (0, 255, 0) if (correct is not None and marked == correct) else (0, 0, 255)
                cv2.rectangle(overlay, tl, br, color, 2)
                if opt.get('center'):
                    cx, cy = opt['center']
                    cv2.circle(overlay, (cx, cy), max(2, int(min(bw, bh) * 0.08)), color, -1)

        if correct is not None and marked is not None and correct != marked:
            if 0 <= correct < len(loc['options']):
                corr_opt = loc['options'][correct]
                corr_bbox = corr_opt.get('bbox')
                if corr_bbox:
                    bx, by, bw, bh = corr_bbox
                    pad_x = max(1, int(bw * 0.12))
                    pad_y = max(1, int(bh * 0.12))
                    tl = (bx + pad_x, by + pad_y)
                    br = (bx + bw - pad_x, by + bh - pad_y)
                    cv2.rectangle(overlay, tl, br, (0, 255, 0), 2)
                    if corr_opt.get('center'):
                        cx, cy = corr_opt['center']
                        cv2.circle(overlay, (cx, cy), max(2, int(min(bw, bh) * 0.08)), (0, 255, 0), -1)
    return overlay

# --- Frontend UI ---
st.title("📝 Automated OMR Evaluation & Scoring System")
st.markdown("Upload an **Answer Key** and a **Student OMR Sheet** to get started.")

with st.container():
    st.subheader("📂 Upload Files")
    col1, col2 = st.columns(2)
    with col1:
        answer_key_file = st.file_uploader("Answer Key (CSV)", type=["csv"])
    with col2:
        omr_sheet_file = st.file_uploader("OMR Sheet (JPG/PNG)", type=["jpg", "jpeg", "png"])
    student_id = st.text_input("🆔 Enter Student ID")

if not answer_key_file or not omr_sheet_file or not student_id:
    st.info("⬆️ Please upload both files and enter Student ID to continue.")
else:
    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        create_table_if_not_exists(conn, cursor)

        answer_key = read_answer_key_from_file(answer_key_file)
        if answer_key:
            omr_bytes = np.asarray(bytearray(omr_sheet_file.read()), dtype=np.uint8)
            omr_img = cv2.imdecode(omr_bytes, 1)

            total, subs, stu_ans, bubbles = detect_and_score_omr(preprocess_omr_sheet(omr_img), answer_key)
            overlay_img = generate_visual_overlay(omr_img, bubbles, stu_ans, answer_key)

            insert_result(conn, cursor, student_id, total, subs, omr_sheet_file.name)

            st.success("🎉 Processing complete!")

            with st.expander("🔍 Visual Evaluation (click to expand)", expanded=True):
                st.image(overlay_img, caption="Green = Correct, Red = Incorrect", use_container_width=True)

            st.subheader("📊 Score Breakdown")
            cols = st.columns(5)
            subjects = ["Python", "Data Analysis", "MySQL", "Power BI", "Adv Stats"]
            for idx, col in enumerate(cols):
                with col:
                    st.markdown(f"""
                        <div class="score-card">
                            <div class="score-title">{subjects[idx]}</div>
                            <div class="score-value">{subs[idx]} / 20</div>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.metric("🏆 Total Score", f"{total} / {len(answer_key)}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
