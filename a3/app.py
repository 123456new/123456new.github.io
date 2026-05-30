"""
医疗手语关键点采集系统

两种运行方式:
  streamlit run app.py          启动 Web 界面（采集手语关键点）
  python app.py --download-model  下载模型文件

环境要求:
  Python 3.9+, mediapipe >= 0.10.0, streamlit >= 1.28.0, opencv-python >= 4.8.0
"""

import cv2
import sqlite3
import json
import tempfile
import os
import sys
import urllib.request
import ssl
import argparse
from pathlib import Path
from datetime import datetime

# ======================================
# 路径配置（不依赖 Streamlit）
# ======================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DB_NAME = str(SCRIPT_DIR / "sign_language_data.db")
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

MODEL_URLS = [
    "https://hf-mirror.com/camenduru/HandRefiner/resolve/main/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
]
MODEL_EXPECTED_SIZE = 5_000_000


# ======================================
# 数据库
# ======================================
def init_db():
    """创建数据库表（如不存在）"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS keypoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                meaning TEXT NOT NULL,
                video_name TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                hand_index INTEGER NOT NULL,
                hand_label TEXT,
                landmarks TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS video_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                category TEXT NOT NULL,
                meaning TEXT NOT NULL,
                total_frames INTEGER DEFAULT 0,
                saved_keypoints INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_name, category, meaning)
            )
        """)
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def get_db_connection():
    return sqlite3.connect(DB_NAME)


# ======================================
# 视频记录操作
# ======================================
def save_video_record(video_name, category, meaning, total_frames, saved_keypoints):
    """保存或更新视频记录"""
    try:
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO video_records (video_name, category, meaning, total_frames, saved_keypoints)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(video_name, category, meaning)
            DO UPDATE SET total_frames = excluded.total_frames,
                          saved_keypoints = excluded.saved_keypoints,
                          created_at = CURRENT_TIMESTAMP
        """, (video_name, category, meaning, total_frames, saved_keypoints))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def get_video_records():
    """获取所有视频记录，按分类分组"""
    try:
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT id, video_name, category, meaning, total_frames, saved_keypoints, created_at
            FROM video_records
            ORDER BY created_at DESC
        """).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def delete_video_record(record_id):
    """删除视频记录及其对应的关键点数据"""
    try:
        conn = get_db_connection()
        row = conn.execute("SELECT video_name, category, meaning FROM video_records WHERE id = ?",
                          (record_id,)).fetchone()
        if row:
            conn.execute("DELETE FROM keypoints WHERE video_name = ? AND category = ? AND meaning = ?",
                        (row[0], row[1], row[2]))
        conn.execute("DELETE FROM video_records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


# ======================================
# 模型文件下载
# ======================================
def download_model():
    """从多个备选地址尝试下载 hand_landmarker.task 模型文件"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > MODEL_EXPECTED_SIZE:
        return True, "模型已存在"

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    last_error = ""
    for url in MODEL_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
                with open(MODEL_PATH, "wb") as f:
                    f.write(resp.read())
            size = MODEL_PATH.stat().st_size
            if size > MODEL_EXPECTED_SIZE:
                return True, f"下载成功 ({size / 1024 / 1024:.1f} MB)"
            else:
                last_error = f"文件大小异常: {size} bytes"
        except Exception as e:
            last_error = str(e)
            continue
    return False, last_error


def _print_download_instructions():
    print()
    print("=" * 60)
    print("  手动下载方法:")
    print()
    print(f"  1. 打开链接: {MODEL_URLS[0]}")
    print(f"  2. 保存文件到: {MODEL_PATH}")
    print()
    print("  或用能访问外网的设备下载后传输过来。")
    print("=" * 60)


# ======================================
# MediaPipe HandLandmarker 加载 (Tasks API)
# ======================================
def load_hand_landmarker():
    """加载 MediaPipe HandLandmarker（Streamlit 模式下带缓存）"""
    if not MODEL_PATH.exists():
        ok, msg = download_model()
        if not ok:
            raise RuntimeError(
                f"模型文件缺失: {msg}\n\n"
                f"请手动下载后放置到: {MODEL_PATH}\n"
                f"下载地址: {MODEL_URLS[0]}"
            )

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


# ======================================
# 视频处理函数
# ======================================
def process_video(video_path, video_name, category, meaning):
    """
    使用 MediaPipe HandLandmarker 处理视频帧，
    提取手部关键点并存入 SQLite 数据库。
    """
    try:
        detector = load_hand_landmarker()
    except Exception as e:
        return 0, 0, f"MediaPipe 模型加载失败: {e}"

    from mediapipe import Image as MpImage, ImageFormat

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        cap.release()
        return 0, 0, f"数据库连接失败: {e}"

    batch_data = []
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    hand_label = "Unknown"
                    if result.handedness and hand_idx < len(result.handedness):
                        hand_label = result.handedness[hand_idx][0].category_name
                    landmarks_data = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
                    landmarks_json = json.dumps(landmarks_data, ensure_ascii=False)
                    batch_data.append((
                        category, meaning, video_name, frame_count,
                        hand_idx, hand_label, landmarks_json,
                    ))
                    saved_count += 1
            frame_count += 1

            if len(batch_data) >= 100:
                cursor.executemany(
                    "INSERT INTO keypoints (category, meaning, video_name, frame_index, hand_index, hand_label, landmarks) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", batch_data)
                conn.commit()
                batch_data.clear()

        if batch_data:
            cursor.executemany(
                "INSERT INTO keypoints (category, meaning, video_name, frame_index, hand_index, hand_label, landmarks) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", batch_data)
            conn.commit()
    except Exception as e:
        return frame_count, saved_count, f"处理过程出错: {e}"
    finally:
        conn.close()
        cap.release()
    return frame_count, saved_count, "success"


# ======================================
# CLI 入口
# ======================================
def cli_main():
    parser = argparse.ArgumentParser(
        description="医疗手语关键点采集系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  streamlit run app.py             启动 Web 界面
  python app.py --download-model   下载模型文件
        """,
    )
    parser.add_argument("--download-model", "-d", action="store_true", help="下载 hand_landmarker.task 模型文件")
    args = parser.parse_args()

    if args.download_model:
        print("=" * 50)
        print("  正在下载 MediaPipe HandLandmarker 模型...")
        print("=" * 50)
        ok, msg = download_model()
        print(f"\n{'' if ok else ''} {msg}")
        if not ok:
            _print_download_instructions()
        return 0 if ok else 1

    parser.print_help()
    return 0


# ======================================
# ———— 准 入 Gate ————
# ======================================
if len(sys.argv) > 1 and sys.argv[1] in ("--download-model", "-d"):
    sys.exit(cli_main())

import contextlib
import logging

logging.getLogger("streamlit").setLevel(logging.CRITICAL)

import streamlit as st

_IN_STREAMLIT = False
with contextlib.redirect_stderr(open(os.devnull, "w")):
    try:
        st.set_page_config(page_title="医疗手语关键点采集系统", layout="wide", page_icon="")
        _IN_STREAMLIT = True
    except Exception:
        pass

if not _IN_STREAMLIT:
    print("=" * 50)
    print("  医疗手语关键点采集系统")
    print("=" * 50)
    print()
    print("  用法:")
    print("    streamlit run app.py             启动 Web 界面")
    print("    python app.py --download-model   下载模型文件")
    print()
    sys.exit(0)

# ----- Python 3.14 + Windows 兼容补丁 -----
import ctypes as _ctypes
from mediapipe.tasks.python.core import mediapipe_c_bindings as _mpb

_mpb_dll_path = str(
    Path(_mpb.__file__).resolve().parent.parent.parent / "c" / "libmediapipe.dll"
)

def _mpb_patched_load(signatures=()):
    if _mpb._shared_lib is not None:
        lib = _mpb._shared_lib
    else:
        lib = _ctypes.CDLL(_mpb_dll_path)
        _mpb._shared_lib = lib
    for sig in signatures:
        func = getattr(lib, sig.func_name)
        func.argtypes = sig.argtypes
        func.restype = sig.restype
    _msvcrt = _ctypes.CDLL("msvcrt.dll")
    lib.free = _msvcrt.free
    lib.free.argtypes = [_ctypes.c_void_p]
    lib.free.restype = None
    return lib

_mpb.load_raw_library = _mpb_patched_load

_load_hand_landmarker_cached = st.cache_resource(
    show_spinner="正在加载 MediaPipe 手部关键点模型..."
)(lambda: load_hand_landmarker())

db_ok, db_err = init_db()
if not db_ok:
    st.error(f"数据库初始化失败: {db_err}")
    st.stop()

# ======================================
# 自定义 CSS
# ======================================
st.markdown("""
<style>
    .stApp {
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        background: transparent !important;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url('https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1920&q=80') no-repeat center center fixed;
        background-size: cover;
        filter: contrast(1.3);
        z-index: -2;
        pointer-events: none;
    }
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.50);
        z-index: -1;
        pointer-events: none;
    }
    .stApp > header, header { display: none !important; }
    #root > div:first-child { border: none !important; outline: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    .main > div { background: transparent !important; border: none !important; }
    .stApp > .main .block-container {
        background: transparent !important;
        padding: 0rem 3rem 2rem !important;
        margin-top: -1rem !important;
        position: relative;
        z-index: 1;
        max-width: 100%;
    }
    .stAlert, .stInfo, .stSuccess, .stError { background: white !important; }

    /* 顶部导航栏 */
    .nav-bar {
        background: rgba(26,28,30,0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 0.8rem 0.8rem;
        margin-bottom: 0.8rem;
        display: flex;
        gap: 0.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .nav-btn {
        flex: 1;
        text-align: center;
        padding: 0.8rem 0.3rem;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        border: none;
        background: transparent;
        color: #9aa2af;
        transition: all 0.2s;
    }
    .nav-btn:hover {
        background: rgba(255,255,255,0.06);
        color: #e0e4ea;
    }
    .nav-btn.active {
        background: #2d5f8a;
        color: white;
        font-weight: 600;
    }

    /* 上传区域 */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #5a6068 !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.04) !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #4a90d9 !important;
    }

    /* 主按钮 */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2d5f8a, #1e3a5f) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* 去除多余边框 */
    [data-testid="stForm"] { border: none !important; padding: 0 !important; }
    [data-testid="stForm"] > div { border: none !important; }
    [data-testid="column"] { border: none !important; }
    .st-emotion-cache-1mi2ry2 { gap: 0 !important; border: none !important; background: transparent !important; }
    [data-testid="stVerticalBlock"] { border: none !important; background: transparent !important; }
    [data-testid="stVerticalBlock"] > div { border: none !important; background: transparent !important; }
    .stSelectbox > div, .stTextInput > div { border: none !important; }
    .st-emotion-cache-1euin81 { border: none !important; }
    section[data-testid="stDecoration"] { display: none !important; }
    div:empty { display: none !important; }

    /* 气球 */
    .stBalloons { z-index: 9999 !important; }

    /* 首页垂直居中容器 */
    .home-center {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: calc(100vh - 140px);
    }
    /* 居中卡片（首页） */
    .center-card {
        max-width: 700px;
        width: 100%;
        margin: 0 auto;
        background: rgba(26,28,30,0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 2.5rem 2.5rem 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .center-card .logo {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .center-card h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .center-card .subtitle {
        color: #9aa2af;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .center-card .divider, hr.card-divider {
        border: none;
        height: 1px;
        background: #2d3035;
        margin: 1.5rem 0;
    }

    /* 深色页面（数据目录、设置等） */
    .dark-page {
        background: rgba(26,28,30,0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 0.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .dark-page .stSubheader {
        color: white !important;
    }
    .dark-page .stExpander {
        background: #242629 !important;
        border: 1px solid #353840 !important;
    }
    .dark-page .stExpander .streamlit-expanderHeader {
        color: #e0e4ea !important;
        background: #242629 !important;
    }
    .dark-page .stExpander .streamlit-expanderContent {
        color: #d0d6de !important;
    }
    .dark-page .stExpander .stMarkdown p,
    .dark-page .stExpander .stMarkdown {
        color: #c8cdd5 !important;
    }
    .dark-page .stExpander .stButton button {
        color: #e74c3c !important;
        border-color: #e74c3c !important;
        background: rgba(231,76,60,0.15) !important;
    }
    .dark-page .stInfo {
        background: rgba(255,255,255,0.06) !important;
        color: #c8cdd5 !important;
        border: 1px solid #353840 !important;
    }
    .dark-page .stSelectbox label,
    .dark-page .stTextInput label {
        color: #c8cdd5 !important;
    }
    .dark-page .stSelectbox div,
    .dark-page .stTextInput input {
        background: #242629 !important;
        color: #e0e4ea !important;
        border-color: #353840 !important;
    }

    /* 系统设置页卡片 */
    .settings-card {
        background: rgba(36,38,41,0.85);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .settings-card .label {
        color: #9aa2af;
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
    }
    .settings-card .value {
        color: #e0e4ea;
        font-size: 0.9rem;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

# ======================================
# 页面导航
# ======================================
PAGES = ["首页", "视频录入", "项目介绍", "数据目录", "系统设置", "数据库工具"]

if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

# 顶部导航栏
st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
nav_cols = st.columns(6, gap="small")
for i, col in enumerate(nav_cols):
    with col:
        active = st.session_state.page == PAGES[i]
        btn_style = "nav-btn active" if active else "nav-btn"
        if st.button(PAGES[i], key=f"nav_{i}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = PAGES[i]
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if "last_video" not in st.session_state:
    st.session_state.last_video = None
    st.session_state.last_video_name = None
    st.session_state.last_result = None

# ======================================
# 页面 1:  首页 (只显示标题)
# ======================================
if st.session_state.page == PAGES[0]:
    st.markdown("<div class='home-center'>", unsafe_allow_html=True)
    st.markdown("<div class='center-card'>", unsafe_allow_html=True)
    st.markdown("<h1>医疗手语关键点采集系统</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>医疗手语视频采集 · 手部关键点提取 (MediaPipe Tasks API) · SQLite 数据库存储</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#5a6068;font-size:0.95rem'>请通过上方导航栏选择功能</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
# 页面 2:  视频录入
# ======================================
elif st.session_state.page == PAGES[1]:
    st.markdown("<div class='dark-page'>", unsafe_allow_html=True)
    st.subheader("视频录入")

    with st.form("video_form", clear_on_submit=False):
        col_cat, col_meaning = st.columns([1, 2])
        with col_cat:
            category = st.selectbox(
                "手语类别",
                ["日常用语", "症状描述", "身体部位", "药品名称",
                 "医院科室", "检查项目", "紧急求助", "疼痛表达",
                 "疾病名称", "康复护理"],
            )
        with col_meaning:
            meaning = st.text_input("手语含义", placeholder="例如：头疼、发烧、挂号、肚子痛...")

        uploaded_file = st.file_uploader("选择手语视频文件", type=["mp4", "avi", "mov"])

        submitted = st.form_submit_button("开始识别并保存数据库", type="primary", use_container_width=True)

    if uploaded_file is not None and submitted:
        if meaning.strip() == "":
            st.warning("请先输入手语表达的意思")
        else:
            video_bytes = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                video_path = tmp.name

            with st.spinner("正在分析视频，请稍候..."):
                total_frames, total_saved, status = process_video(
                    video_path, uploaded_file.name, category, meaning
                )

            if status != "success":
                st.error(f" {status}")
                try: os.unlink(video_path)
                except: pass
            else:
                save_video_record(uploaded_file.name, category, meaning, total_frames, total_saved)
                st.session_state.last_video = video_bytes
                st.session_state.last_video_path = video_path
                st.session_state.last_video_name = uploaded_file.name
                st.session_state.last_result = {
                    "category": category, "meaning": meaning,
                    "total_frames": total_frames, "total_saved": total_saved,
                }
                st.rerun()

    if st.session_state.last_video is not None:
        st.markdown("<hr class='card-divider'>", unsafe_allow_html=True)
        st.video(st.session_state.last_video_path)
        r = st.session_state.last_result
        st.success("视频处理完成！")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="视频文件名", value=st.session_state.last_video_name)
        with col2:
            st.metric(label="总帧数", value=r["total_frames"])
        with col3:
            st.metric(label="关键点", value=r["total_saved"])
        st.info(f"**分类：** {r['category']}  |  **含义：** {r['meaning']}")
        st.markdown("<div style='text-align:center;color:#9aa2af;font-size:0.8rem'>数据已保存到数据库</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
# 页面 3:  数据目录
# ======================================
elif st.session_state.page == PAGES[2]:
    st.markdown("<div class='dark-page'>", unsafe_allow_html=True)
    st.subheader("项目介绍")
    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>关于本系统</div>", unsafe_allow_html=True)
    st.markdown("<div class='value'>本系统利用 MediaPipe 手部关键点检测技术，采集医疗手语视频中的手势数据，构建医疗手语数据集。系统采用深色界面配合自然风景背景，象征医疗手语跨越沟通障碍、连接医患心灵的桥梁作用。山川湖泊的背景寓意医疗手语如自然般包容、流畅，帮助听障患者在医疗场景中无障碍沟通。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>技术栈</div>", unsafe_allow_html=True)
    st.markdown("<div class='value'>Streamlit · MediaPipe Tasks API · OpenCV · SQLite · Python 3.14</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>功能概览</div>", unsafe_allow_html=True)
    st.markdown("<div class='value'>视频录入 → 手部关键点提取 → SQLite 数据库存储 → 数据管理与删除</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>背景图片</div>", unsafe_allow_html=True)
    st.markdown("<div class='value'>页面背景采用自然山脉风景摄影，寓意手语沟通如自然般无碍。深色叠加层确保界面清晰可读，毛玻璃卡片设计让背景与内容和谐统一。</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
elif st.session_state.page == PAGES[3]:
    st.markdown("<div class='dark-page'>", unsafe_allow_html=True)
    st.subheader("已保存数据目录")

    records = get_video_records()

    if not records:
        st.info("暂无已保存的视频数据。请先上传并处理视频。")
    else:
        # 视频 + 目录左右分栏
        col_video, col_dir = st.columns([1, 1])
        with col_video:
            st.markdown("<div style='color:#9aa2af;font-size:0.85rem;margin-bottom:0.5rem'> 最近处理的视频</div>", unsafe_allow_html=True)
            if st.session_state.last_video is not None:
                st.video(st.session_state.last_video_path)
                if st.session_state.last_result:
                    r = st.session_state.last_result
                    st.markdown(f"<div style='color:#c8cdd5;font-size:0.85rem'> {r['meaning']}  ·   {r['total_saved']} 关键点</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#5a6068;padding:2rem 0;text-align:center'>暂无视频预览</div>", unsafe_allow_html=True)

        with col_dir:
            from collections import defaultdict
            grouped = defaultdict(list)
            for rec in records:
                grouped[rec[2]].append(rec)

            for cat_name in sorted(grouped.keys()):
                items = grouped[cat_name]
                with st.expander(f" {cat_name}（{len(items)} 条记录）", expanded=False):
                    for rec in items:
                        rec_id, video_name, _, meaning_text, total_frames, saved_keys, created_at = rec
                        try:
                            dt = datetime.fromisoformat(created_at)
                            time_str = dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            time_str = str(created_at)[:16] if created_at else ""

                        col_a, col_b, col_c, col_d = st.columns([3, 1.5, 1.5, 1])
                        with col_a:
                            st.markdown(f"** {video_name}**")
                        with col_b:
                            st.markdown(f" {meaning_text}")
                        with col_c:
                            st.markdown(f" {total_frames} 帧")
                        with col_d:
                            if st.button("", key=f"del_{rec_id}", help="删除此记录"):
                                ok, msg = delete_video_record(rec_id)
                                if ok:
                                    if st.session_state.last_video_name == video_name:
                                        st.session_state.last_video = None
                                        st.session_state.last_result = None
                                    st.success(f"已删除: {video_name}")
                                    st.rerun()
                                else:
                                    st.error(f"删除失败: {msg}")

        st.divider()
        total_videos = len(records)
        total_kps = sum(r[5] for r in records)
        st.info(f" 总计: {total_videos} 个视频，{total_kps} 条关键点记录")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
# 页面 3:  系统设置
# ======================================
elif st.session_state.page == PAGES[4]:
    st.markdown("<div class='dark-page'>", unsafe_allow_html=True)
    st.subheader("系统设置")

    # 系统状态
    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'> 数据库状态</div>", unsafe_allow_html=True)
    if db_ok:
        try:
            conn = get_db_connection()
            count = conn.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
            conn.close()
            st.markdown(f"<div class='value'> 已连接  ·  {count} 条关键点记录</div>", unsafe_allow_html=True)
        except Exception:
            st.markdown("<div class='value'> 已连接</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='value'> 连接失败</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'> MediaPipe 模型</div>", unsafe_allow_html=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > MODEL_EXPECTED_SIZE:
        st.markdown("<div class='value'> 已加载</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='value'>⏳ 未下载</div>", unsafe_allow_html=True)
        if st.button(" 下载模型", use_container_width=True):
            with st.spinner("正在下载 (约 15MB)..."):
                ok, msg = download_model()
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(f"下载失败: {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(" 手动上传模型文件", expanded=False):
        st.markdown(
            "如果自动下载失败：\n\n"
            "1️⃣ [点此链接](https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)"
            "下载 `hand_landmarker.task`\n\n"
            "2️⃣ 通过下面按钮上传"
        )
        uploaded_model = st.file_uploader("选择 hand_landmarker.task 文件", type=["task"], key="model_uploader_settings")
        if uploaded_model is not None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.read())
            st.success(f" 上传成功")
            st.cache_resource.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
# 页面 4:  数据库工具
# ======================================
elif st.session_state.page == PAGES[5]:
    st.markdown("<div class='dark-page'>", unsafe_allow_html=True)
    st.subheader("数据库工具")

    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'> 数据库路径</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{DB_NAME}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<div class='label'> 模型路径</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{MODEL_PATH}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(" 打开数据库所在文件夹", use_container_width=True):
        try:
            import subprocess
            subprocess.Popen(['explorer', '/select,', DB_NAME])
        except Exception as e:
            st.error(f"无法打开文件夹: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
