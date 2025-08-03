import requests
import zipfile
import streamlit as st
from comparative_model import *
# ---------- 页面配置 ----------
st.set_page_config(
    page_title="Multi-modal Few-shot Anthesis/Flowering time Prediction",
    layout="centered",
)

# ---------- 页面标题 ----------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown(
        "<h1 style='text-align: center;'>Multi-modal Few-shot Wheat Flowering Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h4 style='text-align: center; color: grey;'>Empowering individual wheat phenotyping using weather-aware image analysis</h4>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------- 引用 ----------
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: #666;'>
    Citation: Xie, Y., Roy, S., Schilling, R., & Liu, H. (2025). 
    <i>Multi-Modal Few-Shot Learning for Anthesis Prediction of Individual Wheat Plants</i>. 
    <b>Plant Phenomics</b>, 100091.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- 模型说明与映射 ----------

import os
import torch
import streamlit as st
from model_structure import FeatureExtractNetwork, ComparedNetwork, ComparedNetwork_Transformer

# ---------- 初始化缓存文件夹 ----------
import os
import shutil
import uuid

# 全局临时缓存根目录
base_temp_folder = "__temp__folder"
os.makedirs(base_temp_folder, exist_ok=True)

# 分配每个用户唯一 session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 当前用户专属临时目录
user_temp_folder = os.path.join(base_temp_folder, st.session_state.session_id)

# ✅ 只在用户第一次进入页面时清空一次
if "user_temp_initialized" not in st.session_state:
    if os.path.exists(user_temp_folder):
        shutil.rmtree(user_temp_folder)
    os.makedirs(user_temp_folder, exist_ok=True)
    st.session_state.user_temp_initialized = True
else:
    os.makedirs(user_temp_folder, exist_ok=True)  # 避免解压时报错



# ---------- 模型说明与映射 ----------
model_options = {
    "CONVNEXT + FC": (
        "convnext",
        "Feature extractor: **CONVNEXT** (image + weather) → Comparison module: **Fully Connected (FC)**"
    ),
    "CONVNEXT + TF": (
        "convnext_tf",
        "Feature extractor: **CONVNEXT** (image + weather) → Comparison module: **Transformer (TF)**"
    ),
    "SWIN V2 + FC": (
        "swin_b",
        "Feature extractor: **SWIN V2** (image + weather) → Comparison module: **Fully Connected (FC)**"
    ),
    "SWIN V2 + TF": (
        "swin_b_tf",
        "Feature extractor: **SWIN V2** (image + weather) → Comparison module: **Transformer (TF)**"
    )
}

# ---------- 本地对比模型路径（仅保留 compare model） ----------
comparative_model_paths = {
    "CONVNEXT + FC": "model/Full convnext/compare model.pth",
    "CONVNEXT + TF": "model/Full convnext TF/compare model.pth",
    "SWIN V2 + FC": "model/Full swin/compare model.pth",
    "SWIN V2 + TF": "model/Full swin TF/compare model.pth"
}

# ---------- Step 1：模型选择 ----------
st.header("1️⃣ Select Model")

model_name_display = st.selectbox(
    "Select the model architecture you want to use:",
    options=list(model_options.keys())
)

# 展示说明
_, model_description = model_options[model_name_display]
st.info(model_description)

# ---------- 模型加载按钮 ----------
if st.button("🚀 Load Selected Model"):
    selected_model_code, _ = model_options[model_name_display]
    comparative_model_path = comparative_model_paths[model_name_display]

    # Hugging Face 模型链接
    hf_base_url = "https://huggingface.co/Eting0308/Model_Multi-Modal_Few-Shot_Learning_for_Anthesis_Prediction_of_Individual_Wheat_Plants/resolve/main"
    hf_model_files = {
        "convnext": "convnext.pth",
        "convnext_tf": "convnext_tf.pth",
        "swin_b": "swin_b.pth",
        "swin_b_tf": "swin_b_tf.pth"
    }

    model_url = f"{hf_base_url}/{hf_model_files[selected_model_code]}"
    local_model_path = os.path.join("__temp__folder", hf_model_files[selected_model_code])
    os.makedirs("__temp__folder", exist_ok=True)

    # 下载模型（若本地无缓存）
    if not os.path.exists(local_model_path):
        with st.spinner("⏳ Downloading feature model from Hugging Face (approx. 5 minutes)..."):
            response = requests.get(model_url)
            response.raise_for_status()
            with open(local_model_path, "wb") as f:
                f.write(response.content)

    # 加载 feature 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "convnext" if "convnext" in selected_model_code else "swin_b"
    model_ft = FeatureExtractNetwork(model_name=model_name)
    model_ft.load_state_dict(torch.load(local_model_path, map_location=device))
    model_ft.to(device)
    model_ft.eval()

    # 加载本地对比模型
    if "tf" in comparative_model_path.lower():
        sim_model = ComparedNetwork_Transformer()
    else:
        sim_model = ComparedNetwork()

    sim_model.load_state_dict(torch.load(comparative_model_path, map_location=device))
    sim_model.to(device)
    sim_model.eval()

    # 缓存模型
    st.session_state.model_ft = model_ft
    st.session_state.sim_model = sim_model
    st.session_state.model_display_name = model_name_display
    st.session_state.selected_model_code = selected_model_code

    st.success(f"✅ Feature model loaded from Hugging Face: **{hf_model_files[selected_model_code]}**")
    st.success("✅ Comparative model loaded from local path.")

st.markdown("---")

# ---------- 当前模型提醒 ----------
if "model_display_name" in st.session_state:
    st.markdown(f"📌 **Current model in use**: `{st.session_state.model_display_name}`")

# ---------- Step 2：客户上传 ----------
st.header("2️⃣ Upload Image and Weather Data")
st.image("Figure 1.png", width=700)

# 日期输入
required_date = st.text_input("Enter the imaging date for prediction (format: YYYY-MM-DD)", value="2023-07-10")

# 图像上传（单图、多图或 ZIP）
uploaded_images = st.file_uploader("Upload image(s) or a ZIP file", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

# weather 上传（单个 Excel 文件）
uploaded_weather = st.file_uploader("Upload corresponding weather Excel (.xlsx)", type=["xlsx"])

# 创建目标路径
image_folder = os.path.join(user_temp_folder, 'custom_image')
weather_folder = os.path.join(user_temp_folder, 'custom_weather')

# 如果存在旧文件夹，则先删除
if os.path.exists(image_folder):
    shutil.rmtree(image_folder)
if os.path.exists(weather_folder):
    shutil.rmtree(weather_folder)

# 重新创建干净的文件夹
os.makedirs(image_folder, exist_ok=True)
os.makedirs(weather_folder, exist_ok=True)

# 上传处理逻辑
if uploaded_images and uploaded_weather and required_date:
    # 1. 处理图像上传（可能是 zip 或多张图片）
    for img_file in uploaded_images:
        if img_file.name.endswith(".zip"):
            # 保存 zip 临时路径
            zip_path = os.path.join('__temp__folder', img_file.name)
            with open(zip_path, "wb") as f:
                f.write(img_file.read())
            # 解压到 custom_image
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(image_folder)
            os.remove(zip_path)  # 清理 zip
        else:
            # 直接保存图片文件
            img_save_path = os.path.join(image_folder, img_file.name)
            with open(img_save_path, "wb") as f:
                f.write(img_file.read())

    # 2. 处理 weather 上传并重命名
    weather_save_path = os.path.join(weather_folder, f"{required_date}.xlsx")
    with open(weather_save_path, "wb") as f:
        f.write(uploaded_weather.read())

    st.success(f"✅ All files successfully uploaded and saved.`.\nWeather file renamed to `{required_date}.xlsx`.")

else:
    st.warning("Please upload both image(s) and a weather Excel file, and enter a valid date.")


st.markdown("---")
st.header("3️⃣ Choose Anchor Strategy")

anchor_option = st.radio(
    "How would you like to use anchor vectors?",
    options=[
        "Use Official Anchors",
        "Upload Your Own Anchors",
        "Train Your Own Anchor"
    ]
)

# ---------- Option 1: 使用官方 anchor ----------
if anchor_option == "Use Official Anchors":
    st.subheader("🔍 Select Anchor Climate Type")
    st.info("You will use predefined anchor vectors stored in the selected folder for prediction.")

    anchor_climate_options = {
        "Cold (Avg temp < 12°C, low light)": "cold",
        "Cool (Avg temp 12–14°C, moderate light)": "cool",
        "Warm (Avg temp > 14°C, high light)": "warm",
    }

    anchor_choice_display = st.selectbox(
        "Choose the anchor set based on the environmental conditions during the month you collected your data:",
        options=list(anchor_climate_options.keys())
    )

    anchor_climate_code = anchor_climate_options[anchor_choice_display]

    # 展示支持的预测天数范围
    if anchor_climate_code == "warm":
        supported_days = [8, 10, 12, 14]
    else:
        supported_days = [8, 10, 12, 14, 16, 18]

    st.markdown(f"📅 **This anchor set supports prediction windows of:** `{supported_days}` days")

    if "selected_model_code" not in st.session_state:
        st.warning("⚠️ Please load your model first.")
        st.stop()

    # ✅ 生成 anchor 路径
    anchor_dir = os.path.join("officials_anchor", f"{anchor_climate_code}_anchor", st.session_state.selected_model_code)
    st.session_state.anchor_dir = anchor_dir

    # 上传数据路径
    weather_excel = os.path.join(user_temp_folder, 'custom_weather', f"{required_date}.xlsx")

    # 🔍 统计图像数量并估算时间（以每张图 21.34 秒为参考）
    if os.path.exists(image_folder):
        num_images = len([
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        seconds_per_image = 5
        total_seconds = num_images * seconds_per_image
        total_minutes = total_seconds / 60

        st.markdown(
            f"🕒 **Estimated time**: {num_images} images × {seconds_per_image:.1f} sec = "
            f"~{total_minutes:.1f} minutes total (on CPU)"
        )

    # 🔘 添加运行按钮
    if st.button("▶️ Run Prediction"):
        predict_and_plot_anchor_votes(
            image_folder=image_folder,
            weather_excel=weather_excel,
            model_ft=st.session_state.model_ft,
            sim_model=st.session_state.sim_model,
            anchor_dir=st.session_state.anchor_dir,
        )


# ---------- Option 2: 上传自定义 anchor ----------
elif anchor_option == "Upload Your Own Anchors":
    st.subheader("📥 Upload Your Own Anchors")
    st.info("Upload one `.pth` anchor file and specify the day it represents. If a file for that day already exists, the new anchor will be averaged with the existing one. **Please ensure the uploaded anchor was generated using the same model architecture you selected above.**")

    import torch

    # 获取模型代号
    if "selected_model_code" not in st.session_state:
        st.warning("⚠️ Please load your model first.")
        st.stop()

    model_code = st.session_state.selected_model_code
    custom_anchor_dir = os.path.join(user_temp_folder, 'custom_anchor')
    os.makedirs(custom_anchor_dir, exist_ok=True)

    # 输入上传项：anchor文件 + 天数
    uploaded_anchor = st.file_uploader(
        "📤 Upload a single anchor `.pth` file:",
        type=["pth"],
        accept_multiple_files=False,
        key="custom_anchor_file"
    )

    anchor_day = st.number_input(
        "📅 Enter the prediction day this anchor represents (e.g., 8, 10, 12...):",
        step=1,
        min_value=1,
        key="custom_anchor_day"
    )

    # 上传按钮
    if uploaded_anchor and anchor_day and st.button("📁 Upload Anchor"):
        anchor_filename = f"{anchor_day}_{model_code}.pth"
        anchor_path = os.path.join(custom_anchor_dir, anchor_filename)

        # 读取上传 anchor
        uploaded_tensor = torch.load(uploaded_anchor, map_location='cpu')

        if os.path.exists(anchor_path):
            existing_tensor = torch.load(anchor_path, map_location='cpu')
            avg_tensor = (existing_tensor + uploaded_tensor) / 2
            torch.save(avg_tensor, anchor_path)
            st.success(f"🔁 Existing anchor for day {anchor_day} found. Averaged and updated `{anchor_filename}`.")
        else:
            torch.save(uploaded_tensor, anchor_path)
            st.success(f"✅ Anchor saved as `{anchor_filename}`.")

        st.rerun()

    # 显示当前 anchor 列表 + 删除按钮
    existing_files = sorted(os.listdir(custom_anchor_dir))
    if existing_files:
        st.markdown("### 📂 Current Uploaded Anchors")

        for file in existing_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"- `{file}`")
            with col2:
                if st.button("🗑️ Delete", key=f"del_{file}"):
                    os.remove(os.path.join(custom_anchor_dir, file))
                    st.success(f"Deleted `{file}`.")
                    st.rerun()

    # 设置推理目录为上传的 anchor
    st.session_state.anchor_dir = custom_anchor_dir

    # 上传数据路径
    weather_excel = os.path.join(user_temp_folder, 'custom_weather', f"{required_date}.xlsx")

    # 添加运行按钮
    if st.button("▶️ Run Prediction"):
        import glob
        image_count = len(glob.glob(os.path.join(image_folder, "*.jpg"))) + \
                      len(glob.glob(os.path.join(image_folder, "*.jpeg"))) + \
                      len(glob.glob(os.path.join(image_folder, "*.png")))
        estimated_time_sec = image_count * 10
        estimated_minutes = estimated_time_sec / 60

        st.info(f"🖥️ Running on CPU | Total images: {image_count} | Estimated time: ~{estimated_minutes:.1f} minutes")

        predict_and_plot_anchor_votes(
            image_folder=image_folder,
            weather_excel=weather_excel,
            model_ft=st.session_state.model_ft,
            sim_model=st.session_state.sim_model,
            anchor_dir=st.session_state.anchor_dir,
        )



# ---------- Option 3: 自己训练 anchor ----------
elif anchor_option == "Train Your Own Anchor":
    import shutil
    import glob
    import zipfile
    import os

    st.subheader("🧚️ Train Your Own Anchors")
    st.info("Upload image(s) and weather Excel for specific dates and days-before-flowering.\n"
            "Images will be grouped by (day-before-flowering / date) and weather files by date.")

    # 路径定义
    base_path = os.path.join(user_temp_folder, "custom_train_anchor")
    weather_dir = os.path.join(base_path, "weather")


    os.makedirs(base_path, exist_ok=True)
    os.makedirs(weather_dir, exist_ok=True)

    # ---- 📦 FULL FOLDER ZIP UPLOAD ----
    st.markdown("### 📆 Upload Full Folder ZIP (images + weather)")
    full_zip = st.file_uploader("Upload ZIP with structure: images/day/date and weather/*.xlsx", type=["zip"], key="zip_all")

    if full_zip is not None:
        if st.button("📤 Extract ZIP Contents"):
            zip_save_path = os.path.join(base_path, "uploaded_all.zip")
            with open(zip_save_path, "wb") as f:
                f.write(full_zip.read())

            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
            os.remove(zip_save_path)

            st.success("✅ ZIP extracted successfully into custom_train_anchor/")
            st.rerun()

    # ---- ➕ UPLOAD NEW SAMPLE FORM ----
    with st.form("train_anchor_form", clear_on_submit=True):
        st.markdown("### ➕ Upload New Training Sample")
        images = st.file_uploader("📷 Upload image(s) or ZIP:", type=["jpg", "jpeg", "png", "zip"],
                                  accept_multiple_files=True)
        excel_file = st.file_uploader("🌦️ Upload weather Excel (.xlsx):", type=["xlsx"])
        date_str = st.text_input("📅 Enter the date (YYYY-MM-DD)", value="2023-07-12")
        day_value = st.number_input("📈 Enter days before flowering", min_value=1, step=1)
        submit_upload = st.form_submit_button("📁 Submit")

        if submit_upload:
            if not images or not excel_file or not date_str:
                st.warning("⚠️ Please fill in all fields.")
            else:
                # 保存图像
                image_dir = os.path.join(base_path, 'images', str(day_value), date_str)
                os.makedirs(image_dir, exist_ok=True)

                for img in images:
                    if img.name.endswith(".zip"):
                        zip_path = os.path.join(image_dir, img.name)
                        with open(zip_path, "wb") as f:
                            f.write(img.read())
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(image_dir)
                        os.remove(zip_path)
                    else:
                        img_path = os.path.join(image_dir, img.name)
                        with open(img_path, "wb") as f:
                            f.write(img.read())

                # 保存 Excel
                weather_save_path = os.path.join(weather_dir, f"{date_str}.xlsx")
                is_overwrite = os.path.exists(weather_save_path)
                with open(weather_save_path, "wb") as f:
                    f.write(excel_file.read())

                if is_overwrite:
                    st.warning(f"🔁 Weather file for {date_str} already existed and was overwritten.")
                else:
                    st.success(f"✅ Saved weather file to {weather_save_path}")
                st.success(f"✅ Images saved to {image_dir}")

    st.markdown("---")


    # ---- 📂 BROWSE AND DELETE EXISTING DATA ----
    st.markdown("### 🗂️ Current Training Samples")

    sample_dirs = sorted(glob.glob(os.path.join(base_path, "images", "*", "*")))
    weather_files = sorted(glob.glob(os.path.join(weather_dir, "*.xlsx")))

    if not sample_dirs and not weather_files:
        st.markdown("❗ No uploaded training samples yet.")
    else:
        for path in sample_dirs:
            day = os.path.basename(os.path.dirname(path))
            date = os.path.basename(path)
            st.markdown(f"📸 **Images: Day {day} / Date {date}**  —  `{path}`")
            if st.button(f"🗑️ Delete Image Folder {day}/{date}", key=f"del_img_{day}_{date}"):
                shutil.rmtree(path)
                st.success(f"✅ Deleted image folder {path}")
                st.rerun()

        for wf in weather_files:
            wname = os.path.basename(wf)
            date = wname.replace(".xlsx", "")
            st.markdown(f"🌦️ **Weather: {wname}**")
            if st.button(f"🗑️ Delete Weather {date}", key=f"del_weather_{date}"):
                os.remove(wf)
                st.success(f"✅ Deleted weather file {wname}")
                st.rerun()

    # ---- 🚀 GENERATE ANCHORS ----
    day_dirs = glob.glob(os.path.join(base_path, "images", "*"))
    day_list = sorted([os.path.basename(d) for d in day_dirs if os.path.isdir(d)])

    if not day_list:
        st.warning("⚠️ No day folders found in training data.")
    else:
        if st.button("🚀 Generate All Anchors"):
            estimated_minutes = len(day_list) * 5
            st.info(f"🕒 Estimated total generation time: ~{estimated_minutes} minutes for {len(day_list)} day(s). (Running on CPU)")

            from anchor_create import generate_full_anchor_from_dated_folders

            model_code = st.session_state.selected_model_code
            anchor_dir = os.path.join(base_path, "anchor")
            if os.path.exists(anchor_dir):
                shutil.rmtree(anchor_dir)
            os.makedirs(anchor_dir, exist_ok=True)

            generated_files = []

            for day in day_list:
                image_root = os.path.join(base_path, "images", str(day))
                output_path = os.path.join(anchor_dir, f"{str(day)}_{model_code}.pth")

                st.write(f"🔄 Generating anchor for Day {day}...")
                generate_full_anchor_from_dated_folders(
                    image_root_dir=image_root,
                    weather_root_dir=weather_dir,
                    model_ft=st.session_state.model_ft,
                    output_anchor_path=output_path
                )
                generated_files.append(output_path)

            st.success("🎉 All anchors generated and saved!")

            st.markdown("### 📂 Generated Anchors:")
            for f in generated_files:
                st.markdown(f"- `{os.path.basename(f)}`")

            zip_path = os.path.join(base_path, "custom_anchor_output.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for f in generated_files:
                    zipf.write(f, arcname=os.path.basename(f))

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download All Anchors (.zip)",
                    data=f,
                    file_name="custom_anchors.zip",
                    mime="application/zip"
                )


    # 添加运行按钮
    image_folder = os.path.join(user_temp_folder, "custom_image")
    weather_excel = os.path.join(user_temp_folder, "custom_weather", f"{required_date}.xlsx")

    if st.button("▶️ Run Prediction"):
        import glob

        image_count = len(glob.glob(os.path.join(image_folder, "*.jpg"))) + \
                      len(glob.glob(os.path.join(image_folder, "*.jpeg"))) + \
                      len(glob.glob(os.path.join(image_folder, "*.png")))
        estimated_time_sec = image_count * 5
        estimated_minutes = estimated_time_sec / 60

        st.info(f"🖥️ Running on CPU | Total images: {image_count} | Estimated time: ~{estimated_minutes:.1f} minutes")

        predict_and_plot_anchor_votes(
            image_folder=image_folder,
            weather_excel=weather_excel,
            model_ft=st.session_state.model_ft,
            sim_model=st.session_state.sim_model,
            anchor_dir=os.path.join(user_temp_folder, "custom_train_anchor","anchor"),
        )



