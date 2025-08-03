import shutil

import pandas as pd
from PIL import Image
from torchvision import transforms


def generate_anchor(image_folder, weather_excel_path, model_ft, save_path):
    """
    生成 anchor，并保存到 save_path。
    如果 save_path 已存在，则与已有 anchor 融合（取平均）后保存。
    不返回任何内容。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载天气数据
    weather_df = pd.read_excel(weather_excel_path)
    weather_tensor = torch.tensor(weather_df.select_dtypes(include='number').values, dtype=torch.float).to(device)
    weather_tensor = weather_tensor.unsqueeze(0)


    # 提取图像特征并求平均
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_feature = torch.zeros(256, device=device)

    for fname in image_files:
        img_path = os.path.join(image_folder, fname)
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_ft((img_tensor, weather_tensor)).squeeze()
            total_feature += output

    current_anchor = total_feature / len(image_files)

    # 如果已存在 anchor，进行平均融合
    if os.path.exists(save_path):
        print(f"Anchor file exists at {save_path}, loading and averaging with new anchor.")
        existing_anchor = torch.load(save_path, map_location=device)
        combined_anchor = (existing_anchor.to(device) + current_anchor) / 2
        torch.save(combined_anchor.cpu(), save_path)
    else:
        torch.save(current_anchor.cpu(), save_path)

import os
import torch
from tqdm import tqdm
from anchor_create import generate_anchor


def generate_full_anchor_from_dated_folders(
    image_root_dir: str,
    weather_root_dir: str,
    model_ft,
    output_anchor_path: str
):
    """
    遍历 image_root_dir 下所有日期命名的子文件夹（如 2023-07-19），
    搭配 weather_root_dir 中对应日期的 Excel 文件，生成并融合 anchor。
    最后保存融合 anchor 至 output_anchor_path。
    """
    if not os.path.exists("__temp__folder/custom_train_anchor/tmp"):
        os.makedirs("__temp__folder/custom_train_anchor/tmp")
    if os.path.exists("__temp__folder/custom_train_anchor/tmp/__temp_anchor.pth"):
        os.remove("__temp__folder/custom_train_anchor/tmp/__temp_anchor.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_folders = [f for f in os.listdir(image_root_dir) if os.path.isdir(os.path.join(image_root_dir, f))]
    date_folders.sort()  # 可选：确保日期顺序一致

    accumulated_anchor = torch.zeros(256, device=device)
    count = 0

    print(f"🔁 共检测到 {len(date_folders)} 个日期图像文件夹，开始处理...")

    for date_str in tqdm(date_folders, desc="Processing folders", ncols=80):
        image_folder = os.path.join(image_root_dir, date_str)
        weather_excel = os.path.join(weather_root_dir, f"{date_str}.xlsx")

        if not os.path.exists(weather_excel):
            tqdm.write(f"⚠️ 缺失天气文件：{weather_excel}，跳过 {date_str}")
            continue

        # 临时 anchor 输出
        if not os.path.exists('tmp'):
            os.makedirs('tmp')

        temp_anchor_path = "tmp/__temp_anchor.pth"
        generate_anchor(
            image_folder=image_folder,
            weather_excel_path=weather_excel,
            model_ft=model_ft,
            save_path=temp_anchor_path
        )

        anchor_tensor = torch.load(temp_anchor_path, map_location=device)
        accumulated_anchor += anchor_tensor.to(device)
        count += 1

    if count > 0:
        final_anchor = accumulated_anchor / count
        torch.save(final_anchor.cpu(), output_anchor_path)
        print(f"\n✅ 已保存融合 anchor 至：{output_anchor_path}（共融合 {count} 个日期）")
    else:
        print("❌ 没有生成任何 anchor，请检查图像和天气数据是否匹配。")

    shutil.rmtree("__temp__folder/custom_train_anchor/tmp")