from PIL import Image
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms
from model_structure import *
import streamlit as st

def predict_and_plot_anchor_votes(
    image_folder: str,
    weather_excel: str,
    model_ft,
    sim_model,
    anchor_dir: str,
):
    """
    遍历 anchor 文件夹，对每个 anchor 进行预测（0=Not Flower, 1=Flower），
    完成所有预测后逐个绘制饼图 + 旁边的 legend。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载天气数据
    weather_df = pd.read_excel(weather_excel)
    weather_tensor = torch.tensor(weather_df.select_dtypes(include='number').values, dtype=torch.float).to(device)
    weather_tensor = weather_tensor.unsqueeze(0)

    # 图像文件列表
    image_list = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # 获取 anchor 文件并排序
    anchor_files = [f for f in os.listdir(anchor_dir) if f.endswith(".pth")]
    anchor_files.sort(key=lambda x: int(x.split('_')[0]))  # 8 → 10 → 12 ...

    results = []  # 存储每个 anchor 的预测结果

    for anchor_file in anchor_files:
        anchor_path = os.path.join(anchor_dir, anchor_file)
        anchor_vector = torch.load(anchor_path, map_location=device).unsqueeze(0).to(device)

        predict_labels = []

        for fname in tqdm(image_list, desc=f"Evaluating {anchor_file}"):
            img_path = os.path.join(image_folder, fname)
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model_ft((img_tensor, weather_tensor))
                sim_score = sim_model(feature, anchor_vector)
                _, pred = torch.max(sim_score.data, 1)
                predict_labels.append(pred.item())

        # 0 = 不会开花, 1 = 会开花
        count = Counter(predict_labels)
        count_dict = {
            "Will Not Flower": count.get(0, 0),
            "Will Flower": count.get(1, 0)
        }
        results.append((anchor_file.split(".")[0], count_dict))
    # 🔽 统一绘图，每个 anchor 一个图，旁边显示 legend
    for anchor_name, count_dict in results:
        labels = list(count_dict.keys())
        sizes = list(count_dict.values())
        colors = ["lightgray", "lightgreen"]  # 0 不开花, 1 开花

        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            autopct="%1.1f%%",
            wedgeprops=dict(width=0.5)
        )

        day_number = anchor_name.split("_")[0]
        ax.set_title(f"Prediction: after {day_number} days", fontsize=14)

        # 图例放在右边
        ax.legend(wedges, labels, title="Prediction", loc="center left", bbox_to_anchor=(1, 0.5))

        ax.axis("equal")  # Equal aspect ratio to ensure circle
        plt.tight_layout()
        st.pyplot(fig)  # 将图表输出到 Streamlit 页面中

#
# model_dir = 'model/Full convnext TF/feature extraction model.pth'
# sim_model_dir = 'model/Full convnext TF/compare model.pth'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# if 'convnext' in model_dir:
#     model_name = 'convnext'
# else:
#     model_name = 'swin_b'
#
# model_ft = FeatureExtractNetwork(model_name=str(model_name))
# model_weights = torch.load(model_dir)
# model_ft.load_state_dict(model_weights)
# model_ft.to(device)
# model_ft.eval()
#
# if 'tf' or 'TF' in sim_model_dir:
#     sim_model = ComparedNetwork_Transformer()
# elif 'fc' or 'FC' in sim_model_dir:
#     sim_model = ComparedNetwork()
#
# sim_model_weights = torch.load(sim_model_dir)
# sim_model.load_state_dict(sim_model_weights)
# sim_model.to(device)
# sim_model.eval()
#
#
# predict_and_plot_anchor_votes(
#     image_folder=r"G:\multi-model_few_shot-YX\officials_anchor\cold_test\15\2023-07-12",
#     weather_excel=r"H:\wheat project\few shot flowering project\paper\few shot dataset\weather/2023-07-22.xlsx",
#     model_ft=model_ft,
#     sim_model=sim_model,
#     anchor_dir=r"G:\multi-model_few_shot-YX\officials_anchor\cold_anchor\convnext_tf",
# )
