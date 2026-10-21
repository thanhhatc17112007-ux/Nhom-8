from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

# ---------------------------------------------------------
# 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ---------------------------------------------------------
DATA_FILE = Path("du_lieu_da_lam_sach.xlsx")
OUTPUT_DIR = Path("ket_qua_MANOVA")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file {DATA_FILE}. Vui lòng đặt file Excel cùng thư mục với file code Python!")
try:
    df = pd.read_excel(DATA_FILE)
except ImportError:
    raise ImportError("Cần cài đặt thư viện 'openpyxl' để đọc file Excel (.xlsx). Hãy chạy: pip install openpyxl")

GROUP_COL = "Dinh_Dang"
Y_COLS = [
    "Luot_Tiep_Can",
    "Ty_Le_Tuong_Tac",
    "Thoi_Gian_Trung_Binh",
    "Ty_Le_Chuyen_Doi",
    "So_Chia_Se",
]
EXPECTED_GROUPS = ["Video", "Article", "Infographic"]

required_cols = [GROUP_COL] + Y_COLS
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Thiếu cột bắt buộc trong file Excel: {missing_cols}")

df = df[required_cols].copy()

# Chuyển các cột biến phụ thuộc sang kiểu số
for col in Y_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 70)
print("KIỂM TRA DỮ LIỆU BAN ĐẦU")
print("=" * 70)
print(f"Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")
print("\nSố giá trị thiếu (NaN) theo cột trước khi làm sạch:")
print(df.isna().sum().to_string())

# SỬA LỖI LOGIC: Loại bỏ dòng có NaN trước khi ép kiểu chuỗi cho GROUP_COL
n_before = len(df)
df = df.dropna(subset=required_cols).copy()

# Sau khi đã xóa NaN mới ép kiểu chuỗi và strip khoảng trắng
df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()

print(f"\nSố dòng được dùng phân tích chính thức: {len(df)} (đã loại {n_before - len(df)} dòng thiếu).")
print("\nSố quan sát theo định dạng:")
print(df[GROUP_COL].value_counts().reindex(EXPECTED_GROUPS, fill_value=0).to_string())

unknown_groups = sorted(set(df[GROUP_COL]) - set(EXPECTED_GROUPS))
if unknown_groups:
    raise ValueError(f"Có nhãn định dạng ngoài dự kiến: {unknown_groups}")
if df[GROUP_COL].nunique() < 2:
    raise ValueError("Cần ít nhất hai nhóm định dạng để thực hiện so sánh.")

# ---------------------------------------------------------
# 2. THỐNG KÊ MÔ TẢ THEO ĐỊNH DẠNG
# ---------------------------------------------------------
long_rows = []
for group in EXPECTED_GROUPS:
    part = df.loc[df[GROUP_COL] == group]
    for col in Y_COLS:
        values = part[col]
        long_rows.append({
            "Dinh_Dang": group,
            "Bien": col,
            "N": int(values.count()),
            "Mean": values.mean(),
            "SD": values.std(ddof=1),
            "Min": values.min(),
            "Max": values.max(),
        })

desc = pd.DataFrame(long_rows)
desc.to_csv(OUTPUT_DIR / "Bang_Thong_Ke_Mo_Ta.csv", index=False, encoding="utf-8-sig")
print("\n" + "=" * 70)
print("THỐNG KÊ MÔ TẢ (Mean, SD, Min, Max)")
print("=" * 70)
print(desc.round(3).to_string(index=False))

# ---------------------------------------------------------
# 3. MANOVA
# ---------------------------------------------------------
formula = " + ".join(Y_COLS) + " ~ C(Dinh_Dang)"
manova = MANOVA.from_formula(formula, data=df)
mv_results = manova.mv_test()
print("\n" + "=" * 70)
print("KẾT QUẢ MANOVA")
print("=" * 70)
print(mv_results)

mv_table = mv_results.results["C(Dinh_Dang)"]["stat"]
mv_export = mv_table.reset_index().rename(columns={"index": "Thong_Ke"})
mv_export.to_csv(OUTPUT_DIR / "Bang_MANOVA.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# 4. ANOVA ĐƠN BIẾN VÀ PARTIAL ETA SQUARED
# ---------------------------------------------------------
anova_rows = []
for col in Y_COLS:
    group_values = [
        df.loc[df[GROUP_COL] == group, col].dropna().to_numpy()
        for group in EXPECTED_GROUPS
    ]
    group_values = [x for x in group_values if len(x) > 0]
    
    k = len(group_values)  # Số nhóm thực tế có dữ liệu
    N = sum(len(x) for x in group_values)
    
    f_value, p_value = stats.f_oneway(*group_values)

    grand_mean = df[col].mean()
    ss_between = sum(
        len(df.loc[df[GROUP_COL] == group, col]) *
        (df.loc[df[GROUP_COL] == group, col].mean() - grand_mean) ** 2
        for group in EXPECTED_GROUPS if (df[GROUP_COL] == group).any()
    )
    ss_total = ((df[col] - grand_mean) ** 2).sum()
    ss_within = ss_total - ss_between
    partial_eta2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) else np.nan

    anova_rows.append({
        "Bien": col,
        "F": f_value,
        "df_giua_nhom": k - 1,
        "df_trong_nhom": N - k,
        "p_value": p_value,
        "Partial_Eta_Squared": partial_eta2,
    })

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUTPUT_DIR / "Bang_ANOVA_Don_Bien.csv", index=False, encoding="utf-8-sig")
print("\n" + "=" * 70)
print("ANOVA ĐƠN BIẾN CHO TỪNG CHỈ TIÊU")
print("=" * 70)
print(anova_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

# ---------------------------------------------------------
# 5. MA TRẬN TƯƠNG QUAN PEARSON
# ---------------------------------------------------------
corr = df[Y_COLS].corr(method="pearson")
corr.to_csv(OUTPUT_DIR / "Ma_Tran_Tuong_Quan_Pearson.csv", encoding="utf-8-sig")
print("\n" + "=" * 70)
print("MA TRẬN TƯƠNG QUAN PEARSON")
print("=" * 70)
print(corr.round(3).to_string())

corr_p = pd.DataFrame(np.nan, index=Y_COLS, columns=Y_COLS)
for i, col1 in enumerate(Y_COLS):
    for j, col2 in enumerate(Y_COLS):
        if i == j:
            corr_p.loc[col1, col2] = np.nan
        elif i < j:
            r, p = stats.pearsonr(df[col1], df[col2])
            corr_p.loc[col1, col2] = p
            corr_p.loc[col2, col1] = p
corr_p.to_csv(OUTPUT_DIR / "P_Value_Tuong_Quan.csv", encoding="utf-8-sig")
print(f"\nĐã lưu toàn bộ bảng kết quả tại: {OUTPUT_DIR.resolve()}")
print("Chạy hoàn tất.")

