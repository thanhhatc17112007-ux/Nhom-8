# -*- coding: utf-8 -*-
"""
BÀI PHÂN TÍCH MANOVA - ĐỀ TÀI TƯƠNG TÁC MẠNG XÃ HỘI
Dữ liệu: du_lieu_da_lam_sach(2).xlsx
Chạy trong Google Colab
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# =========================
# 1. THIẾT LẬP VÀ ĐỌC DỮ LIỆU
# =========================
DATA_FILE = Path("du_lieu_da_lam_sach.xlsx")
OUTPUT_DIR = Path("ket_qua_MANOVA")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    try:
        from google.colab import files
        print("Chọn file du_lieu_da_lam_sach(2).xlsx để tải lên:")
        files.upload()
    except ImportError:
        pass

    if not DATA_FILE.exists():
        alternative = Path("du_lieu_da_lam_sach.xlsx")
        if alternative.exists():
            DATA_FILE = alternative
        else:
            excel_files = list(Path.cwd().glob("*.xlsx"))
            if len(excel_files) == 1:
                DATA_FILE = excel_files[0]
            else:
                raise FileNotFoundError(
                    "Chưa tìm thấy file Excel. Hãy tải file du_lieu_da_lam_sach(2).xlsx lên Colab rồi chạy lại."
                )

GROUP_COL = "Dinh_Dang"
Y_COLS = [
    "Luot_Tiep_Can",
    "Ty_Le_Tuong_Tac",
    "Thoi_Gian_Trung_Binh",
    "Ty_Le_Chuyen_Doi",
    "So_Chia_Se",
]
GROUPS = ["Video", "Article", "Infographic"]
REQUIRED = [GROUP_COL] + Y_COLS

df = pd.read_excel(DATA_FILE)
missing_columns = [c for c in REQUIRED if c not in df.columns]
if missing_columns:
    raise ValueError(f"File thiếu các cột bắt buộc: {missing_columns}")

df = df[REQUIRED].copy()
for c in Y_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[GROUP_COL] = df[GROUP_COL].astype("string").str.strip()

print("=" * 72)
print("1. KIỂM TRA DỮ LIỆU ĐẦU VÀO")
print("=" * 72)
print("File:", DATA_FILE.resolve())
print("Kích thước:", df.shape)
print("Số giá trị thiếu theo cột:")
print(df.isna().sum().to_string())

# Không âm thầm loại dòng: file này được kỳ vọng là file đã làm sạch.
if df[REQUIRED].isna().any().any():
    raise ValueError(
        "File đầu vào vẫn còn giá trị thiếu/không phải số. Hãy làm sạch dữ liệu "
        "hoặc dùng file du_lieu_da_lam_sach trước khi chạy phân tích."
    )
unknown = sorted(set(df[GROUP_COL].dropna()) - set(GROUPS))
if unknown:
    raise ValueError(f"Có nhãn nhóm không hợp lệ: {unknown}")
if set(df[GROUP_COL].unique()) != set(GROUPS):
    raise ValueError("Dữ liệu phải có đủ ba nhóm Video, Article, Infographic.")

counts = df[GROUP_COL].value_counts().reindex(GROUPS)
print("\nSố quan sát theo nhóm:")
print(counts.to_string())
pd.DataFrame({"Dinh_Dang": counts.index, "So_Quan_Sat": counts.values}).to_csv(
    OUTPUT_DIR / "Kiem_Tra_So_Quan_Sat.csv", index=False, encoding="utf-8-sig"
)

# =========================
# 2. THỐNG KÊ MÔ TẢ
# =========================
desc = (
    df.groupby(GROUP_COL)[Y_COLS]
      .agg(["count", "mean", "std", "min", "max"])
      .reindex(GROUPS)
)
desc.to_csv(OUTPUT_DIR / "Bang_Thong_Ke_Mo_Ta.csv", encoding="utf-8-sig")
print("\n" + "=" * 72)
print("2. THỐNG KÊ MÔ TẢ")
print("=" * 72)
print(desc.round(3).to_string())

# =========================
# 3. KIỂM TRA GIẢ ĐỊNH
# =========================
assumption_rows = []

# 3a. Shapiro-Wilk trên phần dư của từng biến trong từng nhóm.
# Kiểm tra phân phối của từng biến trong từng nhóm.
for group in GROUPS:
    part = df.loc[df[GROUP_COL] == group, Y_COLS]
    for col in Y_COLS:
        residual = part[col] - part[col].mean()
        # Shapiro hỗ trợ tối đa 5000 quan sát; mỗi nhóm ở đây nhỏ hơn mức này.
        stat_value, p_value = stats.shapiro(residual)
        assumption_rows.append({
            "Kiem_Dinh": "Shapiro-Wilk (phần dư từng biến)",
            "Nhom": group, "Bien": col,
            "Thong_Ke": stat_value, "p_value": p_value,
            "Ghi_Chu": "p<0,05: bằng chứng biến lệch chuẩn trong nhóm"
        })

# 3b. Box's M: so sánh ma trận hiệp phương sai giữa các nhóm.
# Dùng xấp xỉ chi-square theo công thức Box M.
covs = []
ns = []
for group in GROUPS:
    mat = df.loc[df[GROUP_COL] == group, Y_COLS].to_numpy(dtype=float)
    covs.append(np.cov(mat, rowvar=False, ddof=1))
    ns.append(mat.shape[0])
p = len(Y_COLS)
g = len(GROUPS)
N = sum(ns)
pooled = sum((ns[i] - 1) * covs[i] for i in range(g)) / (N - g)
sign_pooled, logdet_pooled = np.linalg.slogdet(pooled)
if sign_pooled <= 0:
    box_m, box_p = np.nan, np.nan
else:
    M = (N - g) * logdet_pooled - sum(
        (ns[i] - 1) * np.linalg.slogdet(covs[i])[1] for i in range(g)
    )
    c1 = sum(1 / (ns[i] - 1) for i in range(g)) - 1 / (N - g)
    c2 = (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1)) * (
        sum(1 / (ns[i] - 1)**2 for i in range(g)) - 1 / (N - g)**2
    )
    df_box = (g - 1) * p * (p + 1) / 2
    correction = 1 - c1 * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    chi_box = M * correction
    box_m = M
    box_p = stats.chi2.sf(chi_box, df_box)
assumption_rows.append({
    "Kiem_Dinh": "Box's M (xấp xỉ chi-square)",
    "Nhom": "Tất cả", "Bien": "Ma trận hiệp phương sai",
    "Thong_Ke": box_m, "p_value": box_p,
    "Ghi_Chu": "p<0,05: giả định đồng nhất ma trận hiệp phương sai có thể bị vi phạm"
})

# 3c. Tính VIF để tham khảo mức độ đa cộng tuyến giữa các biến.
X = df[Y_COLS].astype(float)
Xz = (X - X.mean()) / X.std(ddof=1)
X_design = add_constant(Xz)
vif_rows = []
for i, col in enumerate(Y_COLS, start=1):
    vif = variance_inflation_factor(X_design.to_numpy(), i)
    vif_rows.append({"Bien": col, "VIF": vif,
                     "Ghi_Chu": "VIF cao (thường >5 hoặc >10) cần xem xét"})
vif_df = pd.DataFrame(vif_rows)
vif_df.to_csv(OUTPUT_DIR / "Kiem_Tra_VIF.csv", index=False, encoding="utf-8-sig")

# 3d. Kiểm tra các quan sát có khoảng cách Mahalanobis lớn.
mahal_rows = []
threshold = stats.chi2.ppf(0.999, df=len(Y_COLS))
for group in GROUPS:
    part = df.loc[df[GROUP_COL] == group, Y_COLS]
    arr = part.to_numpy(dtype=float)
    center = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    distances = cdist(arr, center.reshape(1, -1), metric="mahalanobis", VI=inv_cov).ravel()**2
    for row_index, distance in zip(part.index, distances):
        mahal_rows.append({
            "Dong_Du_Lieu": int(row_index) + 2,  # +2 vì dòng Excel có hàng tiêu đề
            "Dinh_Dang": group,
            "Mahalanobis_D2": distance,
            "Nguong_ChiSquare_0_999": threshold,
            "Co_Dau_Hieu_Ngoai_Le": bool(distance > threshold)
        })
mahal_df = pd.DataFrame(mahal_rows)
mahal_df.to_csv(OUTPUT_DIR / "Kiem_Tra_Mahalanobis.csv", index=False, encoding="utf-8-sig")

# Mardia's multivariate skewness and kurtosis, calculated separately by group.
# p-value skewness uses chi-square approximation; kurtosis uses normal approximation.
mardia_rows = []
for group in GROUPS:
    arr = df.loc[df[GROUP_COL] == group, Y_COLS].to_numpy(dtype=float)
    n, pvars = arr.shape
    centered = arr - arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)
    inv_cov = np.linalg.pinv(cov)
    # b1,p = average of all squared inner products of standardized vectors
    A = centered @ inv_cov @ centered.T
    b1p = float(np.mean(A**3))
    d2 = np.diag(A)
    b2p = float(np.mean(d2**2))
    df_skew = pvars * (pvars + 1) * (pvars + 2) // 6
    chi_skew = n * b1p / 6
    p_skew = stats.chi2.sf(chi_skew, df_skew)
    expected_kurt = pvars * (pvars + 2)
    var_kurt = 8 * pvars * (pvars + 2) / n
    z_kurt = (b2p - expected_kurt) / np.sqrt(var_kurt)
    p_kurt = 2 * stats.norm.sf(abs(z_kurt))
    mardia_rows.extend([
        {"Kiem_Dinh": "Mardia - độ lệch", "Nhom": group,
         "Thong_Ke": b1p, "Gia_Tri_Chuan_Hoa": chi_skew,
         "p_value": p_skew,
         "Ghi_Chu": "Xấp xỉ tiệm cận; p<0,05 gợi ý lệch chuẩn đa biến"},
        {"Kiem_Dinh": "Mardia - độ nhọn", "Nhom": group,
         "Thong_Ke": b2p, "Gia_Tri_Chuan_Hoa": z_kurt,
         "p_value": p_kurt,
         "Ghi_Chu": "Xấp xỉ chuẩn; p<0,05 gợi ý lệch chuẩn đa biến"}
    ])
mardia_df = pd.DataFrame(mardia_rows)
mardia_df.to_csv(OUTPUT_DIR / "Kiem_Tra_Mardia.csv", index=False, encoding="utf-8-sig")

assumptions_df = pd.DataFrame(assumption_rows)
assumptions_df.to_csv(OUTPUT_DIR / "Kiem_Tra_Gia_Dinh.csv", index=False, encoding="utf-8-sig")
print("\n" + "=" * 72)
print("3. KIỂM TRA GIẢ ĐỊNH - BOX'S M")
print(f"Box's M = {box_m:.6g}; p = {box_p:.6g}")
print("Nếu p < 0,05: có dấu hiệu ma trận hiệp phương sai không đồng nhất.")
print("\nMardia:")
print(mardia_df.round(5).to_string(index=False))
print("\nVIF:")
print(vif_df.round(3).to_string(index=False))
print("\nNgoại lệ đa biến theo Mahalanobis (ngưỡng chi-square 99,9%):")
print(mahal_df.groupby("Dinh_Dang")["Co_Dau_Hieu_Ngoai_Le"].agg(["sum", "count"]).to_string())

# =========================
# 4. MANOVA
# =========================
formula = " + ".join(Y_COLS) + " ~ C(Dinh_Dang)"
mv_results = MANOVA.from_formula(formula, data=df).mv_test()
print("\n" + "=" * 72)
print("4. KẾT QUẢ MANOVA")
print("=" * 72)
print(mv_results)
mv_table = mv_results.results["C(Dinh_Dang)"]["stat"].copy()
mv_table.index.name = "Thong_Ke"
mv_table.reset_index().to_csv(
    OUTPUT_DIR / "Bang_MANOVA.csv", index=False, encoding="utf-8-sig"
)

# =========================
# 5. ANOVA ĐƠN BIẾN + PARTIAL ETA SQUARED
# =========================
anova_rows = []
for col in Y_COLS:
    arrays = [df.loc[df[GROUP_COL] == group, col].to_numpy(dtype=float) for group in GROUPS]
    f_value, p_value = stats.f_oneway(*arrays)
    grand_mean = df[col].mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean)**2 for a in arrays)
    ss_within = sum(((a - a.mean())**2).sum() for a in arrays)
    eta_p = ss_between / (ss_between + ss_within)
    anova_rows.append({
        "Bien": col, "F": f_value, "df_giua_nhom": len(GROUPS)-1,
        "df_trong_nhom": len(df)-len(GROUPS), "p_value": p_value,
        "Partial_Eta_Squared": eta_p
    })
anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUTPUT_DIR / "Bang_ANOVA_Don_Bien.csv", index=False, encoding="utf-8-sig")
print("\n" + "=" * 72)
print("5. ANOVA ĐƠN BIẾN")
print(anova_df.round(6).to_string(index=False))

# =========================
# 6. POST-HOC TUKEY HSD
# =========================
posthoc_rows = []
for col in Y_COLS:
    tukey = pairwise_tukeyhsd(
        endog=df[col].to_numpy(dtype=float),
        groups=df[GROUP_COL].to_numpy(),
        alpha=0.05
    )
    table = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )
    table.insert(0, "Bien", col)
    posthoc_rows.append(table)
posthoc_df = pd.concat(posthoc_rows, ignore_index=True)
posthoc_df.to_csv(OUTPUT_DIR / "PostHoc_Tukey_HSD.csv", index=False, encoding="utf-8-sig")
print("\n" + "=" * 72)
print("6. POST-HOC TUKEY HSD (các cặp nhóm)")
print(posthoc_df.to_string(index=False))

# =========================
# 7. TƯƠNG QUAN PEARSON
# =========================
corr = df[Y_COLS].corr(method="pearson")
corr.to_csv(OUTPUT_DIR / "Ma_Tran_Tuong_Quan_Pearson.csv", encoding="utf-8-sig")
p_corr = pd.DataFrame(np.nan, index=Y_COLS, columns=Y_COLS)
for i, c1 in enumerate(Y_COLS):
    for j in range(i + 1, len(Y_COLS)):
        c2 = Y_COLS[j]
        r, pval = stats.pearsonr(df[c1], df[c2])
        p_corr.loc[c1, c2] = pval
        p_corr.loc[c2, c1] = pval
p_corr.to_csv(OUTPUT_DIR / "P_Value_Tuong_Quan.csv", encoding="utf-8-sig")
print("\n" + "=" * 72)
print("7. MA TRẬN TƯƠNG QUAN PEARSON")
print(corr.round(4).to_string())

# =========================
# 8. KIỂM TRA LẠI FILE ĐẦU VÀO
# =========================
cleaning_summary = pd.DataFrame([
    {"Noi_Dung": "Số dòng trong file đầu vào", "Gia_Tri": len(df)},
    {"Noi_Dung": "Số giá trị thiếu", "Gia_Tri": int(df[REQUIRED].isna().sum().sum())},
    {"Noi_Dung": "Số định dạng nội dung", "Gia_Tri": df[GROUP_COL].nunique()},
])
cleaning_summary.to_csv(OUTPUT_DIR / "Tom_Tat_Lam_Sach.csv", index=False, encoding="utf-8-sig")

print("\nĐã lưu các bảng kết quả vào:", OUTPUT_DIR.resolve())
print("Lưu ý: không tự động xóa các quan sát Mahalanobis bị gắn cờ; cần kiểm tra dữ liệu gốc và giải thích trước khi quyết định.")
