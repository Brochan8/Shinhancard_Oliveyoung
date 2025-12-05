# ==============================
# 1. 환경 설정
# ==============================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_PATH = "Shinhancard_data20251205_cleaned.csv"  # CSV 파일 경로
# ==============================
# 2. 데이터 로드 및 전처리
# ==============================

# (1) CSV 읽기
df = pd.read_csv(DATA_PATH)

# (2) 사용할 변수 정의
x_cols = [
    "PERIOD_M",         # 가입경과월수
    "AVG_36_M",         # 직전 3년간 월평균 결제액
    "AVG_12_M",         # 18년 1년간 월평균 결제액
    "USE_CNT_18Y",      # 18년 올리브영 이용건수
    "USE_AMT_18Y1",     # 18년 올리브영 총결제액
    "CNT_18Y",          # 18년 오퍼담은 횟수
    "CNT_18Y_OLV",      # 18년 오퍼담은 횟수(올리브영)
    "MSG0",             # 기본 프로모션
    "MSG1",             # 실험 프로모션 1
    "MSG2",             # 실험 프로모션 2
    "MSG3",             # 실험 프로모션 3
    "MSG4",             # 실험 프로모션 4
    "MSG5",             # 실험 프로모션 5
    "Weekend",          # 오퍼 전송 시 주말/주중 여부
    "Badair"            # 오퍼 전송 시 대기오염정도
]

t_col = "OFF_YN"         # Treatment: 오퍼담기 여부(1: 오퍼담음 / 0: 오퍼담지 않음)
y_col = "GAP_MIN1_USE"   # Outcome: 오퍼 담은 이후 첫번째 올리브영 방문 지출액

# (3) 필요한 컬럼만 남기기
cols_needed = x_cols + [t_col, y_col]
df = df[cols_needed].copy()

# (4) 결측값 처리: 일단은 단순 dropna
df[cols_needed] = df[cols_needed].replace(r"^\s*$", np.nan, regex=True) #공백문자열 처리
df = df.dropna(subset=cols_needed)

# (5) X, T, Y 분리
X = df[x_cols].values.astype(np.float32)
t = df[t_col].values.astype(np.float32)
y = df[y_col].values.astype(np.float32)

# (6) 연속형 변수만 표준화 (평균 0, 표준편차 1)
#    더미/범주형은 그대로 둠
continuous_cols = [
    "PERIOD_M",
    "AVG_36_M",
    "AVG_12_M",
    "USE_CNT_18Y",
    "USE_AMT_18Y1",
    "CNT_18Y",
    "CNT_18Y_OLV"
]
binary_cols = [
    "MSG0", "MSG1", "MSG2", "MSG3", "MSG4", "MSG5",
    "Weekend", "Badair"
]

cont_idx = [x_cols.index(c) for c in continuous_cols]

scaler = StandardScaler()
X_cont = scaler.fit_transform(X[:, cont_idx])

# 표준화된 값으로 교체
X_scaled = X.copy()
X_scaled[:, cont_idx] = X_cont

# (7) Train / Test split
X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
    X_scaled, t, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=(t > 0.5)  # treatment 비율 유지
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ==============================
# 3. Dataset & DataLoader 정의
# ==============================

class CausalDataset(Dataset):
    def __init__(self, X, t, y):
        self.X = torch.from_numpy(X).float()
        self.t = torch.from_numpy(t).float().unsqueeze(1)  # (N,1)
        self.y = torch.from_numpy(y).float().unsqueeze(1)  # (N,1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.t[idx], self.y[idx]

batch_size = 256

train_ds = CausalDataset(X_train, t_train, y_train)
test_ds  = CausalDataset(X_test,  t_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ==============================
# 4. DragonNet 모델 정의
# ==============================

class DragonNet(nn.Module):
    def __init__(self, input_dim, hidden_shared=200, hidden_outcome=100):
        super(DragonNet, self).__init__()
        # Shared representation network
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_shared),
            nn.ReLU(),
            nn.Linear(hidden_shared, hidden_shared),
            nn.ReLU()
        )

        # Propensity head (g(X) = P(T=1|X))
        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_shared, hidden_shared // 2),
            nn.ReLU(),
            nn.Linear(hidden_shared // 2, 1),
            nn.Sigmoid()
        )

        # Outcome head for T=0 (y0)
        self.outcome0_head = nn.Sequential(
            nn.Linear(hidden_shared, hidden_outcome),
            nn.ReLU(),
            nn.Linear(hidden_outcome, 1)
        )

        # Outcome head for T=1 (y1)
        self.outcome1_head = nn.Sequential(
            nn.Linear(hidden_shared, hidden_outcome),
            nn.ReLU(),
            nn.Linear(hidden_outcome, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        p = self.propensity_head(h)  # (N,1)
        y0 = self.outcome0_head(h)   # (N,1)
        y1 = self.outcome1_head(h)   # (N,1)
        return y0, y1, p


# ==============================
# 5. DragonNet 손실 함수 정의 (간단 TarReg 버전)
# ==============================

def dragonnet_loss(y_true, t, y0_pred, y1_pred, p_pred,
                   alpha=1.0, beta=1.0):
    """
    y_true : 실제 결과 (N,1)
    t      : 처리 여부 (N,1) 0 또는 1
    y0_pred, y1_pred : DragonNet의 결과 예측 (N,1)
    p_pred : 성향 점수 예측 P(T=1|X) (N,1)
    alpha, beta : 가중치 하이퍼파라미터
    """
    # 관측된 처리에 해당하는 outcome 예측만 사용
    # y_pred = t * y1 + (1-t) * y0
    y_pred = t * y1_pred + (1.0 - t) * y0_pred

    # 1) Outcome prediction loss (MSE)
    loss_y = nn.MSELoss()(y_pred, y_true)

    # 2) Propensity loss (Binary Cross-Entropy)
    loss_t = nn.BCELoss()(p_pred, t)

    # 3) Targeted Regularization (간단 버전)
    #   (t - p) * (y1 - y0)를 0에 가깝게 만들도록 하는 정규화
    #   논문에서는 더 정교하지만 여기서는 직관 위주의 구현
    tar_reg = torch.mean((t - p_pred) * (y1_pred - y0_pred))

    loss = loss_y + alpha * loss_t + beta * (tar_reg ** 2)
    return loss, loss_y, loss_t, tar_reg
#%%
# ==============================
# 6. 모델 생성 및 학습 루프
# ==============================

input_dim = X_train.shape[1]
model = DragonNet(input_dim=input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 10

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    running_loss_y = 0.0
    running_loss_t = 0.0

    for xb, tb, yb in train_loader:
        xb = xb.to(device)
        tb = tb.to(device).float()
        yb = yb.to(device)

        optimizer.zero_grad()
        y0_pred, y1_pred, p_pred = model(xb)
        loss, loss_y, loss_t, tar_reg = dragonnet_loss(
            yb, tb, y0_pred, y1_pred, p_pred,
            alpha=1.0, beta=1.0
        )
        loss.backward()
        optimizer.step()

        batch_size_cur = xb.size(0)
        running_loss   += loss.item()   * batch_size_cur
        running_loss_y += loss_y.item() * batch_size_cur
        running_loss_t += loss_t.item() * batch_size_cur

    epoch_loss   = running_loss   / len(train_loader.dataset)
    epoch_loss_y = running_loss_y / len(train_loader.dataset)
    epoch_loss_t = running_loss_t / len(train_loader.dataset)

    print(f"[Epoch {epoch:03d}] "
          f"Total: {epoch_loss:.4f} | "
          f"Y_loss: {epoch_loss_y:.4f} | "
          f"T_loss: {epoch_loss_t:.4f}")
#%%
# ==============================
# 7. 추론: ITE, ATE 계산
# ==============================

model.eval()
ite_list = []
y0_list = []
y1_list = []
t_list  = []
y_true_list = []

with torch.no_grad():
    for xb, tb, yb in test_loader:
        xb = xb.to(device)
        y0_pred, y1_pred, p_pred = model(xb)

        ite = (y1_pred - y0_pred).cpu().numpy()
        y0_list.append(y0_pred.cpu().numpy())
        y1_list.append(y1_pred.cpu().numpy())
        ite_list.append(ite)
        t_list.append(tb.numpy())
        y_true_list.append(yb.numpy())

ite_array = np.vstack(ite_list)
y0_array  = np.vstack(y0_list)
y1_array  = np.vstack(y1_list)
t_array   = np.vstack(t_list)
y_true_array = np.vstack(y_true_list)

# 개별 ITE 평균 = 추정된 ATE
ate_est = ite_array.mean()
print("\n========================")
print("Estimated ATE (mean ITE):", ate_est)
print("========================")

# 추출값을 CSV로 저장
out_df = pd.DataFrame({
    "ITE_hat": ite_array.flatten(),
    "T": t_array.flatten(),
    "Y_true": y_true_array.flatten(),
    "Y1_hat": y1_array.flatten(),
    "Y0_hat": y0_array.flatten()
})
out_df.to_csv("dragonnet_ITE_results.csv", index=False)
print("Saved ITE results to dragonnet_ITE_results.csv")
#%%
# ==============================
# 전체 고객 ITE 계산
# ==============================
all_ds = CausalDataset(X_scaled, t, y)
all_loader = DataLoader(all_ds, batch_size=256, shuffle=False)

model.eval()
ite_list = []
y0_list = []
y1_list = []
t_list  = []
y_true_list = []

with torch.no_grad():
    for xb, tb, yb in all_loader:
        xb = xb.to(device)
        y0_pred, y1_pred, p_pred = model(xb)

        ite = (y1_pred - y0_pred).cpu().numpy()
        y0_list.append(y0_pred.cpu().numpy())
        y1_list.append(y1_pred.cpu().numpy())
        ite_list.append(ite)
        t_list.append(tb.numpy())
        y_true_list.append(yb.numpy())

ite_array = np.vstack(ite_list)
y0_array  = np.vstack(y0_list)
y1_array  = np.vstack(y1_list)
t_array   = np.vstack(t_list)
y_true_array = np.vstack(y_true_list)

# 전체 ATE
ate_est = ite_array.mean()
print("ATE (전체):", ate_est)

# CSV 저장
out_df = pd.DataFrame({
    "ITE_hat": ite_array.flatten(),
    "T": t_array.flatten(),
    "Y_true": y_true_array.flatten(),
    "Y1_hat": y1_array.flatten(),
    "Y0_hat": y0_array.flatten()
})
out_df.to_csv("dragonnet_ITE_results_full.csv", index=False)
print("Saved FULL ITE results (N =", len(out_df), ")")

#%%
import numpy as np
import pandas as pd

# 1) 입력 변수 X_scaled 저장
X_df = pd.DataFrame(X_scaled, columns=[
    "PERIOD_M",
    "AVG_36_M",
    "AVG_12_M",
    "USE_CNT_18Y",
    "USE_AMT_18Y1",
    "CNT_18Y",
    "CNT_18Y_OLV",
    "MSG0","MSG1","MSG2","MSG3","MSG4","MSG5",
    "Weekend","Badair"
])

# 2) Treatment & Outcome
T_df = pd.DataFrame({"T": t})
Y_df = pd.DataFrame({"Y_true": y})

# 3) DragonNet 추론값 (이미 만든 변수 이용)
ITE_df = pd.DataFrame(ite_array, columns=["ITE_hat"])
Y1_df  = pd.DataFrame(y1_array,  columns=["Y1_hat"])
Y0_df  = pd.DataFrame(y0_array,  columns=["Y0_hat"])

# 4) 모든 변수 병합
df_all = pd.concat([X_df, T_df, Y_df, Y1_df, Y0_df, ITE_df], axis=1)

# 5) CSV 저장
df_all.to_csv("dragonnet_full_dataset_for_R_SHAP.csv", index=False)

print("Saved dragonnet_full_dataset_for_R_SHAP.csv successfully!")
print("Final shape:", df_all.shape)
