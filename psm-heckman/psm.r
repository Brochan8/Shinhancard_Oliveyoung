install.packages(c("MatchIt","cobalt","ggplot2","dplyr"))
library(MatchIt)
library(cobalt)
library(ggplot2)
library(dplyr)


df <- read.csv("Shinhancard_data20221213.csv", stringsAsFactors = FALSE)

## 1) 데이터 전처리
# (권장) 공란 처리 → NA
blank_to_na <- function(x) {
  if (is.character(x)) x[nchar(trimws(x)) == 0] <- NA
  x
}
df <- df %>% mutate(across(everything(), blank_to_na))

# 처리변수(treat): 메시지 응답/리딤 여부
df <- df %>%
  mutate(treat = ifelse(OFF_YN == 1, 1L, 0L))

# 범주형(팩터) 지정
df <- df %>%
  mutate(
    SEX_GB = as.factor(SEX_GB),   # 0/1이면 factor로
    AGE_GB = as.factor(AGE_GB)    # "20_1","30_2" 등 연령그룹
  )


## 2)공변량 결측 처리
zero_vars <- c("AVG_36_M","AVG_12_M",
               "USE_CNT_18Y","USE_AMT_18Y1",
               "CNT_18Y","CNT_18Y_OLV")

df <- df %>%
  mutate(across(all_of(zero_vars), ~ as.numeric(replace(., is.na(.), 0))))

# 공변량 결측을 0으로 처리하는게 싫다면 아래의 코드 실행
#df <- df %>% select(treat, SEX_GB, AGE_GB, all_of(zero_vars), USE_AMT_1M1) %>% na.omit()


## 3) 매칭 전 t-test (raw) — 효과의 “원래 차이” 확인

# 결과변수(y) 예시: 2019년 1월 올영 지출액
yvar <- "USE_AMT_1M1"

# 매칭 전 평균차이
t.test(df[[yvar]] ~ df$treat)


## 4) PSM

# 로짓 PS, 1:1 최근접, caliper=0.001, 재사용X
m.out <- matchit(
  treat ~ SEX_GB + AGE_GB +
    AVG_36_M + AVG_12_M +
    USE_CNT_18Y + USE_AMT_18Y1 +
    CNT_18Y + CNT_18Y_OLV,
  data     = df,
  method   = "nearest",
  distance = "logit",
  caliper  = 0.001,
  replace  = FALSE
)

# 요약 & 균형 진단
summary(m.out)            # 매칭 전/후 SMD, 표본수 등
bal.tab(m.out)            # 수치 테이블
love.plot(m.out, binary="std", thresholds=0.1) +
  geom_vline(xintercept=c(-0.25,0.25), linetype=3, color="blue")

# 성향점수 분포(매칭 전/후)
plot(m.out, type="hist")

## 5) 매칭된 데이터로 ATT 추정 (t-test)
matched <- match.data(m.out)

# 매칭 후 평균차이(ATT)
tt <- t.test(matched[[yvar]] ~ matched$treat)
tt
att <- diff(tt$estimate)   # treated - control
att


