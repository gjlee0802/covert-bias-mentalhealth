# covert-bias-mentalhealth

MentalChat16K 기반 Mental Health matched-guise probing 전용 코드입니다.

## 목적
- SAE 문장과 AAE 변환 문장을 짝(pair)으로 구성
- LLM이 MSE(Mental Status Examination) 기반 adjective 세트에 부여하는 확률을 비교
- `AAE - SAE` 확률 차이로 covert bias를 확인

## Scoring Method
- `openai:*`, `hf:*`, `gpt2*` 모델은 **sequence probability scoring**을 사용합니다.
- 각 adjective 후보에 대해 `prompt + " " + adjective`의 조건부 로그확률을 계산합니다.
- multi-token adjective도 그대로 포함합니다.
- 토큰 길이 편향을 줄이기 위해 후보 점수는 `mean token logprob`(길이 정규화)로 계산하고, 후보 간 softmax로 확률화합니다.
- OpenRouter provider가 `token_logprobs`를 반환하지 않으면, `openai:*`는 자동으로 single-token 방식으로 폴백합니다(이때 multi-token adjective는 0 확률 처리).
- `roberta*`, `t5*`는 기존 single-token next-token 방식입니다.

## 구조
- `probing/mgp.py`: 본 실험 실행
- `probing/helpers.py`: 모델 로딩/확률 계산
- `probing/prompting.py`: mental-health 프롬프트
- `scripts/prepare_mentalchat_pairs.py`: MentalChat16K -> pair 생성
- `scripts/run_experiment.sh`: SAE baseline + SAE/AAE 비교 실행
- `scripts/summarize_mental_health_results.py`: 결과 요약
- `data/attributes/mental_attitudes.txt`: MSE 기반 adjective 단어 목록
- `data/pairs/`: 생성된 pair 파일 저장
- `results/`: 확률 결과 pickle 저장

## 실행
1) 데이터 pair 생성
```bash
cd /home/gjlee/aiethics/covert-bias-mentalhealth
python3.10 scripts/prepare_mentalchat_pairs.py --input /path/to/MentalChat16K.jsonl --prefix mentalchat16k
python3 scripts/prepare_mentalchat_pairs.py --input data/raw/MentalChat16K.jsonl --prefix mentalchat16k
```

2) 실험 실행
```bash
bash scripts/run_experiment.sh 0
```

모델을 직접 지정하려면:
```bash
bash scripts/run_experiment.sh 0 \
  "openai:gpt-4.1-mini hf:meta-llama/Llama-3.1-8B-Instruct hf:Qwen/Qwen3-8B hf:mistralai/Mistral-7B-Instruct-v0.3"
```

`openai:*` 모델은 기본적으로 OpenRouter(`https://openrouter.ai/api/v1`)를 사용합니다.
`OPENAI_API_KEY` 환경변수는 반드시 필요합니다.
기본 URL을 바꾸고 싶을 때만 `OPENAI_BASE_URL`을 설정하세요.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

3) 결과 요약
```bash
python3.10 scripts/summarize_mental_health_results.py --model gpt2 --calibrated
```

## SAE-only (Top100) Next-token Table
Nature 논문 설정과 유사하게 SAE 문장만 대상으로 adjective의 다음 1토큰 확률을 집계:

```bash
python3 scripts/sae_next_token_top100.py \
  --model gpt2-medium \
  --top500 top500.txt \
  --attributes data/attributes/mental_attitudes.txt \
  --n 100 \
  --scoring one_token \
  --calibrate
```

multi-token adjective를 스킵하지 않으려면:

```bash
python3 scripts/sae_next_token_top100.py \
  --model gpt2-medium \
  --top500 top500.txt \
  --attributes data/attributes/mental_attitudes.txt \
  --n 100 \
  --scoring sequence \
  --calibrate
```

출력:
- `results/sae_top100_next_token_table.csv`
- `results/sae_top100_next_token_table.md`

## Paper-style Matched-Guise Table
논문과 같은 형태로 `SAE mean`, `AAE mean`, `delta(AAE-SAE)`를 adjective별 표로 출력:

```bash
python3 scripts/paper_style_next_token_table.py \
  --model gpt2-medium \
  --pair_file data/pairs/top500_sae_aae.txt \
  --n 100 \
  --scoring one_token \
  --calibrate
```

AAE 데이터가 아직 없을 때(점검용 SAE=AAE baseline):

```bash
python3 scripts/paper_style_next_token_table.py \
  --model gpt2-medium \
  --top500 top500.txt \
  --n 100 \
  --scoring one_token \
  --calibrate
```

## 출력 파일
- `data/pairs/mentalchat16k_sae_sae.txt`
- `data/pairs/mentalchat16k_sae_aae.txt`
- `results/{model}_mentalchat16k_sae_sae_mental_attitudes_cal.p`
- `results/{model}_mentalchat16k_sae_aae_mental_attitudes_cal.p`

참고:
- `openai:<model>`: OpenAI API 호출
- `hf:<repo>`: Hugging Face CausalLM 로컬 추론
