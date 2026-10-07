# SFU register 실험 안내

Native SFU microbenchmark가 기본 nonlinear 실험이 되면서 실행·분모·차분 추정량·
검증 절차를 [Nonlinear 기본 실험 안내](nonlinear-experiments.ko.md)로 통합했다.

현재 기본 설정은 `configs/nonlinear.json`, `configs/nonlinear-smoke.json`,
`configs/nonlinear-amortization.json`이다. 기존 `configs/sfu-register*.json`은
같은 native SFU 실험의 호환 설정으로 유지된다.

Global 입출력과 reduction을 포함한 과거 전체 함수 실험은
[legacy streaming 안내](legacy/nonlinear-streaming.ko.md)를 따른다. 새 실행에는
`nonlinear_mode: "streaming"` 또는 worker의 `--nonlinear-mode streaming`을 명시한다.
