# Client image for the quality check: the speed client's image (same Python / transformers) plus lm-eval.
# Build: docker build -t llm-bench-eval:lm_eval-0.4.13 -f docker/eval.Dockerfile docker
FROM vllm/vllm-openai:v0.31.0
RUN pip install --no-cache-dir "lm_eval[api]==0.4.13"
ENTRYPOINT []
