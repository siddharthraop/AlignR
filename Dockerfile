# AlignR API: the engine, the corpus, the launch facts and the OASIS DITA RELAX NG grammars, baked in.
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/siddharthraop/AlignR" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ALIGNR_CORPUS=/app/corpus ALIGNR_GRAMMAR=/app/grammar
WORKDIR /app
COPY pyproject.toml README.md ./
COPY alignr alignr
COPY api api
RUN pip install --no-cache-dir '.[api]' && pip uninstall -y pip
COPY corpus/tpu corpus/tpu
COPY corpus/tools/schematron corpus/tools/schematron
COPY corpus/.grammar/rng grammar/rng
COPY facts facts
ARG CORPUS_COMMIT=main
ENV ALIGNR_CORPUS_COMMIT=${CORPUS_COMMIT} ALIGNR_FACTS=/app/facts
RUN useradd --uid 10001 alignr && chown -R alignr /app/grammar
USER alignr
RUN python -c "from alignr.checks import relaxng; relaxng('concept'); relaxng('reference')"
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
