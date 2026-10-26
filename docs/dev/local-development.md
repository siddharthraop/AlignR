# Local development

You need Python 3.11 or later and Git. Java 17 or later and DITA-OT 4.4 are needed only for the build check and the
help.

```sh
git clone --recurse-submodules https://github.com/siddharthraop/AlignR.git
cd AlignR
pip install -e '.[api,test]'
corpus/tools/fetch_grammar.sh        # OASIS DITA 1.3 grammars into corpus/.grammar (git-ignored)
python -m pytest -q
```

## Draft without a model

The baseline copies the most recent sibling and swaps its keys. It needs no key, costs nothing, and goes through the
same gates:

```sh
alignr draft --model baseline --out out/baseline
alignr check out/baseline/topics/*.dita            # grammar and rules
alignr check out/baseline/topics/*.dita --merge    # also fails on open questions
```

## Draft with Gemini

Use a free-tier key from Google AI Studio, and only with public content: the free tier may use what you send to
improve Google products.

```sh
export GOOGLE_API_KEY=...          # never commit it; CI reads it from Secret Manager
alignr draft --model gemini --out out/run -v
```

`ALIGNR_MODEL` sets the model (default `gemini-3.8-flash`), `ALIGNR_THINKING` the thinking level (default `low`) and
`ALIGNR_TIMEOUT_MS` the request timeout.

## Run the API and the page

```sh
uvicorn api.main:app --port 8080
python -m http.server 8000         # then open http://localhost:8000/app/?api=http://localhost:8080
```

## Build the drafts with the rest of the set

```sh
alignr stage out/run/topics --to build/tpu
$DITA_HOME/bin/dita -i build/tpu/tpu.ditamap -f html5 -o build/html --processing-mode=strict
```
