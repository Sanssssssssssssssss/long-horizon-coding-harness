FROM harness-benchmark-python311:20260908
WORKDIR /testbed
ADD source.tar /testbed/
RUN pip install --no-index --no-deps --no-build-isolation -e . && pip check \
 && git init -q && git add . \
 && git -c user.name=Benchmark -c user.email=benchmark@example.invalid commit -qm "Public task baseline"
