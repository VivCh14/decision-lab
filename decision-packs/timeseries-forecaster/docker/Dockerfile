FROM python:3.12-slim

WORKDIR /workspace

ARG DATABRICKS_CONNECT_VERSION=">=18.0,<19"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    "databricks-connect${DATABRICKS_CONNECT_VERSION}"

# Library importable as "tsf_lib" (PYTHONPATH=/opt finds /opt/tsf_lib/)
COPY tsf_lib /opt/tsf_lib
ENV PYTHONPATH="/opt"

RUN python -c "import pymc, arviz, statsmodels, nutpie; print('Bayesian stack OK')"
RUN python -c "from databricks.connect import DatabricksSession; print('databricks-connect OK')"
RUN python -c "import tsf_lib.worker, tsf_lib.spark_batch; print('tsf_lib import OK')"

CMD ["/bin/bash"]
