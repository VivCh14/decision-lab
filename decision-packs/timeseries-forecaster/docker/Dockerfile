FROM continuumio/miniconda3:25.3.1-1

WORKDIR /workspace

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# numpy/pandas from conda-forge (scipy 1.15.3 has no py312 conda build, so it comes from pip)
# BLAS-linked numeric stack from conda-forge (versions match docker/modal_app/tsf_runner.py)
RUN conda install -y \
    python=3.12 \
    numpy=2.1.3 \
    pandas=2.2.3 \
    matplotlib \
    pyyaml \
    && conda clean -afy

# Everything else via pip; modal deploys and calls the tsf-runner app
RUN pip install --no-cache-dir \
    scipy==1.15.3 \
    statsmodels==0.15.0 \
    pymc==6.3.2 \
    pytensor==3.3.4 \
    arviz==1.3.0 \
    nutpie==0.16.11 \
    pyarrow==21.0.0 \
    modal==1.3.2

# Library importable as "tsf_lib" (PYTHONPATH=/opt finds /opt/tsf_lib/)
COPY tsf_lib /opt/tsf_lib
COPY modal_app /opt/modal_app
ENV PYTHONPATH="/opt"

RUN python -c "import pymc, arviz, statsmodels, nutpie; print('Bayesian stack OK')"
RUN python -c "import tsf_lib.cli, tsf_lib.modal_runner; print('tsf_lib import OK')"

CMD ["/bin/bash"]
