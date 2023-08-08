#!/usr/bin/env bash
mamba activate deepmeteor-py310

export PROJECT_HOME=$(readlink -f $(dirname ${(%):-%N}))
export PROJECT_LOG_DIR=${PROJECT_HOME}/logs
export PROJECT_DATA_DIR=${PROJECT_HOME}/data

export PYTHONPATH=${PROJECT_HOME}:${PYTHONPATH}

python --version
which python
typeset -m PROJECT_HOME
typeset -m PROJECT_LOG_DIR
typeset -m PROJECT_DATA_DIR
