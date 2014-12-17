#!/bin/bash

$PYTHON -sE setup.py install


export TRAVIS_TAG=$(cat ${RECIPE_DIR}/travis_tag.txt | tr -d \n)
export TRAVIS_BRANCH=$(cat ${RECIPE_DIR}/travis_branch.txt | tr -d \n)

$PYTHON -sE ${RECIPE_DIR}/version.py > __conda_version__.txt
