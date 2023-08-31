#!/bin/bash
# This script return the current HEAD commit hash
#
# It fails if the current working directory contain changes.
#

git_status_cmd="git status --untracked-files=no --porcelain"

if [[ ! -z "$($git_status_cmd)" ]]
then
  echo "ERROR can not compute commit idenfier." 1>&2
  echo "Working directory contains modifications." 1>&2
  $git_status_cmd  1>&2
  exit 1
fi

git rev-parse --verify HEAD
