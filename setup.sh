#!/bin/bash

if [ ! -d codalab-cli ]
then
  git clone https://github.com/codalab/codalab-cli.git
  (cd codalab-cli && ./setup.sh server)
fi
bower install --force
