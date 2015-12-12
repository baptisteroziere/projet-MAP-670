#!/bin/bash

ssh sofia.calcagno@master-bigdata.polytechnique.fr "ipython notebook --no-browser --port=8888&" 

ssh -N -f -L localhost:8898:localhost:8888 sofia.calcagno@master-bigdata.polytechnique.fr &
xdg-open http://localhost:8898/tree &
exit 0
