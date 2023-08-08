#!/usr/bin/env zsh
if [ ! -d ${PROJECT_DATA_DIR} ]; then
    mkdir -vp ${PROJECT_DATA_DIR}
fi


for ((IDX= 0; IDX <= 6; IDX++)); do
    LINK="https://cernbox.cern.ch/remote.php/dav/public-files/JK2InUjatHFxFbf/perfNano_TTbar_PU200.110X_set${IDX}.root"
    wget ${LINK} -P ${PROJECT_DATA_DIR}
done
