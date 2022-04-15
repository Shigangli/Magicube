cd ./SpMM_basic
chmod 777 setup.sh
./setup.sh
echo "SpMM basic compiled."

cd -
cd ./16b8b/SpMM_conflict_free
chmod 777 setup.sh
./setup.sh

cd -
cd ./8b8b/SpMM_conflict_free
chmod 777 setup.sh
./setup.sh

cd -
cd ./8b4b/SpMM_conflict_free
chmod 777 setup.sh
./setup.sh

cd -
cd ./4b4b/SpMM_conflict_free
chmod 777 setup.sh
./setup.sh
echo "SpMM with conflict-free SM compiled."

cd -
cd ./16b8b/SpMM_conflict_free_prefetch
chmod 777 setup.sh
./setup.sh

cd -
cd ./8b8b/SpMM_conflict_free_prefetch
chmod 777 setup.sh
./setup.sh

cd -
cd ./8b4b/SpMM_conflict_free_prefetch
chmod 777 setup.sh
./setup.sh

cd -
cd ./4b4b/SpMM_conflict_free_prefetch
chmod 777 setup.sh
./setup.sh
echo "SpMM with conflict-free SM + prefetch compiled."

cd -
cd ./8b4b/SpMM_conflict_free_prefetch_shuffle
chmod 777 setup.sh
./setup.sh

cd -
cd ./4b4b/SpMM_conflict_free_prefetch_shuffle
chmod 777 setup.sh
./setup.sh
echo "SpMM with conflict-free SM + prefetch + shuffle compiled."
