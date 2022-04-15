cd ./SpMM_basic
./setup.sh
echo "SpMM basic compiled."

cd -
cd ./16b8b/SpMM_conflict_free
./setup.sh

cd -
cd ./8b8b/SpMM_conflict_free
./setup.sh

cd -
cd ./8b4b/SpMM_conflict_free
./setup.sh

cd -
cd ./4b4b/SpMM_conflict_free
./setup.sh
echo "SpMM with conflict-free SM compiled."

cd -
cd ./16b8b/SpMM_conflict_free_prefetch
./setup.sh

cd -
cd ./8b8b/SpMM_conflict_free_prefetch
./setup.sh

cd -
cd ./8b4b/SpMM_conflict_free_prefetch
./setup.sh

cd -
cd ./4b4b/SpMM_conflict_free_prefetch
./setup.sh
echo "SpMM with conflict-free SM + prefetch compiled."

cd -
cd ./8b4b/SpMM_conflict_free_prefetch_shuffle
./setup.sh

cd -
cd ./4b4b/SpMM_conflict_free_prefetch_shuffle
./setup.sh
echo "SpMM with conflict-free SM + prefetch + shuffle compiled."
