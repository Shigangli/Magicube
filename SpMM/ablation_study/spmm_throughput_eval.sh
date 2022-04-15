cd ./SpMM_basic
chmod 777 run_jobs.sh
./run_jobs.sh
echo "SpMM basic tested."

cd -
cd ./16b8b/SpMM_conflict_free
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./8b8b/SpMM_conflict_free
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./8b4b/SpMM_conflict_free
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./4b4b/SpMM_conflict_free
chmod 777 run_jobs.sh
./run_jobs.sh
echo "SpMM with conflict-free SM tested."

cd -
cd ./16b8b/SpMM_conflict_free_prefetch
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./8b8b/SpMM_conflict_free_prefetch
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./8b4b/SpMM_conflict_free_prefetch
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./4b4b/SpMM_conflict_free_prefetch
chmod 777 run_jobs.sh
./run_jobs.sh
echo "SpMM with conflict-free SM + prefetch tested."

cd -
cd ./8b4b/SpMM_conflict_free_prefetch_shuffle
chmod 777 run_jobs.sh
./run_jobs.sh

cd -
cd ./4b4b/SpMM_conflict_free_prefetch_shuffle
chmod 777 run_jobs.sh
./run_jobs.sh
echo "SpMM with conflict-free SM + prefetch + shuffle tested."
