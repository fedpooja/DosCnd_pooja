
# Single Dataset

datasets=('PROTEINS' 'COLLAB' 'IMDB-BINARY' 'DHFR' )
# datasets=('COLLAB' 'IMDB-BINARY' 'DHFR')
datasets=('PROTEINS')



cr_ratios=(0.1 0.2 0.3 0.5)
cr_ratios=(0.5)

priv_budgets=(0.3 0.5 0.6 0.7)
priv_budgets=(0.3)

for data in "${datasets[@]}"; do
    eps1=0.03
    eps2=0.06
    if [ "$data" == "PROTEINS" ]; then
        eps1=0.03
        eps2=0.06
    elif [ "$data" == "DHFR" ]; then
        eps1=0.04
        eps2=0.04
    elif [ "$data" == "IMDB-BINARY" ]; then
        eps1=0.025
        eps2=0.045
    elif [ "$data" == "COX2" ]; then
        eps1=0.04
        eps2=0.04
    elif [ "$data" == "COLLAB" ]; then
        eps1=0.04
        eps2=0.04
    fi
    eps12="$eps1"_"$eps2"
    echo $eps12
    python main.py --repeat 1 --data_group $data --num_clients 10 --seed 10  --epsilon1 $eps1 --epsilon2 $eps2 --seq_length 10 --strategy 'SDMC'
    python aggregator.py --inpath "outputs/seqLen10/Standard/oneDS-nonOverlap/$data-10clients/eps_$eps12/repeats/" --outpath "outputs/seqLen10/Standard/oneDS-nonOverlap/$data-10clients/eps_$eps12/" --data_partition SDMC

    for cr_ratio in "${cr_ratios[@]}"; do
        python main.py --repeat 1 --data_group $data --num_clients 10 --seed 10  --epsilon1 $eps1 --epsilon2 $eps2 --seq_length 10 --strategy 'SDMC' --cr 'True' --cr_ratio $cr_ratio
        python aggregator.py --inpath "outputs/seqLen10/Coarsen/$cr_ratio/oneDS-nonOverlap/$data-10clients/eps_$eps12/repeats/" --outpath "outputs/seqLen10/Coarsen/$cr_ratio/oneDS-nonOverlap/$data-10clients/eps_$eps12/" --data_partition SDMC
    done
    for priv in "${priv_budgets[@]}"; do
        python main.py --repeat 1 --data_group $data --num_clients 10 --seed 10  --epsilon1 $eps1 --epsilon2 $eps2 --seq_length 10 --strategy 'SDMC' --dp 'True' --priv_budget $priv
        python aggregator.py --inpath "outputs/seqLen10/DP/$priv/oneDS-nonOverlap/$data-10clients/eps_$eps12/repeats/" --outpath "outputs/seqLen10/DP/$priv/oneDS-nonOverlap/$data-10clients/eps_$eps12/" --data_partition SDMC
    done
done

#Multi Dataset

# datasets=('molecules' 'proteins' 'social' 'md1' 'md2')
datasets=('molecules')

cr_ratios=(0.1 0.2 0.3 0.5)
cr_ratios=(0.5)

priv_budgets=(0.3 0.5 0.6 0.7)
priv_budgets=(0.3)

for data in "${datasets[@]}"; 
do
    if [ "$data" == "molecules" ]; then
        eps1=0.07
        eps2=0.28
    elif [ "$data" == "mol" ]; then
        eps1=0.07
        eps2=0.28
    elif [ "$data" == "proteins" ]; then
        eps1=0.07
        eps2=0.35
    elif [ "$data" == "biochem" ]; then
        eps1=0.07
        eps2=0.35
    elif [ "$data" == "social" ]; then
        eps1=0.08
        eps2=0.4
    elif [ "$data" == "mix" ]; then
        eps1=0.08
        eps2=0.4
    elif [ "$data" == "md1" ]; then
        eps1=0.07
        eps2=0.3
    elif [ "$data" == "md2" ]; then
        eps1=0.08
        eps2=0.35
    fi
    eps12="$eps1"_"$eps2"
    echo $s
    python main.py --repeat 1 --data_group $data --seed 10 --epsilon1 $eps1 --epsilon2 $eps2 --seq_length 5 --strategy 'MDMC'
    python aggregator.py --inpath "outputs/seqLen5/Standard/multiDS-nonOverlap/$data/eps_$eps12/repeats/"  --outpath "outputs/seqLen5/Standard/multiDS-nonOverlap/$data/eps_$eps12/" --data_partition MDMC
    for cr_ratio in "${cr_ratios[@]}"; 
    do
        python main.py --repeat 1 --data_group $data --seed 10 --epsilon1 $eps1 --epsilon2 $eps2 --seq_length 5 --strategy 'MDMC' --cr 'True' --cr_ratio $cr_ratio
        python aggregator.py --inpath "outputs/seqLen5/Coarsen/$cr_ratio/multiDS-nonOverlap/$data/eps_$eps12/repeats/"  --outpath "outputs/seqLen5/Coarsen/$cr_ratio/multiDS-nonOverlap/$data/eps_$eps12/" --data_partition MDMC
    done
    for priv in "${priv_budgets[@]}"; 
    do
        python main.py --repeat 1 --data_group $data --seed 10 --epsilon1 $eps1 --epsilon2 $eps2 --seq_length 5 --strategy 'MDMC' --dp 'True' --priv_budget $priv
        python aggregator.py --inpath "outputs/seqLen5/DP/$priv/multiDS-nonOverlap/$data/eps_$eps12/repeats/"  --outpath "outputs/seqLen5/DP/$priv/multiDS-nonOverlap/$data/eps_$eps12/" --data_partition MDMC
    done
done