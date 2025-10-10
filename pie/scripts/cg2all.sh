eval "$(conda shell.bash hook)" # Init conda
conda activate cg2all # Environment must be available

in_pdb="$1"
out_pdb="$2"
device="$3"

convert_cg2all -p "$in_pdb" \
    -o "$out_pdb" \
    --cg "MainchainModel" \
    --fix \
    --device "$device"