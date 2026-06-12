#!/bin/zsh
# Build the two single-object training dirs via SYMLINKS only -- no copies, no rendering.
# The GaussianRenderDataset globs <h5_dir>/*.h5 and pairs each with <renders_dir>/<stem>_view_<i>.png.
#
#   experiments/overfit/setup_data.sh
set -e
cd "$(dirname "$0")/../.."          # repo root
R=experiments/overfit/data
PWD_ABS="$(pwd)"

# --- boxes: Objaverse scene_1441 (#4 candidate) ---------------------------------
# h5 = the existing N=20k prune; real-GT = data_v9/renders (full-scene rasterization,
# already named scene_1441_view_*.png) -> renders_dir points straight at data_v9/renders.
mkdir -p $R/boxes/h5s
ln -sf "$PWD_ABS/data_v9_n20k/h5s/scene_1441.h5" $R/boxes/h5s/scene_1441.h5

# --- tomatoes: external scan ----------------------------------------------------
# h5 = tomatoes_n20000.h5; real-GT = the full-splat reference renders (gsplat_full),
# renamed to the dataset's <stem>_view_<i>.png convention (gsplat_full uses view_%02d).
mkdir -p $R/tomatoes/h5s $R/tomatoes/renders
ln -sf "$PWD_ABS/data_external/tomatoes/h5/tomatoes_n20000.h5" $R/tomatoes/h5s/tomatoes_n20000.h5
for i in {0..13}; do
  src=$(printf "%s/data_external/tomatoes/renders/gsplat_full/view_%02d.png" "$PWD_ABS" $i)
  ln -sf "$src" "$R/tomatoes/renders/tomatoes_n20000_view_${i}.png"
done

echo "boxes    : h5=$(ls $R/boxes/h5s) ; renders_dir=data_v9/renders (scene_1441_view_*)"
echo "tomatoes : h5=$(ls $R/tomatoes/h5s) ; renders=$(ls $R/tomatoes/renders | wc -l) views linked"
