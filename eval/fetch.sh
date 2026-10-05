#!/usr/bin/env bash
# Download the evaluation datasets into eval/data/<name>/{images,masks}.
# These are third-party copies committed to public GitHub repositories (the
# official academic hosts are often unreachable). Research use only, see
# eval/README.md for sources and citations. Usage: eval/fetch.sh [name...]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data .cache
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(ECSSD PASCAL-S DUTS-TE MSRA10K SALICON)

# sparse <repo-url> <cache-name> <path>...: shallow sparse checkout
sparse() {
  local url=$1 dir=.cache/$2; shift 2
  if [ ! -d "$dir/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --filter=blob:none --sparse "$url" "$dir"
  fi
  git -C "$dir" sparse-checkout set --no-cone "$@"
}

# collect <dest> <src-dir> <glob>: copy matching files into dest
collect() {
  mkdir -p "$1"
  find "$2" -maxdepth 1 -type f -name "$3" -exec cp {} "$1"/ \;
}

# salicon <repo> <name> <coco-split>: attention maps plus COCO images
salicon() {
  sparse "https://github.com/dogsteven/$1" "$1" '/*'
  mkdir -p "data/$2/images" "data/$2/fixations"
  find ".cache/$1" -name '*.png' -exec cp {} "data/$2/fixations/" \;
  for f in "data/$2/fixations"/*.png; do
    base=$(basename "$f" .png)
    [ -f "data/$2/images/$base.jpg" ] || echo "$base"
  done | xargs -P 8 -I{} curl -sSf --retry 3 -o "data/$2/images/{}.jpg" \
    "https://s3.amazonaws.com/images.cocodataset.org/$3/{}.jpg"
}

for name in "${names[@]}"; do
  echo "== $name"
  case $name in
    ECSSD)
      sparse https://github.com/total-black/U2Net-ECSSD-Evaluation ecssd /data/ECSSD/
      collect data/ECSSD/images .cache/ecssd/data/ECSSD/images '*.jpg'
      collect data/ECSSD/masks .cache/ecssd/data/ECSSD/ground_truth_mask '*.png'
      ;;
    PASCAL-S)
      sparse https://github.com/frankLeo123/Saliency_Matlab pascal /config/pascal/ /config/mask/
      collect data/PASCAL-S/images .cache/pascal/config/pascal '*.jpg'
      collect data/PASCAL-S/masks .cache/pascal/config/mask '*.png'
      ;;
    DUTS-TE)
      sparse https://github.com/beqooo09/SOD_Project sod /dataset/DUTS-TE/
      collect data/DUTS-TE/images .cache/sod/dataset/DUTS-TE/DUTS-TE-Image '*.jpg'
      collect data/DUTS-TE/masks .cache/sod/dataset/DUTS-TE/DUTS-TE-Mask '*.png'
      ;;
    MSRA10K)
      sparse https://github.com/beqooo09/SOD_Project sod /dataset/MSRA10K_Imgs_GT/
      collect data/MSRA10K/images .cache/sod/dataset/MSRA10K_Imgs_GT/Imgs '*.jpg'
      collect data/MSRA10K/masks .cache/sod/dataset/MSRA10K_Imgs_GT/Imgs '*.png'
      ;;
    SALICON)
      # Mouse-tracking attention maps for the 5000 SALICON val images (COCO val2014)
      salicon salicon-maps-val SALICON val2014
      ;;
    SALICON-TR)
      # The 10000 SALICON train images (COCO train2014), for training only
      salicon salicon-maps-train SALICON-TR train2014
      ;;
    DUTS-TR)
      # 10553 training images, for training only
      sparse https://github.com/beqooo09/SOD_Project sod /dataset/DUTS-TR/
      collect data/DUTS-TR/images .cache/sod/dataset/DUTS-TR/DUTS-TR-Image '*.jpg'
      collect data/DUTS-TR/masks .cache/sod/dataset/DUTS-TR/DUTS-TR-Mask '*.png'
      ;;
    LLM)
      # The images labelled by a vision LLM (eval/llm): 300 SALICON train
      # images with their attention maps, and 200 Open Images photos
      # (CC BY 2.0, attribution in eval/llm/openimages.csv)
      mkdir -p data/SALICON-LLM/images data/SALICON-LLM/fixations data/OPENIMAGES-LLM/images
      sparse https://github.com/dogsteven/salicon-maps-train salicon-maps-train '/*'
      for n in $(cat llm/salicon.txt); do
        cp ".cache/salicon-maps-train/train/$n.png" data/SALICON-LLM/fixations/
        [ -f "data/SALICON-LLM/images/$n.jpg" ] || echo "$n"
      done | xargs -P 8 -I{} curl -sSf --retry 3 -o "data/SALICON-LLM/images/{}.jpg" \
        "https://s3.amazonaws.com/images.cocodataset.org/train2014/{}.jpg"
      tail -n +2 llm/openimages.csv | cut -d, -f1 | while read -r id; do
        [ -f "data/OPENIMAGES-LLM/images/$id.jpg" ] || echo "$id"
      done | xargs -P 8 -I{} curl -sSf --retry 3 -o "data/OPENIMAGES-LLM/images/{}.jpg" \
        "https://s3.amazonaws.com/open-images-dataset/validation/{}.jpg"
      name=OPENIMAGES-LLM
      ;;
    OPENIMAGES-U)
      # 30000 unlabelled Open Images V7 photos (CC BY 2.0) for distillation:
      # every validation image except the LLM sample, then test images
      mkdir -p data/OPENIMAGES-U/images .cache/openimages
      list=.cache/openimages/unlabelled.txt
      if [ ! -f "$list" ]; then
        for split in validation test; do
          [ -f ".cache/openimages/$split.csv" ] || curl -sSf -o ".cache/openimages/$split.csv" \
            "https://storage.googleapis.com/openimages/2018_04/$split/$split-images-with-rotation.csv"
          tail -n +2 ".cache/openimages/$split.csv" | cut -d, -f1,2
        done | grep -v -F -f <(tail -n +2 llm/openimages.csv | cut -d, -f1) | awk 'NR <= 30000' > "$list"
      fi
      while IFS=, read -r id split; do
        [ -f "data/OPENIMAGES-U/images/$id.jpg" ] || echo "$split/$id"
      done < "$list" | xargs -P 16 -I{} sh -c 'curl -sSf --retry 3 -o "data/OPENIMAGES-U/images/$(basename {}).jpg" "https://s3.amazonaws.com/open-images-dataset/{}.jpg" || true'
      ;;
    *) echo "unknown dataset $name" >&2; exit 1 ;;
  esac
  echo "   $(ls data/$name/images | wc -l) images"
done
