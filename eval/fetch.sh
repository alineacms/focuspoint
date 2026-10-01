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
      # Mouse-tracking density maps for 1000 random COCO val2014 images
      sparse https://github.com/dogsteven/salicon-maps-val salicon '/*'
      mkdir -p data/SALICON/images data/SALICON/fixations
      maps=$(find .cache/salicon -name '*.png' | sort)
      echo "$maps" | python3 -c "import random,sys; l=sys.stdin.read().split(); random.seed(0); print('\n'.join(random.sample(l, 1000)))" |
        while read -r f; do
          base=$(basename "$f" .png)
          cp "$f" data/SALICON/fixations/
          [ -f "data/SALICON/images/$base.jpg" ] ||
            curl -sSf -o "data/SALICON/images/$base.jpg" "https://s3.amazonaws.com/images.cocodataset.org/val2014/$base.jpg"
        done
      ;;
    *) echo "unknown dataset $name" >&2; exit 1 ;;
  esac
  echo "   $(ls data/$name/images | wc -l) images"
done
