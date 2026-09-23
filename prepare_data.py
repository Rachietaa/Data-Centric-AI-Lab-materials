"""
prepare_data.py

Run this ONCE, before the lab notebook, to sort your downloaded images into
the data/images/dog, data/images/cat, data/images/bird structure the lab
notebook expects. It randomly samples about 4000 images total (roughly a
third each for dog, cat, and bird) so the lab stays fast to run -- edit
TOTAL_SAMPLE_TARGET below to change that.

This does NOT do the train/dev split (the notebook itself does that with
lab_utils.split_copy) -- this script only handles getting the raw
downloads into the right starting folders. The notebook's own 70/30
train/dev split, plus its imbalance step, run exactly as they did in the
original lab on whatever this script produces.

------------------------------------------------------------------------
MATCHES THIS FOLDER LAYOUT (edit the two paths in CONFIG if yours differs)
------------------------------------------------------------------------

Lab2/
  C1W2_Ungraded_Lab_Birds_Cats_Dogs.ipynb   <- the lab notebook
  lab_utils.py
  prepare_data.py                            <- this file
  train/
    cats/    (already-sorted cat photos)
    dogs/    (already-sorted dog photos)
  CUB_200_2011/
    ... (whatever the tgz/zip extracted, images live somewhere under here,
         possibly nested one level deeper if it extracted into a folder of
         the same name -- this script searches for it automatically)

After running this script you'll have:

Lab2/
  data/
    images/
      dog/
      cat/
      bird/

Then run the notebook -- its own cells build data_splits/train and
data_splits/dev from what's in data/images/.

Run it with:

    python prepare_data.py

from inside the Lab2 folder (or update the paths below to be absolute).
"""

import os
import shutil
import random

# ------------------------------------------------------------------
# CONFIG -- edit these two paths if your folder names differ.
# ------------------------------------------------------------------

# Folder that contains "cats" and "dogs" subfolders (already sorted)
CATS_DOGS_TRAIN_DIR = "./train"

# Folder you extracted the CUB-200-2011 download into. The script will
# look inside here (and one level deeper, in case of double-nesting from
# extraction) to find the "images" folder with the 200 species subfolders.
BIRDS_ROOT = "./CUB_200_2011"

# Where the sorted output should go -- this should match DATA_DIR in
# the lab notebook (default there is "./data")
OUTPUT_DIR = "./data/images"

# How many bird species folders to pull from. You don't need all 200 --
# a random handful gives good variety without a huge copy job.
NUM_BIRD_SPECIES_TO_USE = 20

# Total number of images to end up with across all three classes combined
# (split roughly evenly: dog, cat, and bird each get about a third).
# 4000 total keeps the lab fast to run while still being plenty of data
# for the train/dev split and imbalance step to work with.
TOTAL_SAMPLE_TARGET = 4000
PER_CLASS_TARGET = TOTAL_SAMPLE_TARGET // 3

random.seed(42)

# ------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_subfolder(root, target_name):
    """Case-insensitive search for a subfolder named target_name, directly
    under root or one level deeper (handles double-nested extractions)."""
    if not os.path.isdir(root):
        return None

    for entry in os.listdir(root):
        if entry.lower() == target_name.lower() and os.path.isdir(os.path.join(root, entry)):
            return os.path.join(root, entry)

    # not found directly -- look one level deeper
    for entry in os.listdir(root):
        candidate = os.path.join(root, entry)
        if os.path.isdir(candidate):
            for sub_entry in os.listdir(candidate):
                if sub_entry.lower() == target_name.lower() and os.path.isdir(os.path.join(candidate, sub_entry)):
                    return os.path.join(candidate, sub_entry)

    return None


def sort_cats_and_dogs(train_dir, output_dir, max_per_class):
    cat_out = os.path.join(output_dir, "cat")
    dog_out = os.path.join(output_dir, "dog")
    os.makedirs(cat_out, exist_ok=True)
    os.makedirs(dog_out, exist_ok=True)

    cat_source = find_subfolder(train_dir, "cats") or find_subfolder(train_dir, "cat")
    dog_source = find_subfolder(train_dir, "dogs") or find_subfolder(train_dir, "dog")

    if not cat_source or not dog_source:
        print(f"  Could not find both 'cats' and 'dogs' subfolders under {train_dir}")
        print("  (update CATS_DOGS_TRAIN_DIR at the top of this script)")
        return

    cat_files = [f for f in sorted(os.listdir(cat_source)) if f.lower().endswith(IMAGE_EXTENSIONS)]
    dog_files = [f for f in sorted(os.listdir(dog_source)) if f.lower().endswith(IMAGE_EXTENSIONS)]

    if max_per_class:
        cat_files = random.sample(cat_files, min(max_per_class, len(cat_files)))
        dog_files = random.sample(dog_files, min(max_per_class, len(dog_files)))

    for f in cat_files:
        shutil.copy2(os.path.join(cat_source, f), os.path.join(cat_out, f))
    for f in dog_files:
        shutil.copy2(os.path.join(dog_source, f), os.path.join(dog_out, f))

    print(f"  Copied {len(cat_files)} cat images from {cat_source} -> {cat_out}")
    print(f"  Copied {len(dog_files)} dog images from {dog_source} -> {dog_out}")


def sort_birds(birds_root, output_dir, num_species, max_total):
    bird_out = os.path.join(output_dir, "bird")
    os.makedirs(bird_out, exist_ok=True)

    images_dir = find_subfolder(birds_root, "images")
    if not images_dir:
        print(f"  Could not find an 'images' folder under {birds_root}")
        print("  (update BIRDS_ROOT at the top of this script)")
        return

    species_folders = sorted(
        f for f in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, f))
    )

    if not species_folders:
        print(f"  No species subfolders found inside {images_dir}")
        return

    chosen = species_folders
    if num_species and num_species < len(species_folders):
        chosen = random.sample(species_folders, num_species)

    per_species_cap = None
    if max_total:
        per_species_cap = max(1, max_total // len(chosen))

    copied = 0
    for species in sorted(chosen):
        species_path = os.path.join(images_dir, species)
        images = sorted(
            f for f in os.listdir(species_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if per_species_cap and per_species_cap < len(images):
            images = random.sample(images, per_species_cap)

        for img in images:
            # Prefix with species name so filenames don't collide
            # once everything is flattened into one folder
            new_name = f"{species}_{img}"
            shutil.copy2(
                os.path.join(species_path, img),
                os.path.join(bird_out, new_name),
            )
            copied += 1

    print(f"  Copied {copied} bird images from {len(chosen)} species (found at {images_dir}) -> {bird_out}")


def main():
    print("Sorting Dogs vs. Cats images (random sample)...")
    sort_cats_and_dogs(CATS_DOGS_TRAIN_DIR, OUTPUT_DIR, PER_CLASS_TARGET)

    print("\nSorting CUB-200-2011 bird images (random sample)...")
    sort_birds(BIRDS_ROOT, OUTPUT_DIR, NUM_BIRD_SPECIES_TO_USE, PER_CLASS_TARGET)

    print("\nDone. Check the counts below match what you expect:")
    for cls in ["dog", "cat", "bird"]:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        n = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
        print(f"  {cls}: {n} images in {cls_dir}")


if __name__ == "__main__":
    main()
