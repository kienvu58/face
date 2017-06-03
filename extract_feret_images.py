import shutil
import os
import errno
import bz2


INPUT_FERET = "D:/colorferet/colorferet"
OUTPUT_FERET = "D:/colorferet"
DVDS = ["dvd1", "dvd2"]
DATA = "data"
GROUND_TRUTHS = "ground_truths"
IMAGES = ["images", "smaller", "thumbnails"]


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def copy(src, dest):
    try:
        copytree(src, dest)
    except OSError as e:
        # if the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print("Directory not copied. Error: {}".format(e))


def copy_ground_truths():
    for dvd in DVDS:
        in_gt_dir = os.path.join(INPUT_FERET, dvd, DATA, GROUND_TRUTHS)
        out_gt_dir = os.path.join(OUTPUT_FERET, DATA, GROUND_TRUTHS)
        copy(in_gt_dir, out_gt_dir)


def extract_images():
    # for each images folder
    # for each subject folder
    # for each bzip2 file
    # extract bzip2 to ppm in new folder
    for dvd in DVDS:
        in_dir = os.path.join(INPUT_FERET, dvd, DATA)
        out_dir = os.path.join(OUTPUT_FERET, DATA)
        for images_folder in IMAGES:
            in_if_dir = os.path.join(in_dir, images_folder)
            out_if_dir = os.path.join(out_dir, images_folder)
            for subject in os.listdir(in_if_dir):
                in_subj_dir = os.path.join(in_if_dir, subject)
                out_subj_dir = os.path.join(out_if_dir, subject)
                if not os.path.isdir(out_subj_dir):
                    os.makedirs(out_subj_dir)

                for bzfile in os.listdir(in_subj_dir):
                    name, ext = os.path.splitext(bzfile)
                    in_bzip2 = os.path.join(in_subj_dir, bzfile)
                    out_ppm = os.path.join(out_subj_dir, name)

                    in_file = bz2.open(in_bzip2)
                    with open(out_ppm, "wb") as f:
                        f.write(in_file.read())


def main():
    copy_ground_truths()
    extract_images()


if __name__ == "__main__":
    main()
