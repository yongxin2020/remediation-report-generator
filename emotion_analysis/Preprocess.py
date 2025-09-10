import os, glob, sys
from funcs import get_files_in_path, printProgressBar
import argparse


def main(inputPath, newPath):
    """
        Transform all audio files under a folder into wav PCM 16kHz 16bits signed-intege.

        example:
            python Preprocess.py --input "/data/sauvegarde/processed_theradia_data/private_dataset/M-subjects/M01E/M01E_seance1_0405/Audio/audio_segments_farfield" --output "/data/yongxin/THERADIA_MRG/Data/WavsProcessed/M01E/M01E_seance1_0405"
            python Preprocess.py --input "../../Data/Wavs" --output "../../Data/WavsProcessed"
    """
    
    path = os.path.join(inputPath, "**")  # "../PMDOM2FR/**/"
    theFiles = get_files_in_path(path)

    for i, filePath in enumerate(theFiles):
        # Making wav files
        fileNewPath = filePath.replace(inputPath, newPath)
        makeDirFor(fileNewPath)
        os.system('sox ' + filePath + ' -r 16000 -c 1 -b 16 -e signed-integer ' + fileNewPath)
        printProgressBar(i + 1, len(theFiles), prefix='Transforming Files:', suffix='Complete')


def makeDirFor(filePath):
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input wavs folder path")
    parser.add_argument('--output', '-o', help="output wavs folder path")
    args = parser.parse_args()
    Flag = False
    if args.input is None: Flag = True
    if args.output is None: Flag = True
    if Flag:
        print(main.__doc__)
        parser.print_help()
    else:
        main(args.input, args.output)
